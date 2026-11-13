#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import open3d as o3d
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_surfel, render_initial, render_volume, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False






def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, model_path, debug_from=None,
             joint_args=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    load_model_path = None if joint_args is None else joint_args.ref_model_path
    load_iteration = None if joint_args is None else joint_args.ref_iteration
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=load_iteration,
        load_model_path=load_model_path,
        shuffle=load_model_path is None,
    )  # init all parameters(pos, scale, rot...) from pcds
    if scene.loaded_iter is not None:
        first_iter = scene.loaded_iter
        if first_iter >= opt.indirect_from_iter:
            opt.indirect = 1
        if opt.indirect:
            gaussians.load_mesh_from_ply(scene.load_model_path, first_iter)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

    joint_model = None
    patch_optimizer = None
    joint_camera_sampler = None

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0


    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)


        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1


        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                envmap.build_mips()

        # Set render
        render = select_render_method(iteration, opt, initial_stage)
        joint_step = joint_args is not None and render is render_surfel

        if joint_step:
            if joint_model is None:
                # [UPSTREAM Ref-Gaussian] af7691a train.py::training
                # [REF-MV MODIFICATION] Lazily allocate joint state at the first
                # surfel step, after any upstream initial/volume topology changes.
                from ref_mv.joint import CameraSampler, initialize_joint
                if getattr(joint_args, "joint_envmap_lr", None) is not None:
                    for param_group in gaussians.optimizer.param_groups:
                        if param_group["name"] in ("env", "env2"):
                            param_group["lr"] = joint_args.joint_envmap_lr
                joint_model, patch_optimizer = initialize_joint(
                    gaussians=gaussians,
                    mv_checkpoint=joint_args.mv_checkpoint,
                    feature_dim=joint_args.feature_dim,
                    feature_lr=joint_args.gs_feature_lr,
                    patch_lr=joint_args.gs_patch_lr,
                    joint_checkpoint=joint_args.joint_checkpoint,
                    expected_iteration=scene.loaded_iter,
                )
                joint_camera_sampler = CameraSampler(
                    scene.getTrainCameras(),
                    joint_args.num_views,
                    joint_args.neighbor_k,
                )

            gaussians.optimizer.zero_grad(set_to_none=True)
            patch_optimizer.zero_grad(set_to_none=True)
            joint_cameras = joint_camera_sampler.sample()
            joint_output = joint_model(
                joint_cameras,
                pipe,
                background,
                opt,
                # [REF-MV MODIFICATION] Keep the old second full render while
                # upstream densification still consumes its screen-space stats.
                reuse_rasterization=iteration >= opt.densify_until_iter,
            )
            joint_losses = []
            joint_tb = []
            for camera, package in zip(joint_cameras, joint_output["render_pkgs"]):
                view_loss, view_tb = calculate_loss(
                    camera, gaussians, package, opt, iteration
                )
                joint_losses.append(view_loss)
                joint_tb.append(view_tb)
            total_loss = torch.stack(joint_losses).mean()

            def mean_joint_stat(name):
                values = [stats[name] for stats in joint_tb]
                if torch.is_tensor(values[0]):
                    return torch.stack(values).mean()
                return sum(values) / len(values)

            dist_loss = mean_joint_stat("loss_dist")
            normal_loss = mean_joint_stat("loss_normal_render_depth")
            loss = mean_joint_stat("loss0")
            Ll1 = mean_joint_stat("loss_l1")
            normal_smooth_loss = mean_joint_stat("loss_normal_smooth")
            depth_smooth_loss = mean_joint_stat("loss_depth_smooth")

            viewpoint_cam = joint_cameras[0]
            render_pkg = joint_output["render_pkgs"][0]
            densification_pkgs = joint_output["render_pkgs"]
            densification_grad_scale = len(joint_cameras)
            render = joint_model.render_camera
        else:
            # Pick a random Camera using the original Ref stack semantics.
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background,
                srgb=opt.srgb, opt=opt,
            )
            total_loss, tb_dict = calculate_loss(
                viewpoint_cam, gaussians, render_pkg, opt, iteration
            )
            dist_loss = tb_dict["loss_dist"]
            normal_loss = tb_dict["loss_normal_render_depth"]
            loss = tb_dict["loss0"]
            Ll1 = tb_dict["loss_l1"]
            normal_smooth_loss = tb_dict["loss_normal_smooth"]
            depth_smooth_loss = tb_dict["loss_depth_smooth"]
            densification_pkgs = [render_pkg]
            densification_grad_scale = 1

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(
                    viewpoint_cam,
                    gaussians,
                    background,
                    render,
                    pipe,
                    opt,
                    iteration,
                    initial_stage,
                    joint_output=joint_output if joint_step else None,
                    joint_model=joint_model if joint_step else None,
                    joint_cameras=joint_cameras if joint_step else None,
                )

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                if joint_model is not None:
                    from ref_mv.joint import save_joint_checkpoint
                    save_joint_checkpoint(
                        scene.model_path, iteration, joint_model, gaussians
                    )

            # [UPSTREAM Ref-Gaussian] af7691a train.py::training
            # [REF-MV MODIFICATION] Initialize this per iteration so a run loaded
            # at/after densify_until_iter does not inherit or miss stale state.
            HAS_RESET0 = False

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                for package in densification_pkgs:
                    visibility_filter = package["visibility_filter"]
                    radii = package["radii"]
                    viewspace_point_tensor = package["viewspace_points"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    if densification_grad_scale != 1:
                        viewspace_point_tensor.grad.mul_(densification_grad_scale)
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    if 'ref_real' in dataset.source_path:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if patch_optimizer is not None:
                    patch_optimizer.step()
                    patch_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        iteration += 1








# ============================================================
# Utils for training


def select_render_method(iteration, opt, initial_stage):

    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    else:   
        render = render_surfel

    return render


def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr




def _material_vis_image(tensor, normal=False):
    tensor = tensor.detach().float()
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if normal:
        tensor = tensor[:3] * 0.5 + 0.5
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor[:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_material_training_vis(
        viewpoint_cam, gaussians, background, pipe, opt, iteration,
        joint_output, joint_model, joint_cameras,
    ):
    """Save Ref-only and MV-material replacement results for a joint step."""
    import matplotlib.pyplot as plt
    from gaussian_renderer import rasterize_surfel, shade_surfel

    rasterized = rasterize_surfel(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        gaussian_feature=gaussians.get_gs_feature,
    )
    ref_pkg = shade_surfel(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        rasterized,
        srgb=opt.srgb,
        opt=opt,
    )
    material_maps = {
        name: values[0, 0].detach()
        for name, values in joint_output["material"].items()
    }
    mv_pkg = shade_surfel(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        rasterized,
        srgb=opt.srgb,
        opt=opt,
        material_maps=material_maps,
    )

    images = {
        "GT RGB": viewpoint_cam.original_image[:3],
        "Ref RGB": ref_pkg["render"],
        "MV Replaced RGB": mv_pkg["render"],
        "GS Feature (1D)": joint_output["feature_maps"][0],
        "Ref Albedo": ref_pkg["base_color_map"],
        "MV Albedo": mv_pkg["base_color_map"],
        "Ref Diffuse": ref_pkg["diffuse_map"],
        "MV Diffuse": mv_pkg["diffuse_map"],
        "Ref Reflection": ref_pkg["refl_strength_map"],
        "MV Metallic": mv_pkg["refl_strength_map"],
        "Ref Specular": ref_pkg["specular_map"],
        "MV Specular": mv_pkg["specular_map"],
        "Ref Roughness": ref_pkg["roughness_map"],
        "MV Roughness": mv_pkg["roughness_map"],
        "Depth": visualize_depth(ref_pkg["surf_depth"]),
        "Normal": ref_pkg["surf_normal"],
    }
    layout = [
        ["GT RGB", "Ref RGB", "MV Replaced RGB", "GS Feature (1D)"],
        ["Ref Albedo", "MV Albedo", "Ref Diffuse", "MV Diffuse"],
        ["Ref Reflection", "MV Metallic", "Ref Specular", "MV Specular"],
        ["Ref Roughness", "MV Roughness", "Depth", "Normal"],
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(f"iteration {iteration} | Ref vs MV material", fontsize=14)
    for row, names in enumerate(layout):
        for col, name in enumerate(names):
            axes[row, col].axis("off")
            if name == "GS Feature (1D)":
                feature = images[name].detach().float().squeeze().cpu()
                scale = torch.quantile(feature.abs().flatten(), 0.99).item()
                scale = max(scale, 1e-8)
                axes[row, col].set_title(
                    f"{name} | min {feature.min():.3g}, max {feature.max():.3g}",
                    fontsize=10,
                    pad=5,
                )
                axes[row, col].imshow(
                    feature.numpy(), cmap="coolwarm", vmin=-scale, vmax=scale
                )
            else:
                axes[row, col].set_title(name, fontsize=10, pad=5)
                mode = name == "Normal"
                axes[row, col].imshow(_material_vis_image(images[name], normal=mode))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(args.visualize_path, f"{iteration:06d}_material.png"), dpi=150)
    plt.close(fig)

    input_images = torch.stack([camera.original_image[:3] for camera in joint_cameras])
    zero_features = torch.zeros_like(joint_output["feature_maps"])
    zero_tokens = joint_model.gs_patch_embed(zero_features).flatten(2).transpose(1, 2)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        init_material = joint_model.mvinverse(
            input_images.unsqueeze(0),
            hidden_patch_residual=zero_tokens.unsqueeze(0),
            output_names=("albedo", "metallic", "roughness"),
        )
    init_images = [
        ("Input RGB", input_images[0]),
        ("Init MV Albedo", init_material["albedo"][0, 0]),
        ("Init MV Metallic", init_material["metallic"][0, 0]),
        ("Init MV Roughness", init_material["roughness"][0, 0]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"iteration {iteration} | MVInverse with zero-initialized GS feature", fontsize=14)
    for axis, (name, image) in zip(axes, init_images):
        axis.axis("off")
        axis.set_title(name, fontsize=10, pad=5)
        axis.imshow(_material_vis_image(image))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(
        os.path.join(args.visualize_path, f"{iteration:06d}_mvinverse_init.png"),
        dpi=150,
    )
    plt.close(fig)


def save_training_vis(
        viewpoint_cam,
        gaussians,
        background,
        render_fn,
        pipe,
        opt,
        iteration,
        initial_stage,
        joint_output=None,
        joint_model=None,
        joint_cameras=None,
    ):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg["base_color_map"], 
                render_pkg["diffuse_map"],      
                render_pkg["specular_map"],  
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]), 
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5,  
                error_map, 
            ]
  

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

        if joint_output is not None:
            save_material_training_vis(
                viewpoint_cam,
                gaussians,
                background,
                pipe,
                opt,
                iteration,
                joint_output,
                joint_model,
                joint_cameras,
            )

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test





# ============================================================================
# Main function


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{current_time}-{last_subdir}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    # All done
    print("\nTraining complete.")
