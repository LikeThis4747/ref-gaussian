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
import glob
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
import shlex  #my add
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
import cv2




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Unify HDR/read_envmap flags: renderer branches check `opt.env_HDR`.
    if not hasattr(opt, "env_HDR"):
        setattr(opt, "env_HDR", bool(getattr(dataset, "env_HDR", False)))
    if not hasattr(opt, "read_envmap"):
        setattr(opt, "read_envmap", bool(getattr(dataset, "read_envmap", False)))
    if bool(getattr(opt, "read_envmap", False)) and not bool(getattr(opt, "env_HDR", False)):
        raise ValueError("--read_envmap must be used together with --env_HDR")

    # my add: optional flag to force refl_strength (metallic-like term) to 0 during training
    if not hasattr(opt, "zero_metallic"):
        setattr(
            opt,
            "zero_metallic",
            bool(getattr(opt, "zero_metalic", False))
            or bool(getattr(dataset, "zero_metallic", False))
            or bool(getattr(dataset, "zero_metalic", False)),
        )

    def _find_single_gt_exr_envmap() -> str:
        env_dir = os.path.join(dataset.source_path, "envmap")
        if not os.path.isdir(env_dir):
            raise FileNotFoundError(f"envmap directory not found: {env_dir}")
        exrs = sorted(glob.glob(os.path.join(env_dir, "*.exr")))
        if len(exrs) != 1:
            raise FileNotFoundError(f"Expected exactly 1 .exr under {env_dir}, found {len(exrs)}: {exrs}")
        return exrs[0]

    def _freeze_envmap_optimizer_lrs():
        if not hasattr(gaussians, "optimizer") or gaussians.optimizer is None:
            return
        for group in gaussians.optimizer.param_groups:
            if group.get("name") in {"env", "env2"}:
                group["lr"] = 0.0

    def _load_and_freeze_gt_envmap():
        env_path = _find_single_gt_exr_envmap()
        # Load GT latlong EXR, flip horizontally to match the convention used in shadow_gaussian.
        setattr(gaussians.get_envmap, "latlong_mode", "direction2")
        gaussians.get_envmap.load(env_path, flip_latlong=True)
        gaussians.get_envmap.build_mips()
        gaussians.get_envmap.base.requires_grad_(False)

        setattr(gaussians.get_envmap_2, "latlong_mode", "direction2")
        gaussians.get_envmap_2.load(env_path, flip_latlong=True)
        gaussians.get_envmap_2.build_mips()
        gaussians.get_envmap_2.base.requires_grad_(False)

    def _load_gt_image_if_needed(viewpoint_cam, folder_name, cache_attr):  #my add
        cached = getattr(viewpoint_cam, cache_attr, None)
        if cached is False:
            return None
        if isinstance(cached, torch.Tensor):
            return cached

        folder_path = os.path.join(dataset.source_path, folder_name)
        if not os.path.isdir(folder_path):
            setattr(viewpoint_cam, cache_attr, False)
            return None

        gt_path = os.path.join(folder_path, f"{viewpoint_cam.image_name}.png")
        if not os.path.exists(gt_path):
            setattr(viewpoint_cam, cache_attr, False)
            return None

        try:
            from PIL import Image
            import torchvision.transforms.functional as tf

            with Image.open(gt_path) as pil_img:
                if pil_img.size != (viewpoint_cam.image_width, viewpoint_cam.image_height):
                    pil_img = pil_img.resize((viewpoint_cam.image_width, viewpoint_cam.image_height), Image.NEAREST)

                gt_map = tf.to_tensor(pil_img).cuda()
                if gt_map.shape[0] > 1:
                    gt_map = gt_map[:1, ...]
                gt_map = gt_map.float().clamp(0.0, 1.0)
                setattr(viewpoint_cam, cache_attr, gt_map)
                return gt_map
        except Exception as e:
            print(f"[ERROR] Failed to load {cache_attr} {gt_path}: {e}")
            setattr(viewpoint_cam, cache_attr, False)  #my add
            return None

    def _as_single_channel_map(x):  #my add
        if not isinstance(x, torch.Tensor):
            return None
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] > 1:
            x = x[:1, ...]
        return x.float()

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
    scene = Scene(dataset, gaussians)  # init all parameters(pos, scale, rot...) from pcds

    # Optional: load a fixed GT envmap from <source_path>/envmap/*.exr and freeze it.
    # This is applied before/after checkpoint restore to ensure it always overrides.
    if bool(getattr(opt, "read_envmap", False)):
        _load_and_freeze_gt_envmap()

    gaussians.training_setup(opt)
    if bool(getattr(opt, "read_envmap", False)):
        _freeze_envmap_optimizer_lrs()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if bool(getattr(opt, "read_envmap", False)):
            _load_and_freeze_gt_envmap()
            _freeze_envmap_optimizer_lrs()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

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


        # if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
        #     reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if bool(getattr(opt, "read_roughness", False)):  #my add
            _load_gt_image_if_needed(viewpoint_cam, "roughness", "gt_roughness")
        if bool(getattr(opt, "read_metallic", False)):  #my add
            _load_gt_image_if_needed(viewpoint_cam, "metallic", "gt_metallic")


        # Set render
        render = select_render_method(iteration, opt, initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        # print(f"gt image shape: {gt_image.shape}")
        # input()
        
        ######################
        # 尝试从相机获取 normal
        gt_normal = getattr(viewpoint_cam, "normal", None)
        dilated_edges = getattr(viewpoint_cam, "dilated_edges", None)

        # We cache missing normals as boolean False to avoid repeated disk checks.
        # Convert the sentinel back to None so downstream logic doesn't treat it as a Tensor.
        if gt_normal is False:
            gt_normal = None
            dilated_edges = None

        # 如果相机里没有 normal，且这是第一次遇到这个相机，我们尝试去硬盘加载
        if gt_normal is None:
            # 拼凑 normal 文件路径
            # dataset.source_path 是数据根目录
            # viewpoint_cam.image_name 是文件名 (如 r_0)
            normal_path = os.path.join(dataset.source_path, "normal", f"{viewpoint_cam.image_name}.png")
            
            # 检查文件是否存在
            if os.path.exists(normal_path):
                try:
                    from PIL import Image
                    import torchvision.transforms.functional as tf   
                    # 加载图片
                    with Image.open(normal_path) as pil_img:
                        # 确保和 GT Image 尺寸一致
                        if pil_img.size != (viewpoint_cam.image_width, viewpoint_cam.image_height):
                            pil_img = pil_img.resize((viewpoint_cam.image_width, viewpoint_cam.image_height), Image.NEAREST)
                        
                        # 转为 Tensor 并移到 CUDA
                        # 此时范围是 [0, 1]
                        gt_normal = tf.to_tensor(pil_img).cuda()
                        
                        # 只取前3个通道 (以防是 RGBA)
                        if gt_normal.shape[0] > 3:
                            gt_normal = gt_normal[:3, ...]
                        # 相机坐标系gt_normal y z 轴翻转
                        # gt_normal[1] = 1 - gt_normal[1]
                        # gt_normal[2] = 1 - gt_normal[2]

                        # 转为 NumPy 数组并转换为灰度图像
                        normal_np = gt_normal.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC
                        gray_image = cv2.cvtColor(normal_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图 
                        # Canny 边缘检测
                        edges = cv2.Canny((gray_image * 255).astype(np.uint8), threshold1=100, threshold2=200)
                        kernel = np.ones((11, 11), np.uint8)  # 使用一个 5x5 的内核
                        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                        dilated_edges = torch.from_numpy(dilated_edges).unsqueeze(0).cuda()
                        # cv2.imwrite('gray_image.jpg', (gray_image * 255).astype(np.uint8))
                        # cv2.imwrite('normal_image.jpg', (normal_np * 255).astype(np.uint8))
                        # cv2.imwrite('dilated_edges.jpg', dilated_edges)
                        # input()
                    
                    # 缓存到相机对象中
                    save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage, env_HDR=bool(getattr(opt, "env_HDR", False)))
                    setattr(viewpoint_cam, "normal", gt_normal)
                    setattr(viewpoint_cam, "dilated_edges", dilated_edges)
                    # print(gt_normal.dtype)
                    # print(gt_normal.shape)
                    # print(dilated_edges.dtype)
                    # print(dilated_edges.shape)
                    # input()
                    
                    # 打印一次成功信息 (仅限前几次)
                    if iteration < first_iter + 50: 
                        print(f"[LOADER] Loaded normal for {viewpoint_cam.image_name}")

                except Exception as e:
                    print(f"[ERROR] Failed to load normal {normal_path}: {e}")
            else:
                setattr(viewpoint_cam, "normal", False) 
        ######################

        gt_roughness = getattr(viewpoint_cam, "gt_roughness", None)  #my add
        gt_metallic = getattr(viewpoint_cam, "gt_metallic", None)  #my add
        if gt_roughness is False:  #my add
            gt_roughness = None
        if gt_metallic is False:  #my add
            gt_metallic = None

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        ######################
        if isinstance(gt_normal, torch.Tensor):
            gt_normal = gt_normal.cuda()
            if dilated_edges is None:
                # Shouldn't happen, but keep training robust.
                dilated_edges = torch.full((1, viewpoint_cam.image_height, viewpoint_cam.image_width), 255, device="cuda", dtype=torch.uint8)
            else:
                dilated_edges = dilated_edges.cuda()
            # rend_normal_cam = render_pkg['rend_normal_cam']
            rend_normal_cam = render_pkg.get('rend_normal_cam', None)
            rend_alpha = render_pkg['rend_alpha'] 

            gt_normal = gt_normal * 2.0 - 1.0 
            gt_normal = torch.nn.functional.normalize(gt_normal, dim=0) 
    
            # 求 Normal Cam GT 的 loss（相机空间），与 train_zjr1.py 的 normal_cam * gt_normal 对齐。
            # 原实现（保留作对照）：
            # normal_gt_error = 1 - (rend_normal_cam * gt_normal).sum(dim=0)[None]
            # normal_gt_loss = 0.5 * (normal_gt_error[dilated_edges < 255]).mean()
            # normal_gt_loss += 0.1 * (normal_gt_error[dilated_edges == 255]).mean()
            # total_loss += normal_gt_loss
            normal_gt_error = None
            if rend_normal_cam is not None:
                normal_gt_error = (1 - (rend_normal_cam * gt_normal).sum(dim=0))[None]

                lambda_gt_normal = float(getattr(opt, "lambda_gt_normal", 0.1))
                # Optional schedule (only if the option exists), matches train_zjr1.py behavior.
                it_normal = getattr(opt, "iterations_normal", None)
                if it_normal is not None:
                    lambda_gt_normal = lambda_gt_normal if iteration > int(it_normal) else 0.0

                mask0 = (dilated_edges < 255)
                mask1 = (dilated_edges == 255)
                normal_gt_loss = torch.zeros((), device=normal_gt_error.device, dtype=normal_gt_error.dtype)
                if bool(mask0.any()):
                    normal_gt_loss = normal_gt_loss + 0.5 * normal_gt_error[mask0].mean()
                if bool(mask1.any()):
                    normal_gt_loss = normal_gt_loss + 0.1 * normal_gt_error[mask1].mean()

                total_loss = total_loss + lambda_gt_normal * normal_gt_loss

            if bool(getattr(opt, "read_roughness", False)) and isinstance(gt_roughness, torch.Tensor):  #my add
                render_roughness = _as_single_channel_map(render_pkg.get("roughness_map", None))
                gt_roughness_map = _as_single_channel_map(gt_roughness)
                if render_roughness is not None and gt_roughness_map is not None:
                    render_roughness = render_roughness.to(gt_roughness_map.device)
                    lambda_gt_roughness = float(getattr(opt, "lambda_gt_roughness", 0.1))
                    total_loss = total_loss + lambda_gt_roughness * F.l1_loss(render_roughness, gt_roughness_map)

            if bool(getattr(opt, "read_metallic", False)) and isinstance(gt_metallic, torch.Tensor):  #my add
                render_metallic = _as_single_channel_map(render_pkg.get("refl_strength_map", None))
                gt_metallic_map = _as_single_channel_map(gt_metallic)
                if render_metallic is not None and gt_metallic_map is not None:
                    render_metallic = render_metallic.to(gt_metallic_map.device)
                    lambda_gt_metallic = float(getattr(opt, "lambda_gt_metallic", 0.1))
                    total_loss = total_loss + lambda_gt_metallic * F.l1_loss(render_metallic, gt_metallic_map)

            alpha_error = torch.abs(1 - rend_alpha)
            lambda_alpha = getattr(opt, "lambda_alpha", 0.1)
            alpha_loss = lambda_alpha * (alpha_error).mean()
            total_loss += alpha_loss

            if iteration % 1000 == 0 and isinstance(normal_gt_error, torch.Tensor) and isinstance(rend_normal_cam, torch.Tensor):
                # 定义保存路径
                debug_dir = os.path.join(args.model_path, "debug_normal_vis")
                os.makedirs(debug_dir, exist_ok=True)
                
                # 准备可视化数据: [-1, 1] -> [0, 1]
                # gt_normal_vis: Ground Truth Normal
                gt_vis = (gt_normal.detach() + 1.0) * 0.5
                gt_vis = gt_vis.clamp(0.0, 1.0)
                
                # render_normal_vis: Rendered Normal
                rend_vis = (rend_normal_cam.detach() + 1.0) * 0.5
                rend_vis = rend_vis.clamp(0.0, 1.0)
                
                # error_vis: Error Map (越亮误差越大)
                # normal_gt_error 是 [1, H, W]，为了可视化方便可以不用 repeat，save_image 会处理
                err_vis = normal_gt_error.detach().clamp(0.0, 1.0)
                
                # 拼接图片方便对比 (GT | Render | Error)
                # cat 在宽度方向拼接 (dim=2)
                combined_vis = torch.cat([gt_vis, rend_vis, err_vis.repeat(3, 1, 1)], dim=2)
                
                # 文件名: iter_camName.png
                fname = f"iter{iteration:05d}_{viewpoint_cam.image_name}.png"
                save_path = os.path.join(debug_dir, fname)
                
                import torchvision
                torchvision.utils.save_image(combined_vis, save_path)
                print(f"[DEBUG] Saved normal visualization to {save_path}")
        ######################

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
                    env_HDR=bool(getattr(opt, "env_HDR", False)),
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

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

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

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    # gaussians.reset_opacity0()
                    # gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        # gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        # gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    # if 'ref_real' in dataset.source_path:
                    #     mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    # else:
                    #     depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                    #     voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                    #     sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                    #     mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

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




def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage, env_HDR: bool = False):
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
            if bool(getattr(opt, "read_roughness", False)) and isinstance(getattr(viewpoint_cam, "gt_roughness", None), torch.Tensor):  #my add
                visualization_list.insert(7, getattr(viewpoint_cam, "gt_roughness").repeat(3, 1, 1))
            if bool(getattr(opt, "read_metallic", False)) and isinstance(getattr(viewpoint_cam, "gt_metallic", None), torch.Tensor):  #my add
                visualization_list.insert(6, getattr(viewpoint_cam, "gt_metallic").repeat(3, 1, 1))
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
                # render_pkg["rend_normal_cam"] * 0.5 + 0.5,
                error_map, 
            ]
            if bool(getattr(opt, "read_roughness", False)) and isinstance(getattr(viewpoint_cam, "gt_roughness", None), torch.Tensor):  #my add
                visualization_list.insert(7, getattr(viewpoint_cam, "gt_roughness").repeat(3, 1, 1))
            if bool(getattr(opt, "read_metallic", False)) and isinstance(getattr(viewpoint_cam, "gt_metallic", None), torch.Tensor):  #my add
                visualization_list.insert(6, getattr(viewpoint_cam, "gt_metallic").repeat(3, 1, 1))
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
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

            print(torch.max(env_dict["env1"]))
            print(torch.max(env_dict["env2"]))
            # input()
            if bool(env_HDR):
                def _tonemap(x):
                    x = torch.clamp(x, min=0.0)
                    return x / (x + 1.0)

                grid = [
                    _tonemap(env_dict["env1"]).permute(2, 0, 1),
                    _tonemap(env_dict["env2"]).permute(2, 0, 1),
                ]
            else:
                grid = [
                    env_dict["env1"].permute(2, 0, 1) / 10.0,
                    env_dict["env2"].permute(2, 0, 1) / 10.0,
                ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # my add: record run command (ref: shadow_gaussian/train_debug.py)
    try:
        cmd_path = os.path.join(args.model_path, "run_command.txt")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cmd_parts = [sys.executable] + sys.argv
        cmd_str = " ".join(shlex.quote(x) for x in cmd_parts)
        if cuda_visible != "":
            cmd_str = f"CUDA_VISIBLE_DEVICES={cuda_visible} " + cmd_str
        with open(cmd_path, "a") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] cwd={os.getcwd()}\n")
            f.write(cmd_str + "\n\n")
    except Exception as e:
        print(f"[WARN] record run command failed: {e}")

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
    parser.add_argument("--zero_metallic", action='store_true', default=False)  #my add
    parser.add_argument("--zero_metalic", action='store_true', default=False, help="(deprecated) use --zero_metallic")  #my add
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # 获取当前时间并格式化为精确到分钟
        current_time = datetime.now().strftime('%m%d_%H%M')
        # 获取args.source_path的最后一个子目录名
        last_subdir = os.path.basename(os.path.normpath(args.source_path))

        
        # 生成带有时间戳和opt属性的简洁输出目录
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Forward flags into `opt` because renderer branches check `opt.env_HDR`.
    opt.env_HDR = bool(getattr(args, "env_HDR", getattr(dataset, "env_HDR", False)))
    opt.read_envmap = bool(getattr(args, "read_envmap", getattr(dataset, "read_envmap", False)))
    opt.read_roughness = bool(getattr(args, "read_roughness", False))  #my add
    opt.read_metallic = bool(getattr(args, "read_metallic", False))  #my add
    opt.zero_metallic = bool(getattr(args, "zero_metallic", False) or getattr(args, "zero_metalic", False))  #my add

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path)

    # All done
    print("\nTraining complete.")