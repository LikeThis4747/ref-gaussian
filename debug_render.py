import inspect
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from utils.graphics_utils import linear_to_srgb


def _tonemap_env(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    return x / (x + 1.0)


def _tonemap_reinhard(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    return x / (1.0 + x + eps)


def _rgb_linear_to_display(x: torch.Tensor, env_hdr: bool, srgb: bool) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    if srgb:
        if env_hdr:
            x = _tonemap_reinhard(x)
        x = linear_to_srgb(x)
    else:
        x = torch.clamp(x, 0.0, 1.0)
    return torch.clamp(x, 0.0, 1.0)


def _ensure_three_channels(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] > 3:
        x = x[:3]
    return x


def _load_png_tensor(
    image_path: Path,
    size_hw: Tuple[int, int],
    *,
    single_channel: bool = False,
    rgb_only: bool = True,
) -> Optional[torch.Tensor]:
    if not image_path.exists():
        return None

    with Image.open(image_path) as pil_img:
        resample = Image.NEAREST if single_channel else Image.BILINEAR
        if pil_img.size != (size_hw[1], size_hw[0]):
            pil_img = pil_img.resize((size_hw[1], size_hw[0]), resample)

        arr = np.asarray(pil_img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[..., None]

    if single_channel:
        arr = arr[..., :1]
    elif rgb_only and arr.shape[-1] > 3:
        arr = arr[..., :3]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().cuda()
    return tensor


def _load_gt_maps(source_path: str, image_name: str, size_hw: Tuple[int, int]) -> Dict[str, Optional[torch.Tensor]]:
    root = Path(source_path)
    out: Dict[str, Optional[torch.Tensor]] = {
        "albedo": _load_png_tensor(root / "albedo" / f"{image_name}.png", size_hw, single_channel=False, rgb_only=True),
        "roughness": _load_png_tensor(root / "roughness" / f"{image_name}.png", size_hw, single_channel=True),
        "metallic": _load_png_tensor(root / "metallic" / f"{image_name}.png", size_hw, single_channel=True),
        "normal_cam_raw": _load_png_tensor(root / "normal" / f"{image_name}.png", size_hw, single_channel=False, rgb_only=True),
    }

    normal_cam_raw = out["normal_cam_raw"]
    if normal_cam_raw is not None:
        normal_cam = normal_cam_raw * 2.0 - 1.0
        normal_cam = F.normalize(normal_cam, dim=0)
        out["normal_cam"] = normal_cam
        out["normal_cam_vis"] = (normal_cam + 1.0) * 0.5
    else:
        out["normal_cam"] = None
        out["normal_cam_vis"] = None

    return out


def _load_gt_envmap(gaussians: Any, source_path: str) -> Optional[str]:
    env_dir = Path(source_path) / "envmap"
    exrs = sorted(env_dir.glob("*.exr"))
    if len(exrs) != 1:
        print(f"[WARN] Expected exactly one EXR under {env_dir}, found {len(exrs)}")
        return None

    env_path = str(exrs[0])
    for env in (gaussians.get_envmap, gaussians.get_envmap_2):
        setattr(env, "latlong_mode", "direction2")
        load_sig = inspect.signature(env.load)
        if "raw_values" in load_sig.parameters:
            env.load(env_path, flip_latlong=True, raw_values=True)
        else:
            env.load(env_path, flip_latlong=True)
        env.build_mips()
        env.base.requires_grad_(False)
    return env_path


def _select_cameras(scene: Any, split: str) -> List:
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()

    test_cams = scene.getTestCameras()
    if len(test_cams):
        return test_cams
    return scene.getTrainCameras()


def _match_cameras(cameras: List, image_name: Optional[str], camera_index: Optional[int], all_cameras: bool, max_cameras: Optional[int]) -> List:
    if image_name:
        key = Path(image_name).stem
        matched = [cam for cam in cameras if cam.image_name == key]
        if not matched:
            raise ValueError(f"Camera/image '{image_name}' not found.")
        return matched

    if all_cameras:
        if max_cameras is not None:
            return cameras[:max_cameras]
        return cameras

    if camera_index is None:
        camera_index = 0
    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(f"camera_index {camera_index} out of range for {len(cameras)} cameras.")
    return [cameras[camera_index]]


def _compose_linear_render(
    final_linear: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor,
    opt,
) -> torch.Tensor:
    final = _rgb_linear_to_display(final_linear, env_hdr=bool(getattr(opt, "env_HDR", False)), srgb=bool(getattr(opt, "srgb", False)))
    final = final + background[:, None, None] * (1.0 - alpha)
    if not (bool(getattr(opt, "env_HDR", False)) and bool(getattr(opt, "srgb", False))):
        final = torch.clamp(final, 0.0, 1.0)
    return final


def _compute_gt_reshade(
    viewpoint,
    render_pkg: Dict[str, torch.Tensor],
    gaussians: Any,
    background: torch.Tensor,
    opt,
    gt_maps: Dict[str, Optional[torch.Tensor]],
    use_gt_normal: bool,
) -> Dict[str, Optional[torch.Tensor]]:
    from utils.refl_utils import get_specular_color_surfel

    gt_albedo = gt_maps.get("albedo")
    gt_roughness = gt_maps.get("roughness")
    gt_metallic = gt_maps.get("metallic")
    if gt_albedo is None or gt_roughness is None or gt_metallic is None:
        return {
            "render": None,
            "diffuse": None,
            "specular": None,
            "direct_light": None,
            "indirect_light": None,
            "visibility": None,
            "albedo_used": None,
            "roughness_used": None,
            "metallic_used": None,
            "normal_used_cam_vis": None,
            "alpha_used": None,
        }

    normal_chw = render_pkg["rend_normal"]
    normal_cam_vis = render_pkg.get("rend_normal_cam", None)
    if isinstance(normal_cam_vis, torch.Tensor):
        normal_cam_vis = normal_cam_vis * 0.5 + 0.5
    if use_gt_normal and gt_maps.get("normal_cam") is not None:
        gt_normal_cam = gt_maps["normal_cam"]
        gt_normal_world = (
            gt_normal_cam.permute(1, 2, 0) @ viewpoint.world_view_transform[:3, :3].T
        ).permute(2, 0, 1)
        normal_chw = F.normalize(gt_normal_world, dim=0)
        normal_cam_vis = gt_maps.get("normal_cam_vis", None)

    indirect_light = render_pkg.get("indirect_light", None)
    c2w = np.linalg.inv(viewpoint.world_view_transform.T.detach().cpu().numpy())
    specular, extra_dict, diffuse = get_specular_color_surfel(
        gaussians.get_envmap,
        gt_albedo.permute(1, 2, 0),
        viewpoint.HWK,
        viewpoint.R,
        viewpoint.T,
        c2w,
        normal_chw.permute(1, 2, 0),
        render_pkg["rend_alpha"].permute(1, 2, 0),
        refl_strength=gt_metallic.permute(1, 2, 0),
        roughness=gt_roughness.permute(1, 2, 0),
        pc=gaussians,
        surf_depth=render_pkg["surf_depth"],
        indirect_light=indirect_light.permute(1, 2, 0) if isinstance(indirect_light, torch.Tensor) else None,
    )

    final_linear = diffuse + specular
    final = _compose_linear_render(final_linear, render_pkg["rend_alpha"], background, opt)
    diffuse_vis = _rgb_linear_to_display(diffuse, env_hdr=bool(getattr(opt, "env_HDR", False)), srgb=bool(getattr(opt, "srgb", False)))
    specular_vis = _rgb_linear_to_display(specular, env_hdr=bool(getattr(opt, "env_HDR", False)), srgb=bool(getattr(opt, "srgb", False)))

    out = {
        "render": final,
        "diffuse": diffuse_vis,
        "specular": specular_vis,
        "direct_light": None,
        "indirect_light": None,
        "visibility": None,
        "albedo_used": gt_albedo,
        "roughness_used": gt_roughness,
        "metallic_used": gt_metallic,
        "normal_used_cam_vis": normal_cam_vis,
        "alpha_used": render_pkg.get("rend_alpha", None),
    }
    if extra_dict is not None:
        direct = extra_dict.get("direct_light", None)
        indirect = extra_dict.get("indirect_light", None)
        visibility = extra_dict.get("visibility", None)
        out["direct_light"] = _rgb_linear_to_display(direct, env_hdr=bool(getattr(opt, "env_HDR", False)), srgb=bool(getattr(opt, "srgb", False))) if isinstance(direct, torch.Tensor) else None
        out["indirect_light"] = _rgb_linear_to_display(indirect, env_hdr=bool(getattr(opt, "env_HDR", False)), srgb=bool(getattr(opt, "srgb", False))) if isinstance(indirect, torch.Tensor) else None
        out["visibility"] = _ensure_three_channels(visibility)
    return out


def _build_env_vis(gaussians: Any, env_hdr: bool) -> torch.Tensor:
    # env_debug.png has two vertically stacked tiles:
    # 1. env1: the envmap visualized with the legacy direction1 convention.
    #    This is mainly for debugging convention mismatches and usually does
    #    not match the actual render path when read_envmap uses direction2.
    # 2. env2: the envmap visualized with the direction2 convention.
    #    This is the one that matches the current GT envmap loading path.
    env_dict = gaussians.render_env_map()
    env1 = env_dict["env1"].permute(2, 0, 1)
    env2 = env_dict["env2"].permute(2, 0, 1)
    if env_hdr:
        env1 = _tonemap_env(env1)
        env2 = _tonemap_env(env2)
    else:
        env1 = env1 / 10.0
        env2 = env2 / 10.0
    return make_grid([env1, env2], nrow=1, padding=10)


def _select_render_fn(gr, iteration: int, opt):
    initial_enabled = bool(getattr(opt, "initial", False))
    init_until_iter = int(getattr(opt, "init_until_iter", 0))
    volume_until_iter = int(getattr(opt, "volume_render_until_iter", 0))

    if initial_enabled and iteration <= init_until_iter:
        return gr.render_initial
    if iteration <= volume_until_iter:
        return gr.render_volume
    return gr.render_surfel


def _maybe_restore_ray_tracer(
    gaussians: Any,
    model_path: str,
    loaded_iteration: int,
    *,
    strict_before: bool = False,
    explicit_iteration: Optional[int] = None,
) -> Optional[int]:
    mesh_root = Path(model_path)
    mesh_iters: List[int] = []
    for p in mesh_root.glob("test_*.ply"):
        stem = p.stem
        try:
            mesh_iters.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    mesh_iters = sorted(set(mesh_iters))

    if explicit_iteration is not None:
        mesh_iters = [it for it in mesh_iters if it == explicit_iteration]
    elif strict_before:
        mesh_iters = [it for it in mesh_iters if it < loaded_iteration]
    else:
        mesh_iters = [it for it in mesh_iters if it <= loaded_iteration]

    if not mesh_iters:
        return None

    chosen_iter = mesh_iters[-1]
    gaussians.load_mesh_from_ply(model_path, chosen_iter)
    return chosen_iter


def _append_if_present(items: List[torch.Tensor], tensor: Optional[torch.Tensor]) -> None:
    if tensor is not None:
        items.append(torch.clamp(_ensure_three_channels(tensor), 0.0, 1.0))


def _blank_like(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(_ensure_three_channels(ref))


def _as_vis_chw(x: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return torch.clamp(_ensure_three_channels(x), 0.0, 1.0)
    return _blank_like(ref)


def _save_debug_grid(
    camera,
    render_pkg: Dict[str, torch.Tensor],
    gt_maps: Dict[str, Optional[torch.Tensor]],
    gt_reshade: Dict[str, Optional[torch.Tensor]],
    output_dir: Path,
    opt,
) -> None:
    # Grid layout is grouped as triplets by row: [GT | Original Render | GT_reshade].
    # For quantities without GT supervision (e.g. direct/indirect/visibility), GT
    # column is a black placeholder to keep alignment consistent.
    gt_rgb = torch.clamp(camera.original_image.detach(), 0.0, 1.0)
    render = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    gt_reshade_render = gt_reshade.get("render", None)

    direct_render = None
    if "direct_light" in render_pkg:
        direct_render = _rgb_linear_to_display(
            render_pkg["direct_light"],
            env_hdr=bool(getattr(opt, "env_HDR", False)),
            srgb=bool(getattr(opt, "srgb", False)),
        )
    indirect_render = None
    if "indirect_light" in render_pkg:
        indirect_render = _rgb_linear_to_display(
            render_pkg["indirect_light"],
            env_hdr=bool(getattr(opt, "env_HDR", False)),
            srgb=bool(getattr(opt, "srgb", False)),
        )

    normal_render_cam_vis = render_pkg.get("rend_normal_cam", None)
    if isinstance(normal_render_cam_vis, torch.Tensor):
        normal_render_cam_vis = normal_render_cam_vis * 0.5 + 0.5
    normal_reshade_cam_vis = gt_reshade.get("normal_used_cam_vis", None)

    triplet_rows: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = [
        (gt_rgb, render, gt_reshade_render),
        (gt_maps.get("albedo", None), render_pkg.get("albedo_map", render_pkg.get("base_color_map", None)), gt_reshade.get("albedo_used", None)),
        (gt_maps.get("roughness", None), render_pkg.get("roughness_map", None), gt_reshade.get("roughness_used", None)),
        (gt_maps.get("metallic", None), render_pkg.get("refl_strength_map", None), gt_reshade.get("metallic_used", None)),
        (None, render_pkg.get("diffuse_map", None), gt_reshade.get("diffuse", None)),
        (None, render_pkg.get("specular_map", None), gt_reshade.get("specular", None)),
        (gt_maps.get("normal_cam_vis", None), normal_render_cam_vis, normal_reshade_cam_vis),
        (None, render_pkg.get("rend_alpha", None), gt_reshade.get("alpha_used", None)),
        (None, direct_render, gt_reshade.get("direct_light", None)),
        (None, indirect_render, gt_reshade.get("indirect_light", None)),
        (None, render_pkg.get("visibility", None), gt_reshade.get("visibility", None)),
    ]

    tiles: List[torch.Tensor] = []
    for gt_item, render_item, reshade_item in triplet_rows:
        ref = gt_item if isinstance(gt_item, torch.Tensor) else (render_item if isinstance(render_item, torch.Tensor) else reshade_item)
        if not isinstance(ref, torch.Tensor):
            ref = gt_rgb
        tiles.extend([
            _as_vis_chw(gt_item, ref),
            _as_vis_chw(render_item, ref),
            _as_vis_chw(reshade_item, ref),
        ])

    grid = make_grid([torch.clamp(x, 0.0, 1.0) for x in tiles], nrow=3)
    save_image(grid, output_dir / f"{camera.image_name}_debug.png")


def _write_grid_layout(output_dir: Path) -> None:
    layout_lines = [
        "debug_render grid layout",
        "Columns are always: [GT | Original Render | GT_reshade]",
        "",
        "Row 1: RGB | render | gt_reshade render",
        "Row 2: albedo | predicted albedo | albedo used in gt_reshade",
        "Row 3: roughness | predicted roughness | roughness used in gt_reshade",
        "Row 4: metallic/refl_strength GT | predicted refl_strength | metallic used in gt_reshade",
        "Row 5: (blank) | diffuse_map | gt_reshade diffuse",
        "Row 6: (blank) | specular_map | gt_reshade specular",
        "Row 7: normal_cam GT vis | rend_normal_cam vis | normal vis used in gt_reshade",
        "Row 8: (blank) | rend_alpha | alpha used in gt_reshade",
        "Row 9: (blank) | direct_light | gt_reshade direct_light",
        "Row 10: (blank) | indirect_light | gt_reshade indirect_light",
        "Row 11: (blank) | visibility | gt_reshade visibility",
    ]
    (output_dir / "grid_layout.txt").write_text("\n".join(layout_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Debug render_surfel with GT comparisons")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration to load. Use -1 for the latest.")
    parser.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    parser.add_argument("--camera_index", type=int, default=None)
    parser.add_argument("--image_name", type=str, default=None, help="Camera/image stem, e.g. 0001")
    parser.add_argument("--all_cameras", action="store_true", default=False)
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_gt_envmap", dest="use_gt_envmap", action="store_true")
    parser.add_argument("--use_model_envmap", dest="use_gt_envmap", action="store_false")
    parser.set_defaults(use_gt_envmap=None)
    parser.add_argument("--with_gt_reshade", dest="with_gt_reshade", action="store_true")
    parser.add_argument("--without_gt_reshade", dest="with_gt_reshade", action="store_false")
    parser.set_defaults(with_gt_reshade=True)
    parser.add_argument("--use_gt_normal_in_reshade", action="store_true", default=True)
    parser.add_argument(
        "--reproduce_training_vis",
        action="store_true",
        default=False,
        help="Match train.py::save_training_vis timing more closely: use the same render function selection and the latest mesh strictly before the target iteration.",
    )
    parser.add_argument(
        "--mesh_iteration",
        type=int,
        default=None,
        help="Explicitly choose which test_XXXXXX.ply mesh to restore for the ray tracer.",
    )
    parser.add_argument(
        "--disable_restore_mesh",
        action="store_true",
        default=False,
        help="Do not restore any saved test_XXXXXX.ply mesh into the ray tracer.",
    )

    # Make cfg_args merging behave as intended: only explicitly provided CLI
    # flags should override values loaded from the saved experiment config.
    for action in parser._actions:
        if action.dest != "help":
            action.default = None

    args = get_combined_args(parser)
    custom_defaults = {
        "iteration": -1,
        "split": "auto",
        "camera_index": None,
        "image_name": None,
        "all_cameras": False,
        "max_cameras": None,
        "output_dir": None,
        "use_gt_envmap": None,
        "with_gt_reshade": True,
        "use_gt_normal_in_reshade": True,
        "reproduce_training_vis": False,
        "mesh_iteration": None,
        "disable_restore_mesh": False,
    }
    for key, value in custom_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if not args.model_path:
        raise ValueError("--model_path is required.")

    args.model_path = os.path.abspath(args.model_path)
    if args.output_dir is None:
        iter_tag = "latest" if int(args.iteration) == -1 else f"iter_{int(args.iteration):05d}"
        args.output_dir = os.path.join(args.model_path, "debug_render", iter_tag)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    import gaussian_renderer as gr
    from scene import GaussianModel, Scene

    opt.env_HDR = bool(getattr(args, "env_HDR", getattr(dataset, "env_HDR", False)))
    opt.read_envmap = bool(getattr(args, "read_envmap", getattr(dataset, "read_envmap", False)))
    opt.read_roughness = bool(getattr(args, "read_roughness", False))
    opt.read_metallic = bool(getattr(args, "read_metallic", False))
    opt.zero_metallic = bool(getattr(args, "zero_metallic", False) or getattr(args, "zero_metalic", False))

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=int(args.iteration), shuffle=False)
    loaded_iteration = int(scene.loaded_iter) if scene.loaded_iter is not None else int(args.iteration)
    opt.indirect = int(loaded_iteration > int(getattr(opt, "indirect_from_iter", 0)))

    if args.use_gt_envmap is None:
        args.use_gt_envmap = bool(getattr(args, "read_envmap", False))
    gt_env_path = None
    if bool(args.use_gt_envmap):
        gt_env_path = _load_gt_envmap(scene.gaussians, dataset.source_path)

    render_fn = _select_render_fn(gr, loaded_iteration, opt) if bool(args.reproduce_training_vis) else gr.render_surfel

    mesh_iter = None
    if bool(opt.indirect) and not bool(args.disable_restore_mesh):
        mesh_iter = _maybe_restore_ray_tracer(
            scene.gaussians,
            args.model_path,
            loaded_iteration,
            strict_before=bool(args.reproduce_training_vis),
            explicit_iteration=args.mesh_iteration,
        )

    cameras = _select_cameras(scene, args.split)
    if not cameras:
        raise RuntimeError(f"No cameras found for split='{args.split}'.")
    cameras = _match_cameras(cameras, args.image_name, args.camera_index, args.all_cameras, args.max_cameras)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[INFO] Loaded iteration: {loaded_iteration}")
    print(f"[INFO] Cameras to render: {len(cameras)}")
    print(f"[INFO] Render function: {render_fn.__name__}")
    if gt_env_path is not None:
        print(f"[INFO] Overriding envmap with GT: {gt_env_path}")
    if mesh_iter is not None:
        print(f"[INFO] Restored ray tracer mesh from test_{mesh_iter:06d}.ply")
    elif bool(args.disable_restore_mesh):
        print("[INFO] Ray tracer mesh restoration disabled by --disable_restore_mesh.")
    elif bool(opt.indirect):
        print("[WARN] No test_*.ply mesh found for ray tracer restoration; visibility may differ from training.")

    env_grid = _build_env_vis(scene.gaussians, env_hdr=bool(getattr(opt, "env_HDR", False)))
    save_image(env_grid, Path(args.output_dir) / "env_debug.png")
    _write_grid_layout(Path(args.output_dir))

    with torch.no_grad():
        for cam in cameras:
            print(f"[RENDER] {cam.image_name}")
            render_pkg = render_fn(cam, scene.gaussians, pipe, background, srgb=opt.srgb, opt=opt)
            gt_maps = _load_gt_maps(dataset.source_path, cam.image_name, (cam.image_height, cam.image_width))
            gt_reshade = {
                "render": None,
                "diffuse": None,
                "specular": None,
                "direct_light": None,
                "indirect_light": None,
                "visibility": None,
                "albedo_used": None,
                "roughness_used": None,
                "metallic_used": None,
                "normal_used_cam_vis": None,
                "alpha_used": None,
            }
            if bool(args.with_gt_reshade):
                gt_reshade = _compute_gt_reshade(
                    cam,
                    render_pkg,
                    scene.gaussians,
                    background,
                    opt,
                    gt_maps,
                    use_gt_normal=bool(args.use_gt_normal_in_reshade),
                )

            _save_debug_grid(cam, render_pkg, gt_maps, gt_reshade, Path(args.output_dir), opt)

    print(f"[DONE] Saved debug outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
