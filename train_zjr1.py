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
import gc
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
# Workaround for some numpy/accelerate combos:
# accelerate may access `numpy._core.multiarray` as an attribute on `numpy._core`.
# Importing the submodule once ensures the attribute is present.
try:
    import numpy._core.multiarray  # noqa: F401
except Exception:
    pass

# Ensure repository root is on sys.path so sibling packages like `utils`
# and `scene` are importable when running this script directly.
import sys
import pathlib
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import kornia
from typing import cast
from scene.light import LightMLPBase, LightMLP1D
import subprocess
import random
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo $CUDA_VISIBLE_DEVICES')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
from utils.mesh2_utils import GaussianExtractor as RefGaussianExtractor, post_process_mesh as ref_post_process_mesh

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import contextlib
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, per_channel_block_scale_l1 ,tv_l1_loss, quantile_loss, weighted_l1_loss, weighted_ssim_loss, weighted_ssim_map, logit_loss, calculate_loss, first_order_edge_aware_loss
from gaussian_renderer import prefilter_voxel, render, network_gui, get_cam_pos, get_world_pos, render_surfel
from render_diffuse_indirect import render_diffuse_indirect
# Precompute helpers (moved to separate module)
from precompute_flashBRDF import (
    precompute_indirect_flash,
    precompute_direct_flash_from_files,
    load_precomp_cache,    
)
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from scene.save_load_gaussian_zjr_2 import (
    load_gaussian_npz_snapshot,
    save_gaussian_npz_snapshot,
)

import open3d as o3d

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, linear2srgb, srgb2linear
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from text_to_image.train_text_to_image import my_main
from utils.tb_utils import LossTensorboardLogger
import imageio.v2 as imageio

import cv2

# Global caches for specular precompute interpolation
ind_spec_L0_cache = {}
ind_spec_L1_cache = {}
ind_diffuse_cache = {}

direct_spec_L0_cache = {}
direct_spec_L1_cache = {}
direct_diffuse_cache = {}

rough_bins_tensor = torch.tensor([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], device="cuda")

def _interp_spec(cam_idx: int, rough_map: torch.Tensor):
    """Interpolate precomputed spec transport L0/L1 for a given roughness map.

    rough_map: [1,H,W] or [H,W], values in [0,1]. Returns (L0,H,W),(L1,H,W) in linear domain or (None,None) if missing.
    """
    if cam_idx not in ind_spec_L0_cache or cam_idx not in ind_spec_L1_cache or cam_idx not in direct_spec_L0_cache or cam_idx not in direct_spec_L1_cache:
        return None, None, None, None
    ind_L0_list = ind_spec_L0_cache[cam_idx]
    ind_L1_list = ind_spec_L1_cache[cam_idx]
    direct_L0_list = direct_spec_L0_cache[cam_idx]
    direct_L1_list = direct_spec_L1_cache[cam_idx]
    if len(ind_L0_list) != 6 or len(ind_L1_list) != 6 or len(direct_L0_list) != 6 or len(direct_L1_list) != 6:
        return None, None, None, None
    def _unify_channels(t_list, target_c: int):
        out = []
        for t in t_list:
            if not torch.is_tensor(t) or t.dim() != 3:
                return None
            if t.shape[0] == target_c:
                out.append(t)
            elif t.shape[0] == 1 and target_c > 1:
                out.append(t.expand(target_c, *t.shape[1:]))
            else:
                return None
        return out

    def _move_list_to_device(t_list, device: torch.device):
        if device is None:
            return t_list
        out = []
        for t in t_list:
            if torch.is_tensor(t) and t.device != device:
                out.append(t.to(device=device, non_blocking=True))
            else:
                out.append(t)
        return out

    # Support both gray ([1,H,W]) and RGB ([3,H,W]) cached EXRs.
    c_candidates = []
    for lst in (ind_L0_list, ind_L1_list, direct_L0_list, direct_L1_list):
        if not lst or (not torch.is_tensor(lst[0])) or lst[0].dim() != 3:
            return None, None, None, None
        c_candidates.append(int(lst[0].shape[0]))
    C = max(c_candidates)
    if C not in (1, 3):
        return None, None, None, None

    ind_L0_u = _unify_channels(ind_L0_list, C)
    ind_L1_u = _unify_channels(ind_L1_list, C)
    direct_L0_u = _unify_channels(direct_L0_list, C)
    direct_L1_u = _unify_channels(direct_L1_list, C)
    if ind_L0_u is None or ind_L1_u is None or direct_L0_u is None or direct_L1_u is None:
        return None, None, None, None

    device = rough_map.device if torch.is_tensor(rough_map) else None
    ind_L0_u = _move_list_to_device(ind_L0_u, device)
    ind_L1_u = _move_list_to_device(ind_L1_u, device)
    direct_L0_u = _move_list_to_device(direct_L0_u, device)
    direct_L1_u = _move_list_to_device(direct_L1_u, device)

    # Stack to [6,C,H,W] on device
    ind_L0_stack = torch.stack(ind_L0_u, dim=0)  # [6,C,H,W]
    ind_L1_stack = torch.stack(ind_L1_u, dim=0)  # [6,C,H,W]
    direct_L0_stack = torch.stack(direct_L0_u, dim=0)  # [6,C,H,W]
    direct_L1_stack = torch.stack(direct_L1_u, dim=0)  # [6,C,H,W]

    rough = rough_map.to(ind_L0_stack.device)
    if rough.dim() == 3 and rough.shape[0] == 1:
        rough = rough.squeeze(0)
    rough = torch.clamp(rough, 0.1, rough_bins_tensor.max()) # TODO rough_bins_tensor.min()
    H, W = rough.shape[-2], rough.shape[-1]
    rough_flat = rough.reshape(-1)

    # Find lower/upper bin indices
    idx_hi = torch.searchsorted(rough_bins_tensor, rough_flat, right=False)
    idx_hi = torch.clamp(idx_hi, 0, len(rough_bins_tensor) - 1)
    idx_lo = torch.clamp(idx_hi - 1, 0, len(rough_bins_tensor) - 1)
    # Ensure hi >= lo
    idx_hi = torch.maximum(idx_hi, idx_lo)

    bins_lo = rough_bins_tensor[idx_lo]
    bins_hi = rough_bins_tensor[idx_hi]
    denom = torch.clamp(bins_hi - bins_lo, min=1e-6)
    t = (rough_flat - bins_lo) / denom

    # Flatten stacks to gather: [6,C,N] -> [6,N,C]
    N = rough_flat.numel()
    ind_L0_flat = ind_L0_stack.view(6, C, -1).permute(0, 2, 1)  # [6,N,C]
    ind_L1_flat = ind_L1_stack.view(6, C, -1).permute(0, 2, 1)  # [6,N,C]
    direct_L0_flat = direct_L0_stack.view(6, C, -1).permute(0, 2, 1)  # [6,N,C]
    direct_L1_flat = direct_L1_stack.view(6, C, -1).permute(0, 2, 1)  # [6,N,C]

    idx_lo_exp = idx_lo.view(1, N, 1).expand(1, N, C)
    idx_hi_exp = idx_hi.view(1, N, 1).expand(1, N, C)

    ind_L0_lo = torch.gather(ind_L0_flat, 0, idx_lo_exp)[0]  # [N,3]
    ind_L0_hi = torch.gather(ind_L0_flat, 0, idx_hi_exp)[0]
    ind_L1_lo = torch.gather(ind_L1_flat, 0, idx_lo_exp)[0]
    ind_L1_hi = torch.gather(ind_L1_flat, 0, idx_hi_exp)[0]

    direct_L0_lo = torch.gather(direct_L0_flat, 0, idx_lo_exp)[0]  # [N,3]
    direct_L0_hi = torch.gather(direct_L0_flat, 0, idx_hi_exp)[0]
    direct_L1_lo = torch.gather(direct_L1_flat, 0, idx_lo_exp)[0]
    direct_L1_hi = torch.gather(direct_L1_flat, 0, idx_hi_exp)[0]

    t_exp = t.view(N, 1)
    ind_L0_interp = (1.0 - t_exp) * ind_L0_lo + t_exp * ind_L0_hi
    ind_L1_interp = (1.0 - t_exp) * ind_L1_lo + t_exp * ind_L1_hi

    direct_L0_interp = (1.0 - t_exp) * direct_L0_lo + t_exp * direct_L0_hi
    direct_L1_interp = (1.0 - t_exp) * direct_L1_lo + t_exp * direct_L1_hi

    ind_L0_interp = ind_L0_interp.permute(1, 0).view(C, H, W)
    ind_L1_interp = ind_L1_interp.permute(1, 0).view(C, H, W)
    direct_L0_interp = direct_L0_interp.permute(1, 0).view(C, H, W)
    direct_L1_interp = direct_L1_interp.permute(1, 0).view(C, H, W)

    return ind_L0_interp, ind_L1_interp, direct_L0_interp, direct_L1_interp


def ensure_bchw(x):
    # [C,H,W] -> [1,C,H,W]
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x

def _ensure_chw(tensor: torch.Tensor):
    """Return tensor in [C,H,W] layout.

    Handles common permutations used across this file: [C,H,W], [H,W,C], [W,C,H], [W,1,H], etc.
    """
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return tensor
    if tensor.dim() == 3:
        s0, s1, s2 = tensor.shape
        # already C,H,W or 1,H,W
        if s0 == 3 or s0 == 1:
            return tensor
        # W,C,H -> C,H,W
        if s1 == 3 or s1 == 1:
            return tensor.permute(1, 2, 0)
        # H,W,C or H,W,1 -> C,H,W
        if s2 == 3 or s2 == 1:
            return tensor.permute(2, 0, 1)
    if tensor.dim() == 2:
        # H,W -> 1,H,W
        return tensor.unsqueeze(0)
    return tensor


def _to_1hw_scalar_map(x: torch.Tensor | None, *, ref_chw: torch.Tensor) -> torch.Tensor | None:
    """Convert a renderer map to [1,H,W] on the same device as ref.

    Accepts common layouts: [1,H,W], [H,W,1], [H,W], [C,H,W], [B,1,H,W].
    If C>1, averages to a single-channel scalar map.
    """
    if x is None:
        return None
    x_chw = _ensure_chw(x)
    if x_chw.dim() == 4:
        # [B,C,H,W] -> use first batch
        x_chw = x_chw[0]
    if x_chw.dim() != 3:
        raise ValueError(f"Expected 2D/3D/4D tensor for map, got shape={tuple(x.shape)}")

    # [C,H,W]
    if x_chw.shape[0] == 1:
        out = x_chw
    else:
        out = x_chw.mean(dim=0, keepdim=True)

    # Ensure spatial size matches reference
    if out.shape[-2:] != ref_chw.shape[-2:]:
        raise ValueError(
            f"Map spatial size {tuple(out.shape[-2:])} != ref {tuple(ref_chw.shape[-2:])}"
        )
    return out.to(device=ref_chw.device)

try:
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
import OpenEXR
import Imath
from arguments.args import dm_args

_exr_export_warned = False

def save_exr_with_fallback(path: str, array, logger=None) -> str:
    """Try to write EXR using Python OpenEXR bindings; on failure save as NPY.

    This function intentionally does NOT attempt to use OpenCV/imageio backends
    for EXR because those may rely on OpenCV builds without EXR support.
    """
    global _exr_export_warned
    arr = np.asarray(array, dtype=np.float32)

    # Prefer direct OpenEXR Python bindings
    try:
        import OpenEXR as _OpenEXR
        import Imath as _Imath
        FLOAT = _Imath.PixelType(_Imath.PixelType.FLOAT)

        # Ensure path ends with .exr
        out_path = str(path)
        if not out_path.lower().endswith('.exr'):
            out_path = out_path + '.exr'

        # Normalize array to (H, W, C) where C is 1 or 3
        if arr.ndim == 2:
            H, W = arr.shape
            header = _OpenEXR.Header(W, H)
            header['channels'] = {'R': _Imath.Channel(FLOAT)}
            exr = _OpenEXR.OutputFile(out_path, header)
            exr.writePixels({'R': arr.astype(np.float32).tobytes()})
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            H, W, C = arr.shape
            header = _OpenEXR.Header(W, H)
            if C == 1:
                header['channels'] = {'R': _Imath.Channel(FLOAT)}
                exr = _OpenEXR.OutputFile(out_path, header)
                exr.writePixels({'R': arr[:, :, 0].astype(np.float32).tobytes()})
            else:
                header['channels'] = {'R': _Imath.Channel(FLOAT), 'G': _Imath.Channel(FLOAT), 'B': _Imath.Channel(FLOAT)}
                exr = _OpenEXR.OutputFile(out_path, header)
                R = arr[:, :, 0].astype(np.float32).tobytes()
                G = arr[:, :, 1].astype(np.float32).tobytes()
                B = arr[:, :, 2].astype(np.float32).tobytes()
                exr.writePixels({'R': R, 'G': G, 'B': B})
        else:
            raise ValueError(f"Unsupported EXR array shape: {arr.shape}")

        return out_path

    except Exception as exc:
        # Fallback: save numpy file. Keep warning behavior minimal.
        fallback_path = str(path[:-4] + '.npy') if str(path).lower().endswith('.exr') else str(path) + '.npy'
        np.save(fallback_path, arr)
        if not _exr_export_warned:
            msg = f"EXR export via OpenEXR bindings failed, wrote {fallback_path} instead."
            if logger is not None:
                logger.warning(msg + f" ({exc})")
            else:
                print(msg)
            _exr_export_warned = True
        return fallback_path


def save_fixed_material_cache(cache: dict, out_dir: str, logger=None) -> str:
    """Save fixed per-view material cache and quick visualizations.

    Exports:
    - {out_dir}/fixed_material_cache.pt (CPU tensors)
    - {out_dir}/viz/<uid>_albedo.png
    - {out_dir}/viz/<uid>_mlp_intensity.(png|exr)
    - {out_dir}/viz/<uid>_attenuation.(png|exr)
    """
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Move everything to CPU for a portable snapshot.
    cpu_cache: dict[int, dict[str, torch.Tensor]] = {}
    for uid, entry in (cache or {}).items():
        try:
            uid_i = int(uid)
        except Exception:
            continue
        cpu_entry = {}
        for k, v in (entry or {}).items():
            if isinstance(v, torch.Tensor):
                cpu_entry[str(k)] = v.detach().cpu()
        if cpu_entry:
            cpu_cache[uid_i] = cpu_entry

    cache_path = os.path.join(out_dir, "fixed_material_cache.pt")
    torch.save({"cache": cpu_cache}, cache_path)

    # Lightweight visual debug
    for uid_i, entry in cpu_cache.items():
        albedo = entry.get("albedo", None)
        if isinstance(albedo, torch.Tensor) and albedo.dim() == 3 and albedo.shape[0] == 3:
            torchvision.utils.save_image(albedo.clamp(0.0, 1.0), os.path.join(viz_dir, f"{uid_i:04d}_albedo.png"))

        mi = entry.get("mlp_intensity", None)
        if isinstance(mi, torch.Tensor):
            mi_chw = _ensure_chw(mi)
            if isinstance(mi_chw, torch.Tensor) and mi_chw.dim() == 3:
                mi1 = mi_chw[:1]
                mi_np = mi1.squeeze(0).numpy()
                save_exr_with_fallback(os.path.join(viz_dir, f"{uid_i:04d}_mlp_intensity.exr"), mi_np, logger)
                denom = float(mi1.max().item()) if mi1.numel() > 0 else 0.0
                mi_vis = (mi1 / max(denom, 1e-6)).clamp(0.0, 1.0)
                torchvision.utils.save_image(mi_vis, os.path.join(viz_dir, f"{uid_i:04d}_mlp_intensity.png"))

        att = entry.get("attenuation", None)
        if isinstance(att, torch.Tensor):
            att_chw = _ensure_chw(att)
            if isinstance(att_chw, torch.Tensor) and att_chw.dim() == 3:
                att1 = att_chw[:1]
                att_np = att1.squeeze(0).numpy()
                save_exr_with_fallback(os.path.join(viz_dir, f"{uid_i:04d}_attenuation.exr"), att_np, logger)
                denom = float(att1.max().item()) if att1.numel() > 0 else 0.0
                att_vis = (att1 / max(denom, 1e-6)).clamp(0.0, 1.0)
                torchvision.utils.save_image(att_vis, os.path.join(viz_dir, f"{uid_i:04d}_attenuation.png"))

    if logger is not None:
        logger.info(f"Saved fixed material cache: {cache_path}")
    return cache_path


def load_fixed_material_cache(gaussians, cache_path: str, device: str = "cuda", logger=None) -> dict:
    """Load fixed material cache from disk and attach to gaussians."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cpu_cache = (payload or {}).get("cache", {})

    cache_gpu: dict[int, dict[str, torch.Tensor]] = {}
    for uid_i, entry in (cpu_cache or {}).items():
        try:
            uid_i = int(uid_i)
        except Exception:
            continue
        out = {}
        for k, v in (entry or {}).items():
            if isinstance(v, torch.Tensor):
                out[str(k)] = v.to(device=device, non_blocking=True)
        if out:
            cache_gpu[uid_i] = out

    setattr(gaussians, "fixed_after40k_cache", cache_gpu)
    setattr(gaussians, "use_fixed_after40k", True)
    setattr(gaussians, "fixed_after40k_cache_readonly", True)

    if logger is not None:
        logger.info(f"Loaded fixed material cache from: {cache_path} (n_views={len(cache_gpu)})")
    return cache_gpu


@torch.no_grad()
def build_fixed_material_cache_at_iter(
    iteration: int,
    gaussians,
    views,
    pipeline,
    background,
    out_dir: str,
    logger=None,
    max_views: int | None = None,
):
    """Render all target views once to populate gaussians.fixed_after40k_cache.

    This avoids the "first time a view is seen" drift: everything is cached at the same iteration.
    """
    # Use a stable key: camera uid (int).
    cache_root = getattr(gaussians, "fixed_after40k_cache", None)
    if cache_root is None:
        cache_root = {}
        setattr(gaussians, "fixed_after40k_cache", cache_root)

    # GaussianModel isn't nn.Module; use sub-mlp state to restore.
    prev_is_training = True
    try:
        prev_is_training = bool(getattr(gaussians.get_color_mlp, "training", True))
    except Exception:
        prev_is_training = True

    gaussians.eval()
    try:
        views_list = list(views)
        if isinstance(max_views, int) and max_views > 0:
            views_list = views_list[:max_views]

        for i, view in enumerate(tqdm(views_list, desc=f"Build fixed cache @ {iteration}")):
            uid = getattr(view, "uid", None)
            if uid is None:
                continue
            uid_i = int(uid)

            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, require_grad=False)
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                visible_mask=voxel_visible_mask,
                isDetach=True,
            )

            entry = cache_root.setdefault(uid_i, {})
            albedo = render_pkg.get("albedo", None)
            if isinstance(albedo, torch.Tensor):
                entry["albedo"] = _ensure_chw(albedo).detach()

            if bool(getattr(view, "isFlash", False)):
                mi = render_pkg.get("mlp_intensity", None)
                att = render_pkg.get("attenuation", None)
                if isinstance(mi, torch.Tensor):
                    entry["mlp_intensity"] = _ensure_chw(mi).detach()
                if isinstance(att, torch.Tensor):
                    entry["attenuation"] = _ensure_chw(att).detach()

            if (i % 4) == 3:
                torch.cuda.empty_cache()
    finally:
        if prev_is_training:
            gaussians.train()
        else:
            gaussians.eval()

    setattr(gaussians, "use_fixed_after40k", True)
    setattr(gaussians, "fixed_after40k_cache_readonly", True)
    save_fixed_material_cache(cache_root, out_dir=out_dir, logger=logger)
    return cache_root

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    """Backup current runtime code to `dst`.

    Behavior:
      - Overwrites any existing `dst` from previous runs.
      - Uses a temporary directory + rename for a safer (near-atomic) replace.
      - Reads ignore patterns from the repo `.gitignore` next to this file when available.
    """

    additionalIgnorePatterns = ['.git', '.gitignore', '__pycache__']
    ignorePatterns = set()

    log_dir = pathlib.Path(__file__).parent.resolve()
    gitignore_path = log_dir / '.gitignore'
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, 'r') as gitIgnoreFile:
                for line in gitIgnoreFile:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.endswith('/'):
                        line = line[:-1]
                    ignorePatterns.add(line)
        except Exception:
            # If gitignore can't be read, continue with minimal ignore patterns.
            pass

    ignorePatterns = list(ignorePatterns) + list(additionalIgnorePatterns)

    dst_path = pathlib.Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dst_path.with_name(dst_path.name + '_tmp')
    # Clean any leftover temp from previous interrupted runs.
    if tmp_path.exists():
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        else:
            tmp_path.unlink()

    try:
        shutil.copytree(log_dir, tmp_path, ignore=shutil.ignore_patterns(*ignorePatterns))

        # Remove old backup then replace.
        if dst_path.exists():
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()
        tmp_path.rename(dst_path)
        print('Backup Finished!')
    except Exception:
        # Best-effort cleanup of partial tmp.
        try:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
        except Exception:
            pass
        raise




def training(dataset, opt, pipeline, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, gaussian_resume_path=None, save_gaussians_npz=False):
    first_iter = 0
    # --- Simple seed control: allow reproducing runs via `opt.seed` or `opt.random_seed`
    try:
        seed_val = getattr(opt, 'seed', None)
        if seed_val is None:
            seed_val = getattr(opt, 'random_seed', None)
        if seed_val is None:
            seed_val = 0
        seed = int(seed_val)
    except Exception:
        seed = 0

    # Local imports to avoid affecting top-level imports
    import random as _random_module
    import numpy as _np_module

    _random_module.seed(seed)
    _np_module.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For more deterministic behavior (may slow training slightly)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    tb_writer = prepare_output_and_logger(dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_tb_logger = LossTensorboardLogger(
        log_root=os.path.join(
            dataset.model_path,
            "tb_logs",
            timestamp
        ),
        run_name=dataset_name
    )
    # Defer stage-2 material MLP until flash stage to reduce VRAM.
    # Keep eager init when resuming from a .pth checkpoint to avoid optimizer-group mismatch.
    defer_material_mlp = (getattr(opt, 'iterations_useflash', 0) > 0) and (not bool(checkpoint))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.feat_albedo_dim,
        dataset.feat_rm_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        mlp_material_feature_dim=dataset.mlp_material_feature_dim,
        mlp_material_encoding=dataset.mlp_material_encoding,
        mlp_material_hidden_dim=getattr(dataset, "mlp_material_hidden_dim", 256),
        mlp_material_num_hidden_layers=getattr(dataset, "mlp_material_num_hidden_layers", 6),
        mlp_material_hash_n_levels=dataset.mlp_material_hash_n_levels,
        mlp_material_hash_n_features_per_level=dataset.mlp_material_hash_n_features_per_level,
        mlp_material_hash_log2_hashmap_size=dataset.mlp_material_hash_log2_hashmap_size,
        mlp_material_hash_base_resolution=dataset.mlp_material_hash_base_resolution,
        mlp_material_hash_finest_resolution=dataset.mlp_material_hash_finest_resolution,
        mlp_material_hash_per_level_scale=getattr(dataset, "mlp_material_hash_per_level_scale", None),
        mlp_material_hash_bbox_pad=dataset.mlp_material_hash_bbox_pad,
        defer_material_mlp=defer_material_mlp,
    )
    # Optional: reduce per-pixel material query cost during stage-2.
    setattr(gaussians, "mlp_material_pixel_stride", int(getattr(dataset, "mlp_material_pixel_stride", 1) or 1))
    # 这里必须确保输入进来的是先noflash 再flash
    if checkpoint:
        scene = Scene(dataset, gaussians, ply_path=ply_path, load_iteration=-1, shuffle=False)
    else:
        scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)

    resume_iter = None
    if gaussian_resume_path:
        if logger:
            logger.info(f"尝试从 {gaussian_resume_path} 恢复Gaussian npz快照")
        try:
            resume_iter = load_gaussian_npz_snapshot(gaussian_resume_path, gaussians, logger)
        except Exception as exc:
            if logger:
                logger.error(f"加载npz快照失败: {exc}")
            raise

    if resume_iter is not None and resume_iter > 0:
        first_iter = resume_iter

    if checkpoint and gaussian_resume_path and logger:
        logger.warning("同时指定checkpoint与npz快照，checkpoint的内容会覆盖npz中迭代计数")

    # If resuming into/after flash stage, ensure material MLP exists before optimizer is built.
    if getattr(opt, 'iterations_useflash', 0) > 0 and resume_iter is not None and resume_iter >= opt.iterations_useflash:
        try:
            gaussians.ensure_material_mlp(training_args=None)
        except Exception:
            pass

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    gaussExtractor = GaussianExtractor(gaussians, render, prefilter_voxel, pipe=pipeline, bg_color=bg_color)
    # gaussExtractor = RefGaussianExtractor(gaussians, render, pipe=pipeline, bg_color=bg_color)
    REF_GAUSSIAN_START_ITER = 10001    # 2k步开始介入
    MESH_UPDATE_INTERVAL = 5000        # 每2k步更新一次Mesh
    ref_gaussian_mode = False          # 初始为 False 

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, map_location="cuda", weights_only=False)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    view_idx_map = {id(v): i for i, v in enumerate(scene.getTrainCameras())}

    global ind_spec_L0_cache, ind_spec_L1_cache, ind_diffuse_cache, direct_spec_L0_cache, direct_spec_L1_cache, direct_diffuse_cache
    ind_spec_L0_cache.clear()
    ind_spec_L1_cache.clear()
    ind_diffuse_cache.clear()
    direct_spec_L0_cache.clear()
    direct_spec_L1_cache.clear()
    direct_diffuse_cache.clear()

    # Cached irradiance/spec loading and ensure helper functions have been moved to
    # `precompute_useflash.py`. Use those centralized helpers instead of local
    # nested definitions.

    if resume_iter is not None and resume_iter >= opt.iterations_useflash:
        cache_device = "cuda" if bool(getattr(opt, "cache_gpu", False)) else "cpu"
        load_precomp_cache(resume_iter, dataset, scene, gaussians, pipeline, view_idx_map, ind_diffuse_cache, ind_spec_L0_cache, ind_spec_L1_cache, direct_diffuse_cache, direct_spec_L0_cache, direct_spec_L1_cache,  extract_mesh_snapshot, logger, cache_device=cache_device)
    first_iter += 1
    # 记录 loss 历史（用于导出曲线 + TensorBoard）。
    # 统一命名：Raw/*（未乘权重）、Weight/*（权重系数）、W/*（raw*weight，真实加到 total 里的贡献）。
    loss_dict = {}

    def _safe_float(x):
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 0:
                    return None
                return float(x.detach().mean().item())
            return float(x)
        except Exception:
            return None

    def _debug_tensor_stats(name: str, t) -> None:
        if not isinstance(t, torch.Tensor):
            print(f"[NaN debug] {name}: not a tensor ({type(t)})")
            return
        with torch.no_grad():
            t_det = t.detach()
            finite = torch.isfinite(t_det)
            n_total = t_det.numel()
            n_finite = int(finite.sum().item()) if n_total > 0 else 0
            n_nan = int(torch.isnan(t_det).sum().item()) if n_total > 0 else 0
            n_inf = int(torch.isinf(t_det).sum().item()) if n_total > 0 else 0
            try:
                t_min = float(t_det[finite].min().item()) if n_finite > 0 else float("nan")
                t_max = float(t_det[finite].max().item()) if n_finite > 0 else float("nan")
            except Exception:
                t_min, t_max = float("nan"), float("nan")
            print(
                f"[NaN debug] {name}: shape={tuple(t_det.shape)} device={t_det.device} "
                f"finite={n_finite}/{n_total} nan={n_nan} inf={n_inf} min={t_min:.6g} max={t_max:.6g}"
            )

    def _morph_open_kornia(mask_1hw: torch.Tensor, radius: int = 1, iters: int = 1) -> torch.Tensor:
        """Morphological opening (erode->dilate) for binary masks using Kornia.

        mask_1hw: [1,H,W], values in {0,1}
        radius=1 => kernel size 3 (diameter=1+2*radius), ellipse kernel (OpenCV MORPH_ELLIPSE)
        """
        if not isinstance(mask_1hw, torch.Tensor):
            return mask_1hw
        if mask_1hw.dim() != 3 or mask_1hw.shape[0] != 1:
            return mask_1hw
        if iters <= 0:
            return mask_1hw

        # Local import to keep startup lightweight; assumes kornia is available in the env.
        import kornia

        k = int(2 * radius + 1)
        if k != 3:
            # Keep this minimal: only the requested 3x3 ellipse kernel is supported.
            return mask_1hw

        # OpenCV MORPH_ELLIPSE for 3x3 equals a cross.
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, 1.0, 1.0],
             [0.0, 1.0, 0.0]],
            device=mask_1hw.device,
            dtype=torch.float32,
        )

        x = mask_1hw.to(dtype=torch.float32).unsqueeze(0)  # [B=1,C=1,H,W]
        for _ in range(int(iters)):
            x = kornia.morphology.erosion(x, kernel)
            x = kornia.morphology.dilation(x, kernel)

        x = (x > 0.5).to(dtype=mask_1hw.dtype)
        return x.squeeze(0)
    
    # 暂时别删这行，会报错NAN
    light_mlp = LightMLP1D().cuda()
    # Accumulators for weighted-loss 200-iteration mean logging (W/*)
    w200_sum = {}
    w200_cnt = {}

    # Debug: keep one 10k-iteration block of total losses and print top-k.
    loss_top_block = []  # items: (loss_float, iter, cam_idx, isFlash)

    # Freeze/caching after 40k iters: make albedo/mlp_intensity/attenuation fixed values (per camera)
    freeze_after40k_done = False

    def _set_lr_zero_by_group_name(optimizer_obj, target_names):
        if optimizer_obj is None:
            return
        for g in getattr(optimizer_obj, "param_groups", []):
            if g.get("name", None) in target_names:
                g["lr"] = 0.0

    def _freeze_module_params(module_obj):
        if module_obj is None:
            return
        for p in module_obj.parameters():
            p.requires_grad_(False)

    def _norm_chw(x):
        if x is None:
            return None
        return _ensure_chw(x)

    def _expand_1_to_3(x_chw, ref_chw):
        if not isinstance(x_chw, torch.Tensor) or not isinstance(ref_chw, torch.Tensor):
            return x_chw
        if ref_chw.dim() != 3 or x_chw.dim() != 3:
            return x_chw
        if ref_chw.shape[0] == 3 and x_chw.shape[0] == 1:
            return x_chw.expand(3, *x_chw.shape[1:])
        return x_chw

    def _enable_and_build_fixed_cache_once(iteration, background):
        """Enable renderer-side fixed cache and build it once for all flash views (strict)."""
        nonlocal freeze_after40k_done
        if freeze_after40k_done:
            return

        _set_lr_zero_by_group_name(
            gaussians.optimizer_light_intensity,
            {"mlp_light_intensity", "mlp_distance_attenuation"},
        )
        _freeze_module_params(getattr(gaussians, "mlp_light_intensity", None))
        _freeze_module_params(getattr(gaussians, "mlp_distance_attenuation", None))

        setattr(gaussians, "use_fixed_after40k", True)
        # Renderer honors this flag: do not overwrite cache after load/build.
        setattr(gaussians, "fixed_after40k_cache_readonly", True)
        if getattr(gaussians, "fixed_after40k_cache", None) is None:
            setattr(gaussians, "fixed_after40k_cache", {})

        flash_views_all = [
            cam for cam in scene.getTrainCameras()
            if bool(getattr(cam, "isFlash", False))
        ]
        cache_out_dir = os.path.join(
            dataset.model_path,
            "train",
            f"ours_{iteration}",
            "fixed_cache",
        )
        gc.collect()
        torch.cuda.empty_cache()
        build_fixed_material_cache_at_iter(
            iteration=iteration,
            gaussians=gaussians,
            views=flash_views_all,
            pipeline=pipeline,
            background=background,
            out_dir=cache_out_dir,
            logger=logger,
            max_views=None,
        )
        freeze_after40k_done = True
    
    for iteration in range(first_iter, opt.iterations + 1):
        
        gaussians.optimizer.zero_grad(set_to_none = False)       
        gaussians.optimizer_light_intensity.zero_grad(set_to_none=False)       

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        bg_color = [random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipeline.convert_SHs_python, pipeline.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipeline, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_intensity_learn_rate(iteration)
   
        # Pick a random Camera
        isDetach = False
        
        if dataset.flash == False:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            # 如果启用闪光灯，则 iterations_useflash 之前使用 nonflash 约束几何
            # iterations_useflash 之后使用 flash 约束材质
            viewpoint_stack = scene.getTrainCameras().copy()
            if opt.iterations_useflash > 0 and iteration <= opt.iterations_useflash:
                nonflash_num = scene.getNonFlashNum()
                viewpoint_cam = viewpoint_stack.pop(randint(0, nonflash_num-1))
                # exclude_indices = [56]  # 这些索引不想选
                # available_indices = [i for i in range(len(viewpoint_stack)) if i not in exclude_indices]

                # # 随机选择
                # idx = available_indices[randint(0, nonflash_num-1)]
                # viewpoint_cam = viewpoint_stack.pop(idx)
            else:
                nonflash_num = scene.getNonFlashNum()
                
                flash_num = scene.getFlashNum()
                # exclude_indices = [318, 319, 320]  # 这些索引不想选
                # available_indices = [i for i in range(len(viewpoint_stack)) if i not in exclude_indices]

                # 随机选择
                # idx = available_indices[randint(nonflash_num, len(available_indices)-1)]
                # viewpoint_cam = viewpoint_stack.pop(idx)
                isDetach = True
                cam_num = randint(nonflash_num, len(viewpoint_stack)-1)
                viewpoint_cam = viewpoint_stack.pop(cam_num)
                viewpoint_cam_nonflash = viewpoint_stack.pop(cam_num - nonflash_num)

        # Render
        if (iteration - 1) == debug_from:
            pipeline.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipeline, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)

        
        if iteration >= opt.indirect_from_iter + 1:
            opt.indirect = 1
        
        
        # 2k-20k 步期间，定期提取 Mesh 并开启 RayTracing
        if iteration >= REF_GAUSSIAN_START_ITER and iteration <= 20000:
            with torch.no_grad():
            # 每隔 MESH_UPDATE_INTERVAL (2000) 步，或者刚开始进入阶段时
                if iteration == REF_GAUSSIAN_START_ITER or (iteration % MESH_UPDATE_INTERVAL == 0):
                    try:
                        torch.cuda.empty_cache()
                        """
                        # voxel_size 可以从 dataset 获取，如果没有就给个默认值 0.004
                        #v_size = dataset.voxel_size if hasattr(dataset, 'voxel_size') else 0.004
                        #mesh = gaussExtractor.extract_mesh_bounded(voxel_size=v_size, sdf_trunc=5*v_size, depth_trunc=3.0)
                        gaussExtractor.reconstruction(scene.getTrainCameras())
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=512)
                        """
                        scene_radius = getattr(gaussExtractor, "radius", None)
                        if scene_radius is None:
                            scene_radius = scene.cameras_extent
                        if scene_radius is not None:
                            depth_trunc = scene_radius * 2.0
                            depth_range = getattr(gaussExtractor, "depth_range", None)
                            if depth_range is not None:
                                depth_max = depth_range[1]
                                depth_trunc = max(depth_trunc, depth_max * 1.05)
                        else:
                            depth_trunc = 3.0
                        mesh_res = 512 
                        voxel_size_TSDF = -1
                        voxel_size = depth_trunc / mesh_res
                        sdf_trunc = 3.0 * voxel_size
                        
                        # 重建 TSDF
                        gaussExtractor.reconstruction(scene.getTrainCameras())
                        # mesh = gaussExtractor.extract_mesh_unbounded(resolution=mesh_res)
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
                        # 去噪
                        mesh = post_process_mesh(mesh, cluster_to_keep=1)

                        # 更新 mesh
                        gaussians.update_mesh(mesh)
                        
                        o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, f"test_{iteration:06d}.ply"), mesh)
                        


                        ref_gaussian_mode = True
                        # print(f"[ITER {iteration}] Ref-Gaussian Mode UPDATED.")
                    except Exception as e:
                        print(f"[Warning] Mesh extraction failed at {iteration}: {e}")
                        ref_gaussian_mode = False
            
        if iteration > 20000:
            ref_gaussian_mode = False
        """
        # --- 监控点 1：Render 前 ---
        if hasattr(gaussians, 'env_map') and gaussians.env_map is not None:
            v_before = list(gaussians.env_map.parameters())[0]._version
            print(f"[DEBUG-INPLACE] 1. Render前 Version: {v_before}")
            input("监控1")
        """
        gaussians.train()

        if ref_gaussian_mode:
            # 使用 Ref-Gaussian 的物理渲染 (带光追)
            # render_surfel 需要 opt 参数来决定是否开启 indirect 等
            if hasattr(gaussians, 'env_map') and gaussians.env_map is not None:
                envmap = gaussians.get_envmap
                envmap.build_mips()
                # gaussians.env_map.build_mips()
            render_pkg = render_surfel(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, opt=opt)
            # render_pkg_2 = render_debug(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, isDetach=isDetach)
            # render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, isDetach=isDetach)
            # input("测试")
            if "selection_mask" not in render_pkg:
                render_pkg["selection_mask"] = (render_pkg["radii"] > 0)
            if "scaling" not in render_pkg:
                render_pkg["scaling"] = gaussians.get_scaling()
            if "neural_opacity" not in render_pkg:
                render_pkg["neural_opacity"] = gaussians.get_opacity()       
        else:
            # 原始
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, isDetach=isDetach)
        """
        # --- 监控点 2：Render 后，Loss 前 ---
        if hasattr(gaussians, 'env_map') and gaussians.env_map is not None:
            v_after_render = list(gaussians.env_map.parameters())[0]._version
            print(f"[DEBUG-INPLACE] 2. Render后 Version: {v_after_render}")
        """
        #render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, isDetach=isDetach)

        # image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        offset_selection_mask = render_pkg.get("selection_mask", None)
        scaling = render_pkg.get("scaling", None)
        opacity = render_pkg.get("neural_opacity", None)

        # Move GT tensors to the same device as render output.
        # NOTE: this can be a major bottleneck if GT lives on CPU; profiler accounts for it.
        dev = image.device if isinstance(image, torch.Tensor) else torch.device("cuda")
        gt_image = viewpoint_cam.original_image
        if isinstance(gt_image, torch.Tensor):
            gt_image = gt_image.to(dev, non_blocking=True)

        gt_albedo = getattr(viewpoint_cam, "albedo", None)
        gt_roughness = getattr(viewpoint_cam, "roughness", None)
        gt_metallic = getattr(viewpoint_cam, "metallic", None)
        gt_normal = getattr(viewpoint_cam, "normal", None)
        gt_depth = getattr(viewpoint_cam, "depth", None)
        gt_segment = getattr(viewpoint_cam, "segment", None)


        if isinstance(gt_albedo, torch.Tensor):
            gt_albedo = gt_albedo.to(dev, non_blocking=True)
        if isinstance(gt_roughness, torch.Tensor):
            gt_roughness = gt_roughness.to(dev, non_blocking=True)
        if isinstance(gt_metallic, torch.Tensor):
            gt_metallic = gt_metallic.to(dev, non_blocking=True)
        if isinstance(gt_normal, torch.Tensor):
            gt_normal = gt_normal.to(dev, non_blocking=True)
        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.to(dev, non_blocking=True)
        if isinstance(gt_segment, torch.Tensor):
            gt_segment = gt_segment.to(dev, non_blocking=True)

        # gt_image_linear = srgb2linear(gt_image)
        # image_linear = srgb2linear(image)
        # torchvision.utils.save_image(linear2srgb(gt_image_linear - image_linear), f'output/flash_expect_srgb/{iteration:04d}_flash.png')
        # input("debug pause")
        diffuse_flash_color = None
        spec_flash_color = None
        rough_map = None
        albedo = None

        

        # 如果是flash图片，给渲染结果加上flashlight_defer，并叠加间接光照（使用缓存的irradiance，一次加载）
        if iteration > opt.iterations_useflash:
            # flashlight_defer = render_pkg.get("flashlight_defer", None)

            cam_idx = view_idx_map.get(id(viewpoint_cam), None)

            image_linear = viewpoint_cam_nonflash.original_image # image
            
            ind_diffuse = _norm_chw(ind_diffuse_cache.get(cam_idx, None))
            direct_diffuse = _norm_chw(direct_diffuse_cache.get(cam_idx, None))
            albedo_lin = _norm_chw(render_pkg.get("albedo", None))
            metallic_lin = _norm_chw(render_pkg.get("metallic", None))
            roughness_lin = _norm_chw(render_pkg.get("roughness", None))
            attenuation = _norm_chw(render_pkg.get("attenuation", None))
            mlp_intensity = _norm_chw(render_pkg.get("mlp_intensity", None))

            if metallic_lin is not None:
                metallic_lin = torch.zeros_like(metallic_lin)  # TODO 临时固定一个金属度，方便测试

            # 40k 后：冻结光照 MLP，并一次性构建 fixed cache；之后渲染器只读复用，避免训练侧再写/覆盖。
            if iteration >= 28000 and (not freeze_after40k_done):
                _enable_and_build_fixed_cache_once(iteration=iteration, background=background)

            # --- Diffuse (early return style)
            if (
                ind_diffuse is not None
                and direct_diffuse is not None
                and albedo_lin is not None
                and metallic_lin is not None
            ):
                ind_diffuse = ind_diffuse.to(image_linear.device).float()
                direct_diffuse = direct_diffuse.to(image_linear.device).float()
                albedo = albedo_lin.to(image_linear.device).float()
                metallic = metallic_lin.to(image_linear.device).float()

                # Require both mlp_intensity and attenuation to be present as Tensors.
                mlp_i = mlp_intensity
                att = attenuation
                if not (isinstance(mlp_i, torch.Tensor) and isinstance(att, torch.Tensor)):
                    msg = f"mlp_intensity or attenuation missing/not Tensor at iter={iteration} cam_idx={cam_idx}"
                    if logger is not None:
                        logger.error(msg)
                    raise RuntimeError(msg)

                # Convert both to device/float once (they are guaranteed Tensors here).
                mlp_i = mlp_i.to(image_linear.device).float()
                att = att.to(image_linear.device).float()

                if isinstance(mlp_i, torch.Tensor):
                    mlp_i = _expand_1_to_3(_ensure_chw(mlp_i), ind_diffuse)
                if isinstance(att, torch.Tensor):
                    att = _expand_1_to_3(_ensure_chw(att), ind_diffuse)

                albedo = _expand_1_to_3(albedo, ind_diffuse)
                metallic = _expand_1_to_3(metallic, ind_diffuse)
                # mlp_i = torch.ones_like(mlp_i) * 1.0
                direct_diffuse_flash_color = direct_diffuse * albedo * (1.0 - metallic) * mlp_i / att
                ind_diffuse_flash_color = ind_diffuse * albedo * (1.0 - metallic)
                diffuse_flash_color = direct_diffuse_flash_color + ind_diffuse_flash_color  # TODO ind
                image_linear = image_linear + diffuse_flash_color

            # --- Specular precomputed part
            if (
                cam_idx is not None
                and roughness_lin is not None
                and albedo_lin is not None
                and metallic_lin is not None
            ):
                rough_map = roughness_lin.to(image_linear.device).float()
                rough_map = _ensure_chw(rough_map)
                if rough_map.shape[0] == 3:
                    rough_map = rough_map[:1]

                ind_L0_s, ind_L1_s, direct_L0_s, direct_L1_s = _interp_spec(int(cam_idx), rough_map)
                if (
                    ind_L0_s is not None
                    and ind_L1_s is not None
                    # and direct_L0_s is not None
                    # and direct_L1_s is not None
                ):
                    ind_L0 = ind_L0_s
                    ind_L1 = ind_L1_s
                    direct_L0 = direct_L0_s
                    direct_L1 = direct_L1_s

                    metallic_spec = _ensure_chw(render_pkg.get("metallic", None))
                    if metallic_spec is None:
                        metallic_spec = torch.zeros_like(albedo[:1])
                    # expand channels if needed
                    if metallic_spec.shape[0] == 1 and albedo.shape[0] == 3:
                        metallic_spec = metallic_spec.expand(3, *metallic_spec.shape[1:])

                    mlp_intensity = _ensure_chw(mlp_intensity) if mlp_intensity is not None else None
                    attenuation = _ensure_chw(attenuation) if attenuation is not None else None
                    # Move L0/L1 to same device as albedo
                    device = albedo.device if isinstance(albedo, torch.Tensor) else None
                    if device is not None:
                        ind_L0 = ind_L0.to(device)
                        ind_L1 = ind_L1.to(device)
                        direct_L0 = direct_L0.to(device)
                        direct_L1 = direct_L1.to(device)
                        metallic_spec = metallic_spec.to(device)
                        if isinstance(mlp_intensity, torch.Tensor):
                            mlp_intensity = mlp_intensity.to(device)
                        if isinstance(attenuation, torch.Tensor):
                            attenuation = attenuation.to(device)

                    
                    # mlp_intensity = torch.ones_like(mlp_intensity) * 1
                    attenuation = attenuation if attenuation is not None else 1.0
                    ks = albedo * metallic_spec + 0.04 * (1.0 - metallic_spec)
                    direct_spec_flash_color = (ks * direct_L0 + direct_L1) * mlp_intensity / (attenuation + 1e-6)
                    ind_spec_flash_color = (ks * ind_L0 + ind_L1)
                    spec_flash_color = direct_spec_flash_color + ind_spec_flash_color # TODO ind
                    image_linear = image_linear + spec_flash_color # TODO Spec
            
            # Run these diagnostics only every 1000 iterations (skip iteration 0)
            if iteration != 0 and (iteration % 500) == 0:
                if isinstance(diffuse_flash_color, torch.Tensor):
                    torchvision.utils.save_image(diffuse_flash_color.clamp(0.0, 1.0), f'output/flashlight_diff_{iteration}.png')
                if isinstance(spec_flash_color, torch.Tensor):
                    torchvision.utils.save_image(spec_flash_color.clamp(0.0, 1.0), f'output/flashlight_spec_{iteration}.png')
                if isinstance(image_linear, torch.Tensor):
                    # torchvision.utils.save_image(image_linear.clamp(0.0, 1.0), f'output/flashlight_total_linear_{iteration}.png')
                    torchvision.utils.save_image(linear2srgb(image_linear).clamp(0.0, 1.0), f'output/flashlight_total_srgb_{iteration}.png')
                if isinstance(albedo, torch.Tensor):
                    torchvision.utils.save_image(albedo.clamp(0.0, 1.0), f'output/albedo_{iteration}.png')
                # torchvision.utils.save_image(metallic.clamp(0.0, 1.0), f'output/flashlight_metallic_{iteration}.png')
                # torchvision.utils.save_image(linear2srgb(direct_spec_flash_color), f'output/direct_spec_flash_color_{iteration}.png')
                # torchvision.utils.save_image(linear2srgb(ind_spec_flash_color), f'output/ind_spec_flash_color_{iteration}.png')
                if isinstance(rough_map, torch.Tensor):
                    torchvision.utils.save_image(rough_map.clamp(0.0, 1.0), f'output/roughness_{iteration}.png')
                # mlp_i_vis = mlp_i / mlp_i.max()
                # torchvision.utils.save_image(mlp_i_vis.clamp(0.0, 1.0), f'output/mlp_i_{iteration}.png')
                torchvision.utils.save_image(viewpoint_cam_nonflash.original_image.clamp(0.0, 1.0), f'output/nonflash_gt_{iteration}.png')
                
            image = image_linear

        # Per-pixel loss weighting by precomputed direct L0 at roughness=0.2
        if not torch.is_tensor(image):
            image = torch.as_tensor(image, device=gt_image.device)
        if not torch.is_tensor(gt_image):
            gt_image = torch.as_tensor(gt_image, device=image.device)

        light_distance = render_pkg.get("light_distance", None)
        n_dot_l = render_pkg.get("n_dot_l", None)
        # torchvision.utils.save_image(n_dot_l, f'output/{iteration:04d}_n_dot_l.png')
        # input("debug pause check n_dot_l")

        # Normalize images to [3,H,W] for loss
        image_chw = _ensure_chw(image)
        gt_image_chw = _ensure_chw(gt_image)

        # Build per-pixel mask in [1,H,W], values in {0,1}
        mask = torch.ones((1, image_chw.shape[-2], image_chw.shape[-1]), device=image_chw.device, dtype=torch.float32)
        light_distance = _to_1hw_scalar_map(light_distance, ref_chw=image_chw)
        if light_distance is not None:
            mask = mask * (light_distance >= 0.0).to(mask.dtype) # * (light_distance <= 3).to(mask.dtype)

        # mask_1hw
        # if iteration > 20000 and iteration % 500 == 0:
        #     torchvision.utils.save_image(mask, f'output/mask_distance_{iteration:04d}.png')
        #     torchvision.utils.save_image(light_distance/light_distance.max(), f'output/light_distance_{iteration:04d}.png')


        # n_dot_l = _to_1hw_scalar_map(n_dot_l, ref_chw=image_chw)
        # if n_dot_l is not None:
        #     mask = mask * (n_dot_l >= 0.2).to(mask.dtype)

        # if iteration > 20000 and iteration % 500 == 0:
        #     torchvision.utils.save_image(mask, f'output/mask_ndotl_{iteration:04d}.png')
        #     torchvision.utils.save_image(n_dot_l, f'output/n_dot_l_{iteration:04d}.png')


        # if iteration > 30000: # opt.iterations_useflash:
        #     weight_map = (ks * direct_L0 + direct_L1) * 6.0
        #     weight_map = torch.clamp_min(weight_map, 0.5)
        #     weight_map = torch.clamp_max(weight_map, 20.0)
        #     # weight_map = torch.where(weight_map > 0.5, torch.ones_like(weight_map) * 10.0, torch.full_like(weight_map, 1))
        #     # torchvision.utils.save_image(weight_map, f'output/weight_map_{iteration}.png')
        #     # input("weight map saved, press enter to continue")

        #     image_L1 = torch.abs(image_chw - gt_image_chw)  # [3,H,W]

        #     # Combine weight_map with 0/1 mask, then broadcast to RGB
        #     weight_1hw = _ensure_chw(weight_map)
        #     if weight_1hw.shape[0] != 1:
        #         weight_1hw = weight_1hw.mean(dim=0, keepdim=True)
        #     weight_1hw = weight_1hw.to(device=image_chw.device, dtype=torch.float32) # * mask

        #     # Keep old normalization behavior: denom is per-pixel weight sum (not multiplied by C)
        #     weight3 = weight_1hw.expand_as(image_L1)

        #     # if iteration > 25000:
        #     image_L1 = image_L1 * weight3.to(dtype=image_L1.dtype)

        #     Ll1 = image_L1.sum() / (weight_1hw.sum() + 1e-6)
        #     # ssim_loss = weighted_ssim_loss(image, gt_image, weight_map)

        #     ssim_map = weighted_ssim_map(image_chw, gt_image_chw)   # expected [1,H,W] or [C,H,W]
        #     ssim_map = _ensure_chw(ssim_map)
        #     if ssim_map.shape[0] != 1:
        #         ssim_map = ssim_map.mean(dim=0, keepdim=True)
        #     ssim_weighted = ssim_map.to(dtype=torch.float32) * weight_1hw

        #     ssim_loss = ssim_weighted.sum() / (weight_1hw.sum() + 1e-6)
        # else:
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)

        # gt_object_mask = getattr(viewpoint_cam, "object_mask", None)
        # if isinstance(gt_object_mask, torch.Tensor):
        #     object_mask = _ensure_chw(gt_object_mask)
        #     if object_mask.shape[0] != 1:
        #         object_mask = object_mask.mean(dim=0, keepdim=True)
        #     object_mask = (object_mask > 0.5).to(dtype=mask.dtype, device=mask.device)
        #     mask = mask * object_mask
        # torchsvision.utils.save_image(object_mask, f'output/object_mask_{iteration:04d}.png')
        # if iteration < 30000:
        #     err_1hw = torch.abs(image_chw - gt_image_chw).mean(dim=0, keepdim=True)  # [1,H,W]
        #     denom = mask.sum().clamp_min(1e-6)
        #     Ll1 = (err_1hw * mask).sum() / denom

        #     ssim_map = weighted_ssim_map(image_chw, gt_image_chw)
        #     ssim_map = _ensure_chw(ssim_map)
        #     if ssim_map.shape[0] != 1:
        #         ssim_map = ssim_map.mean(dim=0, keepdim=True)
        #     ssim_loss = (ssim_map.to(dtype=torch.float32) * mask).sum() / denom
        # else:
        #     def gamma(x):
        #         """ tone mapping function """
        #         mask = x <= 0.0031308
        #         ret = torch.empty_like(x)
        #         ret[mask] = 12.92*x[mask]
        #         mask = ~mask
        #         ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        #         return ret

        #     ssim_loss = 0
        #     err_1hw = ((gamma(gt_image_chw) - gamma(image_chw)) ** 2).mean(dim=0, keepdim=True)  # [1,H,W]
        #     denom = mask.sum().clamp_min(1e-6)
        #     Ll2 = (err_1hw * mask).sum() / denom
        #     Ll1 = Ll2 * 10

        L1Weight = 1 # if iteration < 30000 else 0.05
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 * L1Weight + opt.lambda_dssim * ssim_loss * L1Weight + 0.01 * scaling_reg # 
        """
        rendered_normal = render_pkg["render_normal"]
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
        """
        # if ref_gaussian_mode:
        #     # 直接调用 calculate_loss
        #     # loss, loss_dict_ref = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        #     # if 'loss_l1' in loss_dict_ref: Ll1 = torch.tensor(loss_dict_ref['loss_l1'], device=image.device)
        #     # if 'ssim' in loss_dict_ref: ssim_loss = torch.tensor(1.0 - loss_dict_ref['ssim'], device=image.device)
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 * L1Weight + opt.lambda_dssim * ssim_loss * L1Weight + 0.01 * scaling_reg

        # else:
        #     # 使用原来的公式
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 * L1Weight + opt.lambda_dssim * ssim_loss * L1Weight + 0.01 * scaling_reg

        albedo = render_pkg["albedo"] if "albedo" in render_pkg else None
        roughness = render_pkg["roughness"] if "roughness" in render_pkg else None
        metallic = render_pkg["metallic"] if "metallic" in render_pkg else None
        surf_depth = render_pkg.get("surf_depth", None)

        roughness_to1_raw = torch.zeros((), device=loss.device, dtype=loss.dtype)
        roughness_to1_w = 5e-3
        roughness_to1 = torch.zeros((), device=loss.device, dtype=loss.dtype)
        """
        if isinstance(roughness, torch.Tensor):
            roughness_to1_raw =  (metallic.mean()) # (roughness-1).abs().mean() + 
            roughness_to1 = roughness_to1_raw * float(roughness_to1_w)
            loss += roughness_to1
        """
        # if isinstance(roughness, torch.Tensor): #and ref_gaussian_mode is not True:
        #     roughness_to1_raw =  (metallic.mean())
        #     roughness_to1 = roughness_to1_raw * float(roughness_to1_w)
        #     loss += roughness_to1

        # if iteration > 60000:
        # #     ones_roughness = torch.ones_like(gt_roughness)
        # #     roughness_error = (roughness - ones_roughness).abs()
        # #     loss += 1 * roughness_error.mean()
        
        
        # # zeros_metallic = torch.zeros_like(metallic)
        # # metallic_error = (metallic - zeros_metallic).abs()
        # # loss += 0.01 * metallic_error.mean()
        # metallic_error = (metallic - gt_metallic).abs()
        # loss += 10.0 * metallic_error.mean()
        # if iteration > 20000:
        #     albedo_error = (albedo - gt_albedo).abs()
        #     loss += 10.0 * albedo_error.mean()
        #     roughness_error = (roughness - gt_roughness).abs()
        #     loss += 10.0 * roughness_error.mean()
        
        

        # Optional: one-way roughness penalty (roughness > 0.3) in strong-highlight regions.
        roughness_upper_raw = torch.zeros((), device=loss.device, dtype=loss.dtype)
        roughness_upper_w = 1000.0 # if iteration < 25000 else 1000.0
        roughness_upper = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # 基于分割的传播正则：同一分割内 roughness/metallic 应该一致
        # 预先准备 raw/weight/weighted 三个变量，确保 metrics 记录块安全（缺项不会崩溃）
        loss_seg_raw = torch.zeros((), device=loss.device, dtype=loss.dtype)
        loss_seg_w = float(getattr(opt, "lambda_seg", 0.0) or 0.0)
        loss_seg_w = 0.1 # if iteration < 25000 else 0.25
        loss_seg = torch.zeros((), device=loss.device, dtype=loss.dtype)

        light_distance = render_pkg["light_distance"] if "light_distance" in render_pkg else None
        attenuation = render_pkg["attenuation"] if "attenuation" in render_pkg else None
        if iteration > 20000 and light_distance is not None:
            # Constrain attenuation to be close to distance^2 (用户要求：不要用 0 替代)
            d2 = _ensure_chw(light_distance ** 2)
            atten = _ensure_chw(attenuation) if attenuation is not None else None
            if isinstance(d2, torch.Tensor) and d2.dim() == 3 and d2.shape[0] != 1:
                d2 = d2.mean(dim=0, keepdim=True)
            if isinstance(atten, torch.Tensor) and atten.dim() == 3 and atten.shape[0] != 1:
                atten = atten.mean(dim=0, keepdim=True)
            if isinstance(atten, torch.Tensor):
                atten = atten.to(d2.device)
                
            # Use absolute difference between d^2 and attenuation, reduce to scalar
            """
            if isinstance(atten, torch.Tensor):
                loss = loss + (d2 - atten).abs().mean() * 0.001
            """
            if isinstance(atten, torch.Tensor) :#and ref_gaussian_mode is not True:
                loss = loss + (d2 - atten).abs().mean() * 0.001

        gt_albedo = viewpoint_cam.albedo.cuda()
        gt_roughness = viewpoint_cam.roughness.cuda() # 确保这两个是一维的
        gt_metallic = viewpoint_cam.metallic.cuda()
        gt_normal = viewpoint_cam.normal.cuda()
        gt_depth = getattr(viewpoint_cam, "depth", None)
        gt_segment = getattr(viewpoint_cam, "segment", None)
        # torchvision.utils.save_image(gt_segment, f'output/gt_segment_{iteration:04d}.png')
        # input("debug pause")
        if isinstance(gt_segment, torch.Tensor):
            gt_segment = gt_segment.to(image.device, non_blocking=True)
        cam_idx = view_idx_map.get(id(viewpoint_cam), None)
        # Depth debug + robust masking (avoid -inf/inf/nan exploding backward)
        normal_error = None
        normal_gt_error = None
        normal_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        normal_gt_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        depth_raw = torch.zeros((), device=loss.device, dtype=loss.dtype)
        depth_w = 0.05
        depth_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)

        # regularization
        render_normal  = render_pkg['render_normal']
        surf_normal = render_pkg['surf_normal']
        normal_cam = render_pkg['normal_cam']

        lambda_normal = opt.lambda_normal # if iteration > opt.iterations_normal else 0.0
        lambda_gt_normal = opt.lambda_gt_normal if iteration > opt.iterations_normal else 0.0
        lambda_dist = opt.lambda_dist if iteration > opt.iterations_dist else 0.0

        normal_error = (1 - (render_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        loss += normal_loss

        # if iteration < opt.iterations_useflash  and gt_depth is not None and iteration > opt.iterations_normal and isinstance(gt_depth, torch.Tensor) and isinstance(surf_depth, torch.Tensor):
        if iteration < opt.iterations_useflash  and iteration > opt.iterations_normal:    
            # input("debug pause before depth loss")
            # finite = torch.isfinite(gt_depth) & torch.isfinite(surf_depth)
            # valid = finite & (gt_depth > 0.0) & (surf_depth > 0.0)
            # """
            # if valid.any() :
            #     depth_error = (surf_depth - gt_depth).abs()
            #     depth_raw = depth_error[valid].mean()
            #     depth_loss = depth_raw * float(depth_w)
            #     loss += depth_loss
            # """
            # if valid.any() and ref_gaussian_mode is not True:
            #     depth_error = (surf_depth - gt_depth).abs()
            #     depth_raw = depth_error[valid].mean()
            #     depth_loss = depth_raw * float(depth_w)
            #     loss += depth_loss

            # gt_normal, normal_cam: [3, H, W]
            gt_normal = gt_normal * 2.0 - 1.0 # [-1, 1]
            gt_normal = torch.nn.functional.normalize(gt_normal, dim=0) 
            normal_cam = torch.nn.functional.normalize(normal_cam, dim=0)
            render_normal = torch.nn.functional.normalize(render_normal, dim=0) 

            # 求Normal World GT 的loss
            # normal_gt_error = (1 - (render_normal * gt_normal).sum(dim=0))[None]
            """
            # 求Normal Cam GT 的loss
            normal_gt_error = (1 - (normal_cam * gt_normal).sum(dim=0))[None]

            normal_gt_loss = lambda_gt_normal * (normal_gt_error).mean()
            loss += normal_gt_loss
            """
            
            
        render_dist = render_pkg["render_dist"]

        dist_loss = lambda_dist * (render_dist).mean()
        #loss += dist_loss 

        # if ref_gaussian_mode is not True:
        loss += dist_loss 

        # Track worst total losses in the last 10k-iteration block (debug only; no gradients).
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(f"loss is not a Tensor: {type(loss)} (iter={iteration}, cam_idx={cam_idx})")
        if loss.numel() != 1:
            raise RuntimeError(f"loss is not scalar: shape={tuple(loss.shape)} (iter={iteration}, cam_idx={cam_idx})")
        loss_val = float(loss.detach().item())
        loss_top_block.append(
            (
                loss_val,
                int(iteration),
                cam_idx,
                bool(getattr(viewpoint_cam, "isFlash", False)),
            )
        )

        block_size = 1000
        topk_n = 5
        if (iteration % block_size) == 0 and len(loss_top_block) > 0:
            if logger is None:
                raise RuntimeError("logger is None; loss-topk printing requires logger")
            topk = sorted(loss_top_block, key=lambda t: t[0], reverse=True)[:topk_n]
            header = f"[loss-top{topk_n}@{iteration}] block={block_size} items={len(loss_top_block)}"
            lines = [header]
            for rank, (lv, it_, ci_, is_flash_) in enumerate(topk, start=1):
                lines.append(f"  #{rank}: loss={lv:.6g} iter={it_} cam_idx={ci_} isFlash={int(is_flash_)}")
            logger.info("\n".join(lines))
            loss_top_block.clear()

        # 记录loss到 loss_history
        # 每n步执行一次保存loss，n默认100
        # Accumulate per-iteration weighted contributions (W/*) for 200-step mean logging
        lam_200 = float(getattr(opt, "lambda_dssim", 0.0) or 0.0)
        w_l1_200 = (1.0 - lam_200) * L1Weight
        w_ssim_200 = lam_200 * L1Weight
        w_scaling_200 = 0.01

        w_metrics_step = {
            "Total": loss,
            "W/ImageL1": Ll1 * float(w_l1_200),
            "W/SSIM": ssim_loss * float(w_ssim_200),
            "W/ScalingReg": scaling_reg * float(w_scaling_200),
            "W/Normal": normal_loss,
            "W/NormalGt": normal_gt_loss,
            "W/Dist": dist_loss,
            "W/Depth": depth_loss,
            "W/Roughness>0.3": roughness_upper,
            "W/RoughnessTo1": roughness_to1,
            "W/Seg": loss_seg,
        }
        for k, v in w_metrics_step.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            w200_sum[k] = w200_sum.get(k, 0.0) + fv
            w200_cnt[k] = w200_cnt.get(k, 0) + 1

        if (iteration % 200) == 0 and len(w200_cnt) > 0:
            tb_metrics_200 = {}
            for k, sum_v in w200_sum.items():
                cnt = int(w200_cnt.get(k, 0) or 0)
                if cnt <= 0:
                    continue
                tb_metrics_200[f"{k}_200"] = float(sum_v) / float(cnt)

            for k, v in tb_metrics_200.items():
                loss_dict.setdefault(k, []).append(v)
            loss_tb_logger.log_losses(iteration, tb_metrics_200)

            w200_sum.clear()
            w200_cnt.clear()

        loss_save_interval = getattr(opt, 'loss_save_interval', 100)

        if iteration % 1000 == 0:
            print(f"gaussians count: {gaussians._anchor.shape[0]}, offsets count: {gaussians._scaling_offsets.shape[0]}")

        # 每n步保存一下loss
        if iteration % loss_save_interval == 0:
            lam = float(getattr(opt, "lambda_dssim", 0.0) or 0.0)
            w_l1 = (1.0 - lam) * L1Weight
            w_ssim = lam * L1Weight
            w_scaling = 0.01

            normal_error_for_log = locals().get("normal_error", None)
            normal_gt_error_for_log = locals().get("normal_gt_error", None)
            render_dist_for_log = locals().get("render_dist", None)
            metrics = {
                "Total": loss,

                "Raw/ImageL1": Ll1,
                "Weight/ImageL1": w_l1,
                "W/ImageL1": Ll1 * float(w_l1),

                "Raw/SSIM": ssim_loss,
                "Weight/SSIM": w_ssim,
                "W/SSIM": ssim_loss * float(w_ssim),

                "Raw/ScalingReg": scaling_reg,
                "Weight/ScalingReg": w_scaling,
                "W/ScalingReg": scaling_reg * float(w_scaling),

                "Raw/Normal": normal_error_for_log.mean() if isinstance(normal_error_for_log, torch.Tensor) else normal_loss,
                "Weight/Normal": float(lambda_normal),
                "W/Normal": normal_loss,

                "Raw/NormalGt": normal_gt_error_for_log.mean() if isinstance(normal_gt_error_for_log, torch.Tensor) else normal_gt_loss,
                "Weight/NormalGt": float(lambda_gt_normal),
                "W/NormalGt": normal_gt_loss,

                "Raw/Dist": render_dist_for_log.mean() if isinstance(render_dist_for_log, torch.Tensor) else dist_loss,
                "Weight/Dist": float(lambda_dist),
                "W/Dist": dist_loss,

                "Raw/Depth": depth_raw,
                "Weight/Depth": float(depth_w) if (iteration > opt.iterations_normal and isinstance(gt_depth, torch.Tensor) and isinstance(surf_depth, torch.Tensor)) else 0.0,
                "W/Depth": depth_loss,

                "Raw/Roughness>0.3": roughness_upper_raw,
                "Weight/Roughness>0.3": float(roughness_upper_w) if (diffuse_flash_color is not None and iteration > 25000) else 0.0,
                "W/Roughness>0.3": roughness_upper,

                "Raw/RoughnessTo1": roughness_to1_raw,
                "Weight/RoughnessTo1": float(roughness_to1_w) if isinstance(roughness, torch.Tensor) else 0.0,
                "W/RoughnessTo1": roughness_to1,
                
                "Raw/Seg": loss_seg_raw,
                "Weight/Seg": float(loss_seg_w),
                "W/Seg": loss_seg,
            }

            tb_metrics = {}
            for k, v in metrics.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                loss_dict.setdefault(k, []).append(fv)
                tb_metrics[k] = fv

            loss_tb_logger.log_losses(iteration, tb_metrics)
            
        if not torch.isfinite(loss).all():
            
            _debug_tensor_stats("image_chw", locals().get("image_chw", None))
            _debug_tensor_stats("gt_image_chw", locals().get("gt_image_chw", None))
            _debug_tensor_stats("mask", locals().get("mask", None))
            _debug_tensor_stats("Ll1", locals().get("Ll1", None))
            _debug_tensor_stats("ssim_loss", locals().get("ssim_loss", None))
            _debug_tensor_stats("scaling_reg", locals().get("scaling_reg", None))
            _debug_tensor_stats("loss", loss)
            _debug_tensor_stats("n_dot_l", locals().get("n_dot_l", None))
            _debug_tensor_stats("light_distance", locals().get("light_distance", None))
            _debug_tensor_stats("albedo", locals().get("albedo", None))
            _debug_tensor_stats("roughness", locals().get("roughness", None))
            _debug_tensor_stats("metallic", locals().get("metallic", None))
            input('NaN detected!')

        """
        # --- 监控点 3：Backward 前 ---
        if hasattr(gaussians, 'env_map') and gaussians.env_map is not None:
            v_before_back = list(gaussians.env_map.parameters())[0]._version
            print(f"[DEBUG-INPLACE] 3. Backward前 Version: {v_before_back}")
        """

        
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipeline, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                #if not ref_gaussian_mode:
                #if opacity is not None and offset_selection_mask is not None:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                        
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                # gaussians.optimizer.zero_grad(set_to_none = False)
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_light_intensity.step()
                gaussians.optimizer_light_intensity.zero_grad(set_to_none=False)




            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        if iteration % 1000 == 0 and iteration > 10000 : 
            save_training_vis(viewpoint_cam, gaussians, background, render, pipeline, opt, iteration)

        # 每n步执行一次render_set，n默认10000 可通过yaml的render_save_interval设置
        # render_save_interval = getattr(opt, 'render_save_interval', 10000)
        save_list = [2001, 2999, 9999] # 25000, , opt.iterations_useflash, opt.iterations
        # 每n步执行一次render_set
        if iteration == 19999:
            export_root = os.path.join(dataset.model_path, "train", "ours_{}".format(iteration))
            save_gaussian_npz_snapshot(export_root, iteration, gaussians, None, logger)

        if iteration in save_list:
            flash_views = [cam for cam in scene.getTrainCameras() if getattr(cam, "isFlash", False)]
            # render_set(dataset.model_path, "train", iteration, opt.iterations_useflash, scene.getTrainCameras(), gaussians, pipeline, background, export_gaussians=save_gaussians_npz, logger=logger, view_idx_map=view_idx_map)
            max_views = 30
            # render_set(dataset.model_path, "train", iteration, opt.iterations_useflash, flash_views, gaussians, pipeline, background, export_gaussians=save_gaussians_npz, logger=logger, view_idx_map=view_idx_map) # , max_views=max_views)  # 
            
            
            render_set(dataset.model_path, "train", iteration, opt.iterations_useflash, scene.getTrainCameras(), gaussians, pipeline, background, export_gaussians=save_gaussians_npz, logger=logger, view_idx_map=view_idx_map, opt=opt)


            if iteration == opt.iterations_useflash:
                # Delegate mesh extraction, flash_expect generation and irradiance/spec precomputation (strict).
                precompute_indirect_flash(
                    iteration=iteration,
                    opt=opt,
                    dataset=dataset,
                    scene=scene,
                    gaussians=gaussians,
                    pipeline=pipeline,
                    background=background,
                    logger=logger,
                    view_idx_map=view_idx_map,
                    resume_iter=resume_iter,
                    ply_path=ply_path,
                    extract_mesh_snapshot=extract_mesh_snapshot,
                    prefilter_voxel=prefilter_voxel,
                    render=render,
                )

                # After indirect precompute, also precompute flashlight diffuse/spec (L0/L1) from normals (strict).
                transforms_path = os.path.join(dataset.source_path, "transforms_train.json")
                if not os.path.isfile(transforms_path):
                    raise FileNotFoundError(f"Missing transforms file: {transforms_path}")
                normal_dir = os.path.join(dataset.source_path, "normal")
                out_dir = os.path.join(dataset.model_path, "train", f"ours_{iteration}")
                precompute_direct_flash_from_files(
                    transforms_path=transforms_path,
                    normal_dir=normal_dir,
                    model_path=dataset.model_path,
                    iteration=iteration,
                    out_dir=out_dir,
                    device="cuda",
                )

                # ensure irradiance cache is populated as before
                cache_device = "cuda" if bool(getattr(opt, "cache_gpu", False)) else "cpu"
                load_precomp_cache(
                    opt.iterations_useflash,
                    dataset,
                    scene,
                    gaussians,
                    pipeline,
                    view_idx_map,
                    ind_diffuse_cache,
                    ind_spec_L0_cache,
                    ind_spec_L1_cache,
                    direct_diffuse_cache,
                    direct_spec_L0_cache,
                    direct_spec_L1_cache,
                    extract_mesh_snapshot,
                    logger,
                    cache_device=cache_device,
                )

                # Initialize stage-2 material MLP right after flash precompute to save VRAM in stage-1.
                # Stage-2/flash rendering is required to use NeRF(MaterialMLP) materials.
                if opt.iterations_useflash > 0:
                    gaussians.ensure_material_mlp(training_args=opt)
            logger.info(f"\n[ITER {iteration}] Saving rendered images via render_set")

            # If we are in/after flash stage, ensure precomputed irradiance/spec caches are loaded.
            # Otherwise flash_spec/flash_diff/flash_all export branches will be skipped and those
            # output dirs will be empty (common after resume/freeze changes).
            it_useflash = opt.iterations_useflash
            if it_useflash > 0 and iteration >= it_useflash:
                need_cache = (
                    len(ind_diffuse_cache) == 0
                    or len(direct_diffuse_cache) == 0
                    or len(ind_spec_L0_cache) == 0
                    or len(direct_spec_L0_cache) == 0
                )
                if need_cache:
                    cache_device = "cuda" if bool(getattr(opt, "cache_gpu", False)) else "cpu"
                    load_precomp_cache(
                        it_useflash,
                        dataset,
                        scene,
                        gaussians,
                        pipeline,
                        view_idx_map,
                        ind_diffuse_cache,
                        ind_spec_L0_cache,
                        ind_spec_L1_cache,
                        direct_diffuse_cache,
                        direct_spec_L0_cache,
                        direct_spec_L1_cache,
                        extract_mesh_snapshot,
                        logger,
                        cache_device=cache_device,
                    )
            
            
    # Export loss curves to TensorBoard figures and close the writer
    loss_tb_logger.save_curves(loss_dict, loss_save_interval, opt.iterations)
    loss_tb_logger.close()


    # logger.info(f"\n[ITER {opt.iterations}] Saving rendered images via render_set")
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # render_set(dataset.model_path, "train", opt.iterations, scene.getTrainCameras(), gaussians, pipeline, background)

    # diffusion model
    """
    main_args, args_list = dm_args()

    flash_cameras = [cam for cam in scene.getTrainCameras() if hasattr(cam, 'isFlash') and cam.isFlash]
    albedos = []
    rms = []
    normals = []
    # depths = []
    render_nonflash = []
    scale_w, scale_h = 0.0, 0.0
    world_positions = []
    light_distances = []
    for step, viewpoint_camera in enumerate(tqdm(flash_cameras), start=0):
        orig_w, orig_h = viewpoint_camera.image_width, viewpoint_camera.image_height
        resolution = main_args.resolution

        if orig_w < orig_h:
            new_w = resolution
            new_h = int(orig_h * (resolution / orig_w))
            new_h = (new_h // 8) * 8
        else:
            new_h = resolution
            new_w = int(orig_w * (resolution / orig_h))
            new_w = (new_w // 8) * 8

        scale_w = new_w / orig_w
        scale_h = new_h / orig_h

        # Resize the original image to new_w and new_h
        # viewpoint_camera.original_image: (C, H, W), torch.Tensor
        orig_img = viewpoint_camera.original_image
        resized_img = torch.nn.functional.interpolate(
            orig_img.unsqueeze(0),  # (1, C, H, W)
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (C, new_h, new_w)
        viewpoint_camera.original_image = resized_img.permute(1, 2, 0)  # (new_h, new_w, C)

        viewpoint_camera.image_width = new_w
        viewpoint_camera.image_height = new_h
        # 根据 FoVx/FoVy 推出旧焦距
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        focal_x = orig_w / (2 * tanfovx)
        focal_y = orig_h / (2 * tanfovy)

        # 不改焦距（假设是物理不变），重新推算新的 FoV
        FoVx_new = 2 * math.atan(new_w / (2 * focal_x))
        FoVy_new = 2 * math.atan(new_h / (2 * focal_y))

        # 更新 FoV
        viewpoint_camera.FoVx = FoVx_new
        viewpoint_camera.FoVy = FoVy_new

        # 焦距也重新计算一遍（和新 FoV 匹配）
        focal_length_x = new_w / (2 * math.tan(FoVx_new * 0.5))
        focal_length_y = new_h / (2 * math.tan(FoVy_new * 0.5))

        K = torch.tensor([
            [focal_length_x, 0, new_w / 2.0],
            [0, focal_length_y, new_h / 2.0],
            [0, 0, 1],
        ], device="cuda")
        
        # --- 2. 渲染图像并获取所需数据 ---
        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(viewpoint_camera, gaussians, pipeline, background)
        render_pkg = render(viewpoint_camera, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        # 读取各项
        albedo = render_pkg.get("albedo", None)
        roughness = render_pkg.get("roughness", None)
        metallic = render_pkg.get("metallic", None)
        depth = render_pkg.get("surf_depth", None)
        rendering = render_pkg.get("render", None).permute(1,2,0)
        normal = render_pkg.get("render_normal", None).permute(1,2,0)

        # (C, H, W) -> (B, C, H, W)
        albedo = albedo.unsqueeze(0)  # B=1
        zero_tensor = torch.zeros_like(roughness)
        rm = torch.cat([roughness, metallic, zero_tensor], dim=0).unsqueeze(0)

        albedos.append(albedo)
        rms.append(rm)
        normals.append(normal)
        # depths.append(depth)
        render_nonflash.append(rendering)


        # --- 3. 立即计算世界坐标和光照距离 ---
        
        # 使用当前循环得到的depth
        world_pos = get_world_pos(depth, K, viewpoint_camera.world_view_transform.transpose(0, 1))
        light_distance = (viewpoint_camera.camera_center - world_pos).norm(dim=-1, keepdim=True)
        world_positions.append(world_pos)
        light_distances.append(light_distance)

    # torchvision.utils.save_image(albedos[0].permute(2,0,1), "albedo_example.png")

    # 从exr图像读取mlp_intensity
    def load_first_channel_exr(path):
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        channels = list(header['channels'].keys())
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        first_channel = channels[0]
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        data = np.frombuffer(exr.channel(first_channel, FLOAT), dtype=np.float32).reshape(height, width)

        return data

    mlp_intensity_path = "/home/wdh/Project/scaffold_output/1120show_tabletop/diffusion-model_wallnight2_roughness/train/ours_30000/mlp_light_intensity/00058.exr" 
    mlp_intensity_np = load_first_channel_exr(mlp_intensity_path)
    mlp_intensity = torch.from_numpy(mlp_intensity_np).float().cuda()
    # 调用diffusion model 优化材质
    if scale_w > 0.0 and scale_h > 0.0:
        orig_h, orig_w = mlp_intensity.shape
        new_h, new_w = int(orig_h * scale_h), int(orig_w * scale_w)
        mlp_intensity = torch.nn.functional.interpolate(
            mlp_intensity.unsqueeze(0).unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
    # 保证 mlp_intensity shape 为 (H, W, 1)
    if mlp_intensity.dim() == 2:
        mlp_intensity = mlp_intensity.unsqueeze(-1)  # (H, W, 1)

    logger.info("Pre-computation for flashlight rendering is now integrated into the main loop.")

    my_main(
        model_path = dataset.model_path,
        main_args = main_args,
        args_list = args_list,
        scene = scene,
        albedos = albedos,
        rms = rms,
        normals = normals,
        world_positions = world_positions,
        light_distances = light_distances,
        mlp_intensity = mlp_intensity,
        render_nonflash = render_nonflash,
    )
    """

def plot_losses(loss_dict, loss_save_interval, total_iters, save_dir):
    """
    loss_dict: { "Total Loss": [...], "L1 Loss": [...], "SSIM Loss": [...], ... }
    每个 key 对应一个 list，长度 = total_iters // loss_save_interval
    """
    os.makedirs(save_dir, exist_ok=True)

    # 横轴: step
    steps = list(range(loss_save_interval, total_iters + 1, loss_save_interval))

    # ---------------------
    # 1. 所有曲线在一张图
    # ---------------------
    plt.figure(figsize=(10, 6))
    for name, values in loss_dict.items():
        if len(values) == 0:
            continue  # 跳过空的loss
        plt.plot(steps, values, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("All Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "all_losses.png"))
    plt.close()

    # ---------------------
    # 2. 每个曲线单独画
    # ---------------------
    for name, values in loss_dict.items():
        if len(values) == 0:
            continue  # 跳过空的loss
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, label=name, color="tab:blue")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{name} Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.png"))
        plt.close()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    albedo_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    # TODO 这里psnr算的可能有问题，因为没加flashlight
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

import matplotlib    
def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)

import torch.nn.functional as F

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration):
    with torch.no_grad():
        render_pkg = render_surfel(viewpoint_cam, gaussians, pipe, background, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        """
        if iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg["base_color_map"], 
                render_pkg["base_color_map1"],
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
                render_pkg["base_color_map1"],  
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
        grid = torchvision.utils.make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        torchvision.utils.save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        # if not initial_stage:
        if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
            env_dict = gaussians.render_env_map_2() 
        else:
        """
        env_dict = gaussians.render_env_map()

        grid = [
            env_dict["env1"].permute(2, 0, 1),
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = torchvision.utils.make_grid(grid, nrow=1, padding=10)
        torchvision.utils.save_image(grid, os.path.join("output/", f"{iteration:06d}_env.png"))



def render_set(model_path, name, iteration, iterations_useflash=0, views=None, gaussians=None, pipeline=None, background=None, export_gaussians=False, logger=None, max_flash=None, max_nonflash=None, view_idx_map=None, max_views=None, opt=None):
    # IMPORTANT:
    # Flash caches are keyed by `idx = view_idx_map[id(cam)]` where view_idx_map is built from
    # `scene.getTrainCameras()` at cache precompute/load time.
    # Therefore, render-time lookup must use the SAME mapping (not a local enumerate over a filtered/reordered list).
    views = list(views) if views is not None else []
    if view_idx_map is None:
        # Fallback: local enumerate. This will NOT match precomputed caches if `views` is a subset or reordered.
        view_idx_map = {id(v): i for i, v in enumerate(views)}
    if max_flash is not None or max_nonflash is not None:
        sel = []
        cnt_flash = 0
        cnt_non = 0
        for v in views:
            is_flash = getattr(v, "isFlash", False)
            if is_flash:
                if max_flash is not None and cnt_flash >= max_flash:
                    continue
                cnt_flash += 1
            else:
                if max_nonflash is not None and cnt_non >= max_nonflash:
                    continue
                cnt_non += 1
            sel.append(v)
        views = sel
        if logger is not None:
            logger.info(f"Render subset: {len(views)} views (flash {cnt_flash}, non-flash {cnt_non})")

    if max_views is not None and max_views > 0:
        views = views[:max_views]
        if logger is not None:
            logger.info(f"Render subset: first {len(views)} views (max_views={max_views})")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    # normal_cam_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_cam")
    albedo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "albedo")
    roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness")
    metallic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "metallic")
    surf_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "surf_depth")
    surf_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "surf_normal")
    # flashlight_defer_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flashlight_defer") 
    render_flash_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_flash")
    mlp_intensity_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mlp_light_intensity")
    render_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_alpha")
    # n_dot_l_path = os.path.join(model_path, name, "ours_{}".format(iteration), "n_dot_l")
    light_dir_path = os.path.join(model_path, name, "ours_{}".format(iteration), "light_dir")
    world_pos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "world_pos")
    attenuation_path = os.path.join(model_path, name, "ours_{}".format(iteration), "attenuation")
    light_distance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "light_distance")
    flash_spec_dir_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flash_spec_dir_color")
    flash_spec_ind_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flash_spec_ind_color")

    flash_diff_dir_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flash_diff_dir_color")
    flash_diff_ind_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flash_diff_ind_color")

    flash_all_color = os.path.join(model_path, name, "ours_{}".format(iteration), "flash_all_color")

    makedirs(surf_depth_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(albedo_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    # makedirs(normal_cam_path, exist_ok=True)
    makedirs(roughness_path, exist_ok=True)
    makedirs(metallic_path, exist_ok=True)
    makedirs(surf_normal_path, exist_ok=True)
    # makedirs(flashlight_defer_path, exist_ok=True)
    makedirs(render_flash_path, exist_ok=True)
    makedirs(mlp_intensity_path, exist_ok=True)
    makedirs(render_alpha_path, exist_ok=True)
    # makedirs(n_dot_l_path, exist_ok=True)
    makedirs(light_dir_path, exist_ok=True)
    makedirs(world_pos_path, exist_ok=True)
    makedirs(attenuation_path, exist_ok=True)
    makedirs(light_distance_path, exist_ok=True)
    # 合成间接光乘积保存目录
    # makedirs(indirect_flash_path, exist_ok=True)
    makedirs(flash_spec_dir_color_path, exist_ok=True)
    makedirs(flash_spec_ind_color_path, exist_ok=True)
    makedirs(flash_diff_ind_color_path, exist_ok=True)
    makedirs(flash_diff_dir_color_path, exist_ok=True)
    makedirs(flash_all_color, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}

    if export_gaussians:
        export_root = os.path.join(model_path, name, "ours_{}".format(iteration))
        save_gaussian_npz_snapshot(export_root, iteration, gaussians, None, logger)

    def save_render_outputs(render_pkg, view, idx):
        """
        保存render_pkg中的各项到对应目录。
        """
        global ind_spec_L0_cache, ind_spec_L1_cache, ind_diffuse_cache
        fname = '{:04d}.png'.format(idx)
        flash_diffuse = None
        flash_spec = None
        flash_all = None
        # 渲染结果
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendering_srgb = rendering #linear2srgb(rendering)
        torchvision.utils.save_image(rendering_srgb, os.path.join(render_path, fname))
        # GT
        gt = view.original_image[0:3, :, :]
        gt_srgb = gt #linear2srgb(gt)
        torchvision.utils.save_image(gt_srgb, os.path.join(gts_path, fname))
        # error map
        errormap = (rendering_srgb - gt_srgb).abs()
        torchvision.utils.save_image(errormap, os.path.join(error_path, fname))
        # normal
        normal = render_pkg.get("render_normal", None)
        if normal is not None:
            # normalize normal
            normal = torch.nn.functional.normalize(normal, dim=0) 
            normals_vis = (normal + 1.0) * 0.5
            normals_vis = normals_vis.clamp(0.0, 1.0)
            torchvision.utils.save_image(normals_vis, os.path.join(normal_path, fname))
        # normal_cam = render_pkg.get("normal_cam", None)
        # if normal_cam is not None:
        #     normal_cam = torch.nn.functional.normalize(normal_cam, dim=0)
        #     normal_cam_vis = (normal_cam + 1.0) * 0.5
        #     normal_cam_vis = normal_cam_vis.clamp(0.0, 1.0)
        #     torchvision.utils.save_image(normal_cam_vis, os.path.join(normal_cam_path, fname))
        # surf_normal = render_pkg.get("surf_normal", None)
        # if surf_normal is not None:
        #     surf_normal = torch.nn.functional.normalize(surf_normal, dim=0)
        #     surf_normal_vis = (surf_normal + 1.0) * 0.5
        #     surf_normal_vis = surf_normal_vis.clamp(0.0, 1.0)
        #     torchvision.utils.save_image(surf_normal_vis, os.path.join(surf_normal_path, fname))
        mlp_intensity = render_pkg.get("mlp_intensity", None)
        if mlp_intensity is not None:
            mlp_intensity = render_pkg["mlp_intensity"].squeeze(-1).detach().cpu().numpy()
            if mlp_intensity.ndim == 3 and mlp_intensity.shape[0] == 1:
                mlp_intensity = mlp_intensity[0]
            save_exr_with_fallback(
                os.path.join(mlp_intensity_path, '{0:04d}'.format(idx) + ".exr"),
                mlp_intensity,
                logger,
            )

            mlp_intensity_vis = render_pkg["mlp_intensity"].squeeze(-1) / render_pkg["mlp_intensity"].squeeze(-1).max()
            torchvision.utils.save_image(mlp_intensity_vis, os.path.join(mlp_intensity_path, '{0:04d}'.format(idx) + ".png"))

        attenuation = render_pkg.get("attenuation", None)
        if attenuation is not None:
            attenuation = render_pkg["attenuation"].squeeze(-1).detach().cpu().numpy()
            if attenuation.ndim == 3 and attenuation.shape[0] == 1:
                attenuation = attenuation[0]
            save_exr_with_fallback(
                os.path.join(attenuation_path, '{0:04d}'.format(idx) + ".exr"),
                attenuation,
                logger,
            )

            attenuation_vis = render_pkg["attenuation"].squeeze(-1) / render_pkg["attenuation"].squeeze(-1).max()
            torchvision.utils.save_image(attenuation_vis, os.path.join(attenuation_path, '{0:04d}'.format(idx) + ".png"))


        # light_distance
        light_distance = render_pkg.get("light_distance", None)
        if light_distance is not None:
            light_distance_vis = render_pkg["light_distance"].squeeze(-1) / render_pkg["light_distance"].squeeze(-1).max() 
            torchvision.utils.save_image(light_distance_vis, os.path.join(light_distance_path, '{0:04d}'.format(idx) + ".png"))

            light_distance = render_pkg["light_distance"].squeeze(-1).detach().cpu().numpy()
            if light_distance.ndim == 3 and light_distance.shape[0] == 1:
                light_distance = light_distance[0]
            save_exr_with_fallback(
                os.path.join(light_distance_path, '{0:04d}'.format(idx) + ".exr"),
                light_distance,
                logger,
            )
        
        light_dir = render_pkg.get("light_dir", None)
        if light_dir is not None:
            light_dir = (light_dir + 1.0) * 0.5
            light_dir = light_dir.clamp(0.0, 1.0)
            torchvision.utils.save_image(light_dir, os.path.join(light_dir_path, fname))


        render_alpha = render_pkg.get("render_alpha", None)
        if render_alpha is not None:
            render_alpha = torch.clamp(render_alpha, 0.0, 1.0)
            torchvision.utils.save_image(render_alpha, os.path.join(render_alpha_path, fname))

        # albedo
        albedo = render_pkg.get("albedo", None)
        if albedo is not None:
            albedo = torch.clamp(albedo, 0.0, 1.0)
            torchvision.utils.save_image(albedo, os.path.join(albedo_path, fname))
        
        # base_color
        base_color = render_pkg.get("base_color_map1", None) 
        if base_color is None:
             base_color = render_pkg.get("base_color_map", None)
        if base_color is not None:
            base_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "base_color")
            os.makedirs(base_color_path, exist_ok=True)
            
            base_color = torch.clamp(base_color, 0.0, 1.0)
            torchvision.utils.save_image(base_color, os.path.join(base_color_path, fname))

        # Specular 
        specular = render_pkg.get("specular_map", None)
        if specular is not None:
            # 确保目录存在
            specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "specular")
            os.makedirs(specular_path, exist_ok=True)
            specular_vis = torch.clamp(specular, 0.0, 1.0)
            torchvision.utils.save_image(specular_vis, os.path.join(specular_path, fname))

        # 如果是 flash 视角，直接使用 diffuse_cache 中的 irradiance，并基于缓存的 spec 插值
        if getattr(view, "isFlash", False) and iteration >= iterations_useflash:
            cam_idx = view_idx_map.get(id(view), None)
            flash_fname = '{:04d}.png'.format(cam_idx if cam_idx is not None else idx)
            ind_diffuse = ind_diffuse_cache.get(cam_idx, None) if cam_idx is not None else None
            direct_diffuse = direct_diffuse_cache.get(cam_idx, None) if cam_idx is not None else None

            metallic_tensor = render_pkg.get("metallic", None)
            mlp_intensity = render_pkg.get("mlp_intensity", None)
            attenuation = render_pkg.get("attenuation", None)

            # Normalize inputs to [C,H,W] and broadcast safely before combining
            if ind_diffuse is not None and albedo is not None and metallic_tensor is not None and mlp_intensity is not None and attenuation is not None:
                ind_diffuse = _ensure_chw(ind_diffuse)
                direct_diffuse = _ensure_chw(direct_diffuse)
                albedo = _ensure_chw(albedo)
                metallic = _ensure_chw(metallic_tensor)
                mlp_intensity = _ensure_chw(mlp_intensity)
                attenuation = _ensure_chw(attenuation)

                # Move to same device as albedo if possible
                device = albedo.device if isinstance(albedo, torch.Tensor) else None
                if isinstance(ind_diffuse, torch.Tensor) and device is not None:
                    ind_diffuse = ind_diffuse.to(device)
                if isinstance(direct_diffuse, torch.Tensor) and device is not None:
                    direct_diffuse = direct_diffuse.to(device)
                if isinstance(metallic, torch.Tensor) and device is not None:
                    metallic = metallic.to(device)
                if isinstance(mlp_intensity, torch.Tensor) and device is not None:
                    mlp_intensity = mlp_intensity.to(device)
                if isinstance(attenuation, torch.Tensor) and device is not None:
                    attenuation = attenuation.to(device)

                # Ensure channel alignment: prefer 3-channel color for fd/alb
                # If a tensor has shape [1,H,W] but other is [3,H,W] expand it
                def _maybe_expand(t_src, t_ref):
                    if t_src is None or t_ref is None:
                        return t_src
                    if not (isinstance(t_src, torch.Tensor) and isinstance(t_ref, torch.Tensor)):
                        return t_src
                    if t_src.shape[0] == 1 and t_ref.shape[0] == 3:
                        return t_src.expand(3, *t_src.shape[1:])
                    return t_src

                ind_diffuse = _maybe_expand(ind_diffuse, albedo)
                direct_diffuse = _maybe_expand(direct_diffuse, albedo)
                metallic = _maybe_expand(metallic, albedo)
                mlp_intensity = _maybe_expand(mlp_intensity, albedo)
                attenuation = _maybe_expand(attenuation, albedo)

                # Final type/device consistency
                if isinstance(ind_diffuse, torch.Tensor):
                    ind_diffuse = ind_diffuse.float()
                if isinstance(direct_diffuse, torch.Tensor):
                    direct_diffuse = direct_diffuse.float()
                if isinstance(albedo, torch.Tensor):
                    albedo = albedo.float()
                if isinstance(metallic, torch.Tensor):
                    metallic = metallic.float()
                if isinstance(mlp_intensity, torch.Tensor):
                    mlp_intensity = mlp_intensity.float()
                if isinstance(attenuation, torch.Tensor):
                    attenuation = attenuation.float()

                # Avoid division by zero
                if isinstance(attenuation, torch.Tensor):
                    att_safe = attenuation + 1e-6
                else:
                    att_safe = attenuation

                ind_flash_diff = ind_diffuse * albedo * (1.0 - metallic)
                ind_flash_diff_np = np.clip(ind_flash_diff.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
                direct_flash_diff = direct_diffuse * albedo * (1.0 - metallic) * mlp_intensity / att_safe
                direct_flash_diff_np = np.clip(direct_flash_diff.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
                flash_diffuse =  direct_flash_diff_np + ind_flash_diff_np # TODO Ind
                direct_flash_diff_np = np.clip(np.asarray(direct_flash_diff_np), 0.0, 1.0)
                imageio.imwrite(os.path.join(flash_diff_dir_color_path, flash_fname), (linear2srgb(direct_flash_diff_np) * 255.0).astype(np.uint8))
                ind_flash_diff_np = np.clip(np.asarray(ind_flash_diff_np), 0.0, 1.0)

                imageio.imwrite(os.path.join(flash_diff_ind_color_path, flash_fname), (linear2srgb(ind_flash_diff_np) * 255.0).astype(np.uint8))

            # 直接使用缓存的 spec_L0/L1 插值
            rough_map = render_pkg.get("roughness", None)
            if rough_map is None:
                rough_map = torch.zeros_like(albedo[:1])
            # ensure rough_map in expected [1,H,W] or [H,W]
            rough_map = _ensure_chw(rough_map)
            if rough_map.shape[0] == 3:
                rough_map = rough_map[:1]
            ind_L0_s, ind_L1_s, direct_L0_s, direct_L1_s = _interp_spec(cam_idx, rough_map) if cam_idx is not None else (None, None, None, None)
            if ind_L0_s is not None and ind_L1_s is not None and direct_L0_s is not None and direct_L1_s is not None:
                ind_L0 = ind_L0_s
                ind_L1 = ind_L1_s
                direct_L0 = direct_L0_s
                direct_L1 = direct_L1_s

                metallic_spec = _ensure_chw(render_pkg.get("metallic", None))
                if metallic_spec is None:
                    metallic_spec = torch.zeros_like(albedo[:1])
                # expand channels if needed
                if metallic_spec.shape[0] == 1 and albedo.shape[0] == 3:
                    metallic_spec = metallic_spec.expand(3, *metallic_spec.shape[1:])

                mlp_intensity = _ensure_chw(mlp_intensity) if mlp_intensity is not None else None
                attenuation = _ensure_chw(attenuation) if attenuation is not None else None
                # Move L0/L1 to same device as albedo
                device = albedo.device if isinstance(albedo, torch.Tensor) else None
                if device is not None:
                    ind_L0 = ind_L0.to(device)
                    ind_L1 = ind_L1.to(device)
                    direct_L0 = direct_L0.to(device)
                    direct_L1 = direct_L1.to(device)
                    metallic_spec = metallic_spec.to(device)
                    if isinstance(mlp_intensity, torch.Tensor):
                        mlp_intensity = mlp_intensity.to(device)
                    if isinstance(attenuation, torch.Tensor):
                        attenuation = attenuation.to(device)

                mlp_intensity = mlp_intensity if mlp_intensity is not None else 1.0
                attenuation = attenuation if attenuation is not None else 1.0

                ks = albedo.detach() * metallic_spec + 0.04 * (1.0 - metallic_spec)
                direct_flash_spec = (ks * direct_L0 + direct_L1) * mlp_intensity / (attenuation + 1e-6)
                ind_flash_spec = (ks * ind_L0 + ind_L1)
                flash_spec = direct_flash_spec + ind_flash_spec # TODO Ind
                flash_spec_out = os.path.join(flash_spec_dir_color_path, flash_fname)
                torchvision.utils.save_image(linear2srgb(direct_flash_spec), flash_spec_out)

                flash_spec_out = os.path.join(flash_spec_ind_color_path, flash_fname)
                torchvision.utils.save_image(linear2srgb(ind_flash_spec), flash_spec_out)
            
            if flash_diffuse is not None and flash_spec is not None:
                flash_diffuse = torch.from_numpy(flash_diffuse).permute(2, 0, 1)
                flash_diffuse = flash_diffuse.to(device=flash_spec.device, dtype=flash_spec.dtype)
                flash_all = flash_diffuse + flash_spec # TODO spec
                flash_all_out = os.path.join(flash_all_color, flash_fname)
                torchvision.utils.save_image(linear2srgb(flash_all), flash_all_out)
            
            if flash_all is not None:
                render_flash = linear2srgb(rendering + flash_all)
                errormap = (render_flash - gt_srgb).abs()
                torchvision.utils.save_image(errormap, os.path.join(error_path, '{:04d}.png'.format(idx)))
                torchvision.utils.save_image(torch.clamp(render_flash, 0.0, 1.0), os.path.join(render_flash_path, fname))

        # roughness
        roughness = render_pkg.get("roughness", None)
        if roughness is not None:
            roughness = torch.clamp(roughness, 0.0, 1.0)
            torchvision.utils.save_image(roughness, os.path.join(roughness_path, fname))

        # metallic
        metallic = render_pkg.get("metallic", None)
        if metallic is not None:
            metallic = torch.clamp(metallic, 0.0, 1.0)
            torchvision.utils.save_image(metallic, os.path.join(metallic_path, fname))
                
        # surf_depth
        surf_depth = render_pkg.get("surf_depth", None)
        if surf_depth is not None:
            surf_depth_vis = render_pkg["surf_depth"].squeeze(-1) / render_pkg["surf_depth"].squeeze(-1).max() 
            torchvision.utils.save_image(surf_depth_vis, os.path.join(surf_depth_path, '{0:04d}'.format(idx) + ".png"))

            surf_depth = render_pkg["surf_depth"].squeeze(-1).detach().cpu().numpy()
            if surf_depth.ndim == 3 and surf_depth.shape[0] == 1:
                surf_depth = surf_depth[0]
            save_exr_with_fallback(
                os.path.join(surf_depth_path, '{0:04d}'.format(idx) + ".exr"),
                surf_depth,
                logger,
            )

    # Evaluation render should not build autograd graphs (saves a lot of VRAM).
    prev_is_training = True
    try:
        prev_is_training = bool(getattr(gaussians.get_color_mlp, "training", True))
    except Exception:
        prev_is_training = True

    gaussians.eval()
    try:
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize(); t_start = time.time()
            with torch.no_grad():
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, require_grad=False)
                """
                # ================= [触发 NPY 保存] =================
                if iteration == 2000 and idx == 0:
                    # 强制同时跑两边，带上名字，把它们的值存下来
                    _ = render_debug(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, isDetach=True, debug_name="base_render")
                    _ = render_surfel_debug(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, opt=opt, debug_name="surfel_render")
                # =======================================================

                render_pkg = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    visible_mask=voxel_visible_mask,
                    isDetach=True,
                )
                """
                if iteration >= 10001:
                
                    # Ref-Gaussian 
                    render_pkg = render_surfel(
                        view,
                        gaussians,
                        pipeline,
                        background,
                        visible_mask=voxel_visible_mask,
                        opt=opt,
                    )
                
                else:
                
                    render_pkg = render(
                        view,
                        gaussians,
                        pipeline,
                        background,
                        visible_mask=voxel_visible_mask,
                        isDetach=True, 
                    )

                # ================= [ 保存 Visibility Map] =================
                if iteration == 20000:  # 我们只在 2w 步一刻保存
                    # 尝试从 render_pkg 中提取 visibility
                    vis_map = render_pkg.get("visibility", None)
                    
                    if vis_map is not None:
                        
                        # 创建 visibility 保存文件夹
                        vis_dir = os.path.join(model_path, name, f"ours_{iteration}", "visibility")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # visibility 通常是 [1, H, W]，数值在 0.0 ~ 1.0 之间
                        # 直接保存，0.0会变成纯黑(被遮挡)，1.0会变成纯白(可见)
                        vis_save_path = os.path.join(vis_dir, f"{idx:05d}.png")
                        torchvision.utils.save_image(vis_map, vis_save_path)
                        
                        if idx == 0: # 只打印一次提示
                            print(f"\n[DEBUG-VIS] 成功提取到 Visibility Map，正在保存至 {vis_dir}")
                    else:
                        if idx == 0:
                            print(f"\n[DEBUG-VIS] 警告: render_pkg 中没有找到 visibility 相关的键值！请检查 get_specular_color_surfel 是否返回了它。当前 keys: {list(render_pkg.keys())}")
                # ===============================================================
                
            torch.cuda.synchronize(); t_end = time.time()

            # Low-frequency render/export: free cached blocks to reduce OOM risk.
            if (idx % 4) == 3:
                torch.cuda.empty_cache()

            t_list.append(t_end - t_start)
            visible_count = (render_pkg["radii"] > 0).sum()
            visible_count_list.append(visible_count)
            name_list.append('{:04d}.png'.format(idx))
            save_render_outputs(render_pkg, view, idx)
            # save_render_outputs(render_surfel_pkg, view, idx)
            per_view_dict['{:04d}.png'.format(idx)] = visible_count.item()
    finally:
        if prev_is_training:
            gaussians.train()
        else:
            gaussians.eval()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    
    return t_list, visible_count_list


@torch.no_grad()
def extract_mesh_snapshot(model_path, iteration, scene, gaussians, pipeline, dataset, logger=None, num_cluster=50):
    """在指定迭代导出无颜色的网格 ply。"""
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    # 对齐 render.py 的导出流程：锁定到 eval、active_sh_degree=0 并保持 MLP 为 eval
    prev_sh = getattr(gaussians, "active_sh_degree", None)
    gaussians.active_sh_degree = 0
    gaussians.eval()
    for module in [gaussians.get_color_mlp, gaussians.get_opacity_mlp, gaussians.get_albedo_mlp, gaussians.get_rm_mlp]:
        if module is not None:
            module.eval()

    extractor = GaussianExtractor(gaussians, render, prefilter_voxel, pipeline, bg_color=bg_color)
    extractor.reconstruction(scene.getTrainCameras())

    # 与 render.py 一致的 depth_trunc / voxel_size / sdf_trunc 计算
    if getattr(extractor, "radius", None) is not None:
        depth_trunc = extractor.radius * 2.0
        depth_range = getattr(extractor, "depth_range", None)
        if depth_range is not None:
            depth_max = depth_range[1]
            depth_trunc = max(depth_trunc, depth_max * 1.05)
    else:
        depth_trunc = 3.0
    mesh_res = 1024
    voxel_size_TSDF = -1
    voxel_size = (depth_trunc / mesh_res) if voxel_size_TSDF < 0 else voxel_size_TSDF
    sdf_trunc = 3.0 * voxel_size

    mesh = extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    # mesh = extractor.extract_mesh_unbounded(resolution=mesh_res)
    mesh_dir = os.path.join(model_path, "train", f"ours_{iteration}", "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, "fuse.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=False)

    # 后处理并保存彩色网格（不再清空颜色）
    mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
    post_path = os.path.join(mesh_dir, "fuse_post.ply")
    o3d.io.write_triangle_mesh(post_path, mesh_post, write_triangle_uvs=False)

    # 恢复训练状态
    if prev_sh is not None:
        gaussians.active_sh_degree = prev_sh
    gaussians.train()

    if logger is not None:
        logger.info(f"[ITER {iteration}] Mesh saved to {mesh_path}, post-processed mesh saved to {post_path}")
    return mesh_path


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None, export_gaussians=False, max_flash=None, max_nonflash=None, max_views=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.feat_albedo_dim,
            dataset.feat_rm_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            mlp_material_feature_dim=dataset.mlp_material_feature_dim,
            mlp_material_encoding=dataset.mlp_material_encoding,
            mlp_material_hidden_dim=getattr(dataset, "mlp_material_hidden_dim", 256),
            mlp_material_num_hidden_layers=getattr(dataset, "mlp_material_num_hidden_layers", 6),
            mlp_material_hash_n_levels=dataset.mlp_material_hash_n_levels,
            mlp_material_hash_n_features_per_level=dataset.mlp_material_hash_n_features_per_level,
            mlp_material_hash_log2_hashmap_size=dataset.mlp_material_hash_log2_hashmap_size,
            mlp_material_hash_base_resolution=dataset.mlp_material_hash_base_resolution,
            mlp_material_hash_finest_resolution=dataset.mlp_material_hash_finest_resolution,
            mlp_material_hash_per_level_scale=getattr(dataset, "mlp_material_hash_per_level_scale", None),
            mlp_material_hash_bbox_pad=dataset.mlp_material_hash_bbox_pad,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # bg_color = [random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            view_idx_map = {id(v): i for i, v in enumerate(scene.getTrainCameras())}
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, dataset.iterations_useflash if hasattr(dataset, 'iterations_useflash') else 0, scene.getTrainCameras(), gaussians, pipeline, background, export_gaussians=export_gaussians, logger=logger, max_flash=max_flash, max_nonflash=max_nonflash, view_idx_map=view_idx_map, max_views=max_views)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            view_idx_map = {id(v): i for i, v in enumerate(scene.getTrainCameras())}
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, dataset.iterations_useflash if hasattr(dataset, 'iterations_useflash') else 0, scene.getTestCameras(), gaussians, pipeline, background, export_gaussians=export_gaussians, logger=logger, max_flash=max_flash, max_nonflash=max_nonflash, view_idx_map=view_idx_map, max_views=max_views)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        # Avoid leaking file descriptors when reading lots of images.
        with Image.open(renders_dir / fname) as render_im:
            render_t = tf.to_tensor(render_im).unsqueeze(0)[:, :3, :, :].cuda()
        with Image.open(gt_dir / fname) as gt_im:
            gt_t = tf.to_tensor(gt_im).unsqueeze(0)[:, :3, :, :].cuda()
        renders.append(render_t)
        gts.append(gt_t)
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6099)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument('--no_save_gaussians_npz', action='store_true', help="在render_set阶段不额外导出Gaussian参数npz快照")
    parser.add_argument('--resume_gaussians_npz', type=str, default=None, help="从指定的Gaussian npz快照恢复继续训练")
    parser.add_argument('--render_subset_flash', type=int, default=-1, help="渲染集时最多输出多少张flash视角（<0 表示全部）")
    parser.add_argument('--render_subset_nonflash', type=int, default=-1, help="渲染集时最多输出多少张非flash视角（<0 表示全部）")
    # parser.add_argument("--flash", type=bool, default = False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Set default for save_gaussians_npz to True
    args.save_gaussians_npz = not args.no_save_gaussians_npz
    
    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    # if args.gpu != '-1':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #     os.system("echo $CUDA_VISIBLE_DEVICES")
    #     logger.info(f'using GPU {args.gpu}')


    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # TODO 加了accelerate
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # if accelerator.is_local_main_process:
    #     network_gui.init(args.ip, args.port)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(bool(args.detect_anomaly))
    
    # training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        dataset,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        wandb,
        logger,
        gaussian_resume_path=args.resume_gaussians_npz,
        save_gaussians_npz=args.save_gaussians_npz,
    )
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            dataset,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            wandb=wandb,
            logger=logger,
            ply_path=new_ply_path,
            gaussian_resume_path=args.resume_gaussians_npz,
            save_gaussians_npz=args.save_gaussians_npz,
        )

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    subset_flash = None if args.render_subset_flash < 0 else args.render_subset_flash
    subset_nonflash = None if args.render_subset_nonflash < 0 else args.render_subset_nonflash
    max_views = None if getattr(args, "render_save_n_views", -1) is None or getattr(args, "render_save_n_views", -1) <= 0 else int(getattr(args, "render_save_n_views"))
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger, export_gaussians=args.save_gaussians_npz, max_flash=subset_flash, max_nonflash=subset_nonflash, max_views=max_views)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")


