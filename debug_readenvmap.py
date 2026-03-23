import os
import argparse
import atexit
import sys
from typing import TextIO

import numpy as np
import torch
import imageio

from scene.light import EnvLight, inverse_softplus


# my add: write logs to txt
class _Tee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._streams[0], "isatty")())
        except Exception:
            return False


def _enable_log_tee(log_path: str) -> None:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(old_stdout, log_f)
    sys.stderr = _Tee(old_stderr, log_f)

    def _cleanup() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_stdout, old_stderr
        try:
            log_f.close()
        except Exception:
            pass

    atexit.register(_cleanup)


# my add: read_envmap
# This script debugs GT EXR envmap reading + latlong->cubemap conversion + resampling.
# It follows the same code paths as training:
# - imageio.imread(.exr)
# - optional horizontal flip (to match direction2 convention in this repo)
# - HDR branch: inverse_softplus on linear HDR -> latlong_to_cubemap -> sampling -> softplus


def _read_exr_level0_openexr(path: str) -> np.ndarray:
    """Read EXR level0 as float32 HxWx3 using OpenEXR.

    This avoids imageio/opencv sometimes returning a lower mip level.
    """
    import OpenEXR  # type: ignore
    import Imath  # type: ignore

    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header["dataWindow"]
    width = int(dw.max.x - dw.min.x + 1)
    height = int(dw.max.y - dw.min.y + 1)

    # Request float32 output regardless of stored HALF/FLOAT.
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    ch_names = set(header.get("channels", {}).keys())
    # Most HDRIs are RGB; accept lowercase too.
    candidates = [("R", "G", "B"), ("r", "g", "b")]
    rgb = None
    for r, g, b in candidates:
        if r in ch_names and g in ch_names and b in ch_names:
            rgb = (r, g, b)
            break
    if rgb is None:
        raise ValueError(f"EXR missing RGB channels, found: {sorted(ch_names)}")

    r = np.frombuffer(f.channel(rgb[0], pt), dtype=np.float32).reshape(height, width)
    g = np.frombuffer(f.channel(rgb[1], pt), dtype=np.float32).reshape(height, width)
    b = np.frombuffer(f.channel(rgb[2], pt), dtype=np.float32).reshape(height, width)
    img = np.stack([r, g, b], axis=-1)
    return img


def _read_latlong_hdr(path: str) -> np.ndarray:
    """Read HDR latlong (.exr or .hdr) as float32 HxWx3."""
    p = str(path)
    if p.lower().endswith(".exr"):
        try:
            img = _read_exr_level0_openexr(p)
            return img.astype(np.float32, copy=False)
        except Exception as e:
            print(f"[WARN] OpenEXR read failed, fallback to imageio.imread: {e}")

    # Fallback: imageio (may use opencv plugin).
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    img = imageio.imread(p)
    if isinstance(img, np.ndarray) and (img.dtype != np.float32):
        img = img.astype(np.float32, copy=False)
    if isinstance(img, np.ndarray) and (img.ndim == 3) and (img.shape[-1] > 3):
        img = img[..., :3]
    return img


def _tonemap_env_for_vis(x_hwc: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Match train_debug.py::_tonemap_env_for_vis (for PNG visualization only)."""
    x = x_hwc
    x = np.clip(x, 0.0, None)
    x = x * float(exposure)
    x = x / (1.0 + x)  # Reinhard
    x = np.power(x, 1.0 / 2.2)  # display gamma
    x = np.clip(x, 0.0, 1.0)
    return x


def _save_png(path: str, img_hwc_01: np.ndarray) -> None:
    arr = np.clip(img_hwc_01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, arr)


def _print_stats(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.all():
        x = x[finite]
    if x.size == 0:
        print(f"[{name}] empty")
        return
    p = np.percentile(x, [0.0, 50.0, 90.0, 99.0, 99.9, 100.0])
    print(
        f"[{name}] dtype={x.dtype} shape={tuple(x.shape)} "
        f"min={p[0]:.6g} med={p[1]:.6g} p90={p[2]:.6g} p99={p[3]:.6g} p99.9={p[4]:.6g} max={p[5]:.6g}"
    )


def _print_quality(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    total = x.size
    if total == 0:
        return
    finite = np.isfinite(x)
    finite_ratio = float(finite.mean())
    neg_ratio = float((x < 0).mean()) if np.issubdtype(x.dtype, np.floating) else 0.0
    print(f"[{name}] finite_ratio={finite_ratio:.6f} neg_ratio={neg_ratio:.6f}")


def _print_diff_stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"[{name}] shape mismatch: {a.shape} vs {b.shape}")
        return
    d = np.abs(a - b)
    _print_stats(name, d)


def _find_single_exr(source_path: str) -> str:
    env_dir = os.path.join(source_path, "envmap")
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(f"envmap folder not found: {env_dir}")
    exrs = [
        os.path.join(env_dir, f)
        for f in os.listdir(env_dir)
        if os.path.isfile(os.path.join(env_dir, f)) and f.lower().endswith(".exr")
    ]
    exrs.sort()
    if len(exrs) == 0:
        raise FileNotFoundError(f"no .exr found under: {env_dir}")
    if len(exrs) > 1:
        print(f"[WARN] multiple .exr found under {env_dir}, using first: {exrs[0]}")
    return exrs[0]


# Copied from scene/gaussian_model.py (keep identical).
def get_env_direction1(H: int, W: int, device: str = "cuda"):
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device),
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device),
        indexing="ij",
    )
    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    env_directions = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    return env_directions


# Copied from scene/gaussian_model.py (keep identical).
def get_env_direction2(H: int, W: int, device: str = "cuda"):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device=device),
        torch.linspace(0, torch.pi, H, device=device),
        indexing="xy",
    )
    env_directions = torch.stack(
        (torch.sin(gy) * torch.cos(gx), torch.sin(gy) * torch.sin(gx), torch.cos(gy)),
        dim=-1,
    )
    return env_directions


def main() -> int:
    parser = argparse.ArgumentParser("debug_readenvmap")
    # my add: read_envmap
    # Default to the user's GT EXR path for quick debugging.
    parser.add_argument(
        "--exr_path",
        type=str,
        default="/nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/diffuse/envmap/kloofendal_48d_partly_cloudy_puresky_2k.exr",
        help="Path to a single latlong HDR EXR envmap.",
    )
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        default="",
        help="Dataset root (used only when --exr_path is empty; will search under <source_path>/envmap/*.exr).",
    )
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--max_res", type=int, default=128)
    parser.add_argument("--latlong_mode", type=str, default="direction2", choices=["direction1", "direction2"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    # my add: read_envmap
    exr_path = str(getattr(args, "exr_path", "") or "").strip()
    if not exr_path:
        source_path = str(getattr(args, "source_path", "") or "").strip()
        if not source_path:
            raise ValueError("Please provide --exr_path or -s/--source_path")
        exr_path = _find_single_exr(source_path)

    out_dir = args.out_dir
    if not out_dir:
        base = os.path.splitext(os.path.basename(exr_path))[0]
        # Write into a stable directory (no timestamps) so repeated runs overwrite.
        out_dir = os.path.join("./output", f"debug_readenvmap_{base}")
    os.makedirs(out_dir, exist_ok=True)

    # my add: write logs to txt
    log_path = os.path.join(out_dir, "debug_readenvmap_log.txt")
    _enable_log_tee(log_path)
    print("=" * 80)
    print("debug_readenvmap")
    print(f"log_path={log_path}")
    print(f"exr_path={exr_path}")
    print(f"out_dir={out_dir}")
    print("=" * 80)

    # --- A) EXR direct read (no cubemap conversion) ---
    raw = _read_latlong_hdr(exr_path)
    raw_scaled = raw.astype(np.float32, copy=False) * float(args.scale)
    raw_clamped = np.clip(raw_scaled, 1e-6, None)
    raw_preact = inverse_softplus(torch.from_numpy(raw_clamped)).cpu().numpy()

    flipped = np.flip(raw, axis=1).copy()
    flipped_scaled = flipped.astype(np.float32, copy=False) * float(args.scale)
    flipped_clamped = np.clip(flipped_scaled, 1e-6, None)
    flipped_preact = inverse_softplus(torch.from_numpy(flipped_clamped)).cpu().numpy()

    _print_stats("exr_raw", raw)
    _print_stats("exr_flipped", flipped)
    _print_stats("exr_raw_scaled", raw_scaled)
    _print_stats("exr_raw_preact", raw_preact)
    _print_stats("exr_flipped_scaled", flipped_scaled)
    _print_stats("exr_flipped_preact", flipped_preact)
    _print_quality("exr_raw", raw)
    _print_quality("exr_flipped", flipped)
    _print_quality("exr_raw_preact", raw_preact)
    _print_quality("exr_flipped_preact", flipped_preact)

    _save_png(os.path.join(out_dir, "exr_raw_tonemap.png"), _tonemap_env_for_vis(raw, exposure=args.exposure))
    _save_png(os.path.join(out_dir, "exr_flipped_tonemap.png"), _tonemap_env_for_vis(flipped, exposure=args.exposure))

    # --- B) Load into EnvLight (same as training), then resample to latlong via direction1/2 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This debug script expects CUDA (nvdiffrast).")

    with torch.no_grad():
        env = EnvLight(
            path=None,
            device=device,
            scale=float(args.scale),
            max_res=int(args.max_res),
            trainable=False,
            env_HDR=True,
        ).cuda()
        # mimic training: flip at read time
        setattr(env, "flip_latlong", True)
        setattr(env, "latlong_mode", str(args.latlong_mode))
        env.load(exr_path)
        env.build_mips()
        env.trainable = False
        env.base.requires_grad_(False)

        H = int(args.H)
        W = H * 2
        dirs1 = get_env_direction1(H, W, device=device)
        dirs2 = get_env_direction2(H, W, device=device)

        base_preact = env.base.detach().float().cpu().numpy()
        base_linear = torch.nn.functional.softplus(env.base).detach().float().cpu().numpy()

        _print_stats("cubemap_base_preact", base_preact)
        _print_stats("cubemap_base_linear", base_linear)
        _print_quality("cubemap_base_preact", base_preact)
        _print_quality("cubemap_base_linear", base_linear)

        for i, mip in enumerate(getattr(env, "specular", [])):
            mip_preact = mip.detach().float().cpu().numpy()
            mip_linear = torch.nn.functional.softplus(mip).detach().float().cpu().numpy()
            _print_stats(f"specular_mip_{i}_preact", mip_preact)
            _print_stats(f"specular_mip_{i}_linear", mip_linear)

        diffuse_linear = env(dirs2, mode="diffuse").detach().float().cpu().numpy()

        env1 = env(dirs1, mode="pure_env").detach().float().cpu().numpy()
        env2 = env(dirs2, mode="pure_env").detach().float().cpu().numpy()

        sampled_spec = {}
        for roughness in (0.05, 0.08, 0.1, 0.2, 0.5, 0.9, 1.0):
            roughness_map = torch.full((H, W, 1), float(roughness), device=device, dtype=torch.float32)
            sampled = env(dirs2, roughness=roughness_map).detach().float().cpu().numpy()
            sampled_spec[roughness] = sampled

    _print_stats("resample_env1(direction1)", env1)
    _print_stats("resample_env2(direction2)", env2)
    _print_stats("resample_diffuse(direction2)", diffuse_linear)
    _print_quality("resample_env1(direction1)", env1)
    _print_quality("resample_env2(direction2)", env2)
    _print_quality("resample_diffuse(direction2)", diffuse_linear)

    for roughness, sampled in sampled_spec.items():
        _print_stats(f"resample_spec(direction2)_r={roughness:.2f}", sampled)
        _print_quality(f"resample_spec(direction2)_r={roughness:.2f}", sampled)

    _print_diff_stats("absdiff_resample_env2_vs_flipped_raw", env2, flipped)

    _save_png(os.path.join(out_dir, "resample_env1_direction1_tonemap.png"), _tonemap_env_for_vis(env1, exposure=args.exposure))
    _save_png(os.path.join(out_dir, "resample_env2_direction2_tonemap.png"), _tonemap_env_for_vis(env2, exposure=args.exposure))
    _save_png(os.path.join(out_dir, "resample_diffuse_direction2_tonemap.png"), _tonemap_env_for_vis(diffuse_linear, exposure=args.exposure))

    for roughness, sampled in sampled_spec.items():
        _save_png(
            os.path.join(out_dir, f"resample_spec_direction2_r{roughness:.2f}_tonemap.png"),
            _tonemap_env_for_vis(sampled, exposure=args.exposure),
        )

    print(f"\n[OK] wrote outputs to: {out_dir}")
    print(f"- EXR direct: exr_raw_tonemap.png, exr_flipped_tonemap.png")
    print(f"- Resampled:  resample_env1_direction1_tonemap.png, resample_env2_direction2_tonemap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
