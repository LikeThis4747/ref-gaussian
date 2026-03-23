import os
import torch
import imageio
import numpy as np
from . import renderutils as ru
from .light_utils import *
import nvdiffrast.torch as dr


def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.clip(linear,1e-4,255) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def inverse_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse of softplus.

    softplus(x) = log(1 + exp(x)) maps R -> (0, +inf)
    inverse_softplus(y) = log(exp(y) - 1)
    For large y, inverse_softplus(y) ~ y.
    """
    y = torch.clamp(y, min=eps)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))

class EnvLight(torch.nn.Module):

    def __init__(
        self,
        path=None,
        device=None,
        scale=1.0,
        min_res=16,
        max_res=128,
        min_roughness=0.08,
        max_roughness=0.5,
        trainable=False,
        env_HDR: bool = False,
    ):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable
        self.env_HDR = bool(env_HDR)

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )

        # Keep behavior roughly comparable across LDR/HDR defaults.
        # LDR path returns sigmoid(0)*10 = 5.0 by default.
        if path is None and self.env_HDR:
            with torch.no_grad():
                self.base.data.fill_(float(inverse_softplus(torch.tensor(5.0, device=self.device)).item()))
        
        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()


    def load(self, path, flip_latlong: bool = False):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        path_str = str(path)
        hdr_image = None

        # Prefer OpenEXR for reliable level0 read (full resolution).
        if path_str.lower().endswith(".exr"):
            try:
                import OpenEXR  # type: ignore
                import Imath  # type: ignore

                f = OpenEXR.InputFile(path_str)
                header = f.header()
                dw = header["dataWindow"]
                width = int(dw.max.x - dw.min.x + 1)
                height = int(dw.max.y - dw.min.y + 1)

                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                ch_names = set(header.get("channels", {}).keys())
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
                hdr_image = np.stack([r, g, b], axis=-1)
            except Exception as e:
                hdr_image = None
                print(f"[WARN] OpenEXR read failed for {path_str}, fallback to imageio: {e}")

        if hdr_image is None:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            hdr_image = imageio.imread(path_str)

        if not isinstance(hdr_image, np.ndarray):
            raise ValueError(f"Failed to read envmap: {path_str}")

        if hdr_image.ndim == 2:
            hdr_image = np.repeat(hdr_image[..., None], 3, axis=-1)

        if hdr_image.ndim != 3 or hdr_image.shape[-1] < 3:
            raise ValueError(f"Unexpected envmap shape: {hdr_image.shape} from {path_str}")

        # Drop alpha if present
        if hdr_image.shape[-1] > 3:
            hdr_image = hdr_image[..., :3]

        if hdr_image.dtype != np.float32:
            if np.issubdtype(hdr_image.dtype, np.floating):
                hdr_image = hdr_image.astype(np.float32, copy=False)
            else:
                hdr_image = hdr_image.astype(np.float32) / 255.0

        if flip_latlong:
            hdr_image = np.flip(hdr_image, axis=1).copy()

        if self.env_HDR:
            image = torch.from_numpy(hdr_image).to(self.device) * self.scale
            image = torch.clamp(image, min=1e-6)
            image = inverse_softplus(image)
        else:
            ldr_image = linear_to_srgb(hdr_image)
            image = torch.from_numpy(ldr_image).to(self.device) * self.scale
            image = torch.clamp(image, 0.001, 1 - 0.001)
            image = inverse_sigmoid(image)

        # Convert from latlong to cubemap format.
        # Default to direction2 mapping (z-up) to match the requested convention.
        # To switch back, set: env.latlong_mode = "direction1" before calling load().
        latlong_mode = str(getattr(self, "latlong_mode", "direction2")).lower()
        if latlong_mode in ("direction2", "dir2", "d2"):
            cubemap = latlong_to_cubemap2(image, [self.max_res, self.max_res], self.device)
        else:
            cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        # Assign the cubemap to the base parameter
        self.base.data = cubemap 

    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        light = light.view(*prefix, -1)
        
        if self.env_HDR:
            return torch.nn.functional.softplus(light)
        return torch.sigmoid(light) * 10.0
