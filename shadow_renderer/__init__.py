import torch
from torch import nn
import torch.nn.functional as F
import math
from scene.gaussian_model import GaussianModel
import nvdiffrast.torch as dr
from utils.refl_utils import sample_camera_rays, reflection, sample_camera_rays_unnormalize

class ShadowRenderer(nn.Module):
    def __init__(self):
        super(ShadowRenderer, self).__init__()
        # self.env_map = env_map  # Environment map for lighting
        # self.HWK = HWK          # Height, Width, Focal Length
        # self.R = R              # Rotation matrix of the camera
        # self.T = T              # Translation vector of the camera
        # self.surf_depth = surf_depth  # Surface depth map
        # self.normal_map = normal_map  # Surface normal map
        # self.pc = pc            # Gaussian model
        # self.mesh = mesh        # Mesh data
    
    @staticmethod
    def get_single_camera_shadow(pc : GaussianModel,env_map, HWK, R, T, surf_depth, normal_map, render_alpha, n_samples=128, randomize_phi: bool = False):
        H,W,K = HWK
        rays_cam, rays_o = sample_camera_rays_unnormalize(HWK, R, T) # rays_cam: [H,W,3] 相机到像素平面光线, rays_o: [3,] 相机位置
        mask = (render_alpha>0)[..., 0] # [H,W,1] 有效渲染区域掩码
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam  # [H,W,3]

        # 扩展到多样本维度: (n_samples, H, W, 3)
        intersections_ns = intersections.unsqueeze(0).expand(n_samples, -1, -1, -1)

        # 基于表面法线在半球上采样 n_samples 条方向（仅占位，未实现）
        # 约定: 返回形状为 (n_samples, H, W, 3)，且为单位向量（世界坐标系）。
        # rays_samples = sample_hemisphere_directions(normal_map, n_samples, min_elevation_deg=5.0, randomize_phi=randomize_phi)
        rays_samples = sample_hemisphere_directions_random(normal_map, n_samples, min_elevation_deg=5.0)

        # 将掩码扩展至 (n_samples, H, W)
        mask_ns = mask.unsqueeze(0).expand(n_samples, -1, -1)

        # 选择有效像素的射线起点与方向，形状均为 (N, 3)
        rays_o_masked  = intersections_ns[mask_ns]
        rays_d_masked  = rays_samples[mask_ns]

        # 追踪求交（RayTracer.trace 支持任意前缀，会在内部展平），此处已是 (N,3)
        positions, face_normals, depth = pc.ray_tracer.trace(rays_o_masked, rays_d_masked)

        # 获取visibility，将可见性写回到 (n_samples, H, W, 1) 形状
        visibility = torch.zeros((n_samples, H, W, 1), device=depth.device, dtype=torch.float32)
        vis_masked = (depth >= 10).to(visibility.dtype).unsqueeze(-1)  # (N,1)
        visibility[mask_ns] = vis_masked

        # 从采样方向查询环境光强度
        # - RGB env: (S,H,W,3)
        # - shadow env mask: (S,H,W,1)
        sampled_I = env_map(rays_samples, mode='pure_env')

        # 计算余弦权重 cos_theta = max(0, n · l)，并作为权重乘到分子与分母（Lambert 权重）
        normals_n = F.normalize(normal_map, dim=-1)                 # (H,W,3)
        cos_theta = (rays_samples * normals_n.unsqueeze(0)).sum(-1) # (S,H,W)
        cos_theta = cos_theta.clamp(min=0.0).unsqueeze(-1)          # (S,H,W,1)

        # 使用环境亮度/权重对可见性进行加权平均，得到标量阴影掩码 (H,W,1)
        if sampled_I.shape[-1] == 1:
            numerator   = (visibility * sampled_I * cos_theta).sum(dim=0)            # (H,W,1)
            denominator = (sampled_I * cos_theta).sum(dim=0).clamp(min=1e-6)         # (H,W,1)
            shadow_mask = numerator / denominator                                      # (H,W,1)
        else:
            numerator   = (visibility * sampled_I * cos_theta).sum(dim=0)            # (H,W,3)
            denominator = (sampled_I * cos_theta).sum(dim=0).clamp(min=1e-6)         # (H,W,3)
            shadow_values = numerator / denominator                                     # (H,W,3)
            shadow_mask = shadow_values.mean(dim=-1, keepdim=True)                     # (H,W,1)

        # 计算unocclusion，一个像素的所有采样128条光线中，可见光线条数/所有光线条数
        unocclusion = visibility.mean(dim=0)  # (H,W,1)

        # 对无效像素置零，保持形状
        unocclusion = unocclusion * mask.unsqueeze(-1)

        # 对无效像素置零，保持形状
        shadow_mask = shadow_mask * mask.unsqueeze(-1)

        return shadow_mask, unocclusion


def sample_hemisphere_directions(normal_map: torch.Tensor, n_samples: int, min_elevation_deg=5.0, randomize_phi: bool = False) -> torch.Tensor:
    """根据每个像素法线在半球上生成 n_samples 个方向。

    假设 normal_map 形状固定为 (H, W, 3)。
    返回形状 (n_samples, H, W, 3) 的归一化世界空间方向。
    """
    device = normal_map.device
    dtype = normal_map.dtype

    H, W, _ = normal_map.shape
    # normals = F.normalize(normal_map.view(-1, 3), dim=-1)  # (Npix,3)
    # NOTE: normal_map may be non-contiguous (e.g. from permute), so use reshape() instead of view().
    normals = F.normalize(normal_map.reshape(-1, 3), dim=-1)  # (Npix,3)
    Npix = normals.shape[0]

    # 规则网格采样 (phi, theta)
    n_phi = max(1, int(math.sqrt(n_samples * 2)))
    n_theta = max(1, n_samples // max(1, n_phi))
    if n_phi * n_theta == 0:
        n_phi, n_theta = 1, n_samples

    epsilon = 1e-6
    phi_vals = torch.linspace(0, 2 * math.pi - epsilon, n_phi, device=device, dtype=dtype)

    # my add: randomize sampling pattern by applying a global azimuth rotation.
    # This keeps the distribution but breaks the fixed set of envmap texels hit each iteration.
    if randomize_phi:
        phi_offset = (2 * math.pi) * torch.rand((), device=device, dtype=dtype)
        phi_vals = torch.remainder(phi_vals + phi_offset, 2 * math.pi)
    # 设置最小抬升角度
    # min_elevation_deg = 5.0
    min_elevation_rad = min_elevation_deg * math.pi / 180.0
    max_theta = (math.pi / 2) - min_elevation_rad
    theta_vals = torch.linspace(0, max_theta - epsilon, n_theta, device=device, dtype=dtype)

    phi_grid, theta_grid = torch.meshgrid(phi_vals, theta_vals, indexing='ij')
    phi = phi_grid.reshape(-1)
    theta = theta_grid.reshape(-1)
    S = phi.shape[0]

    if S > n_samples:
        phi = phi[:n_samples]
        theta = theta[:n_samples]
    elif S < n_samples:
        extra = n_samples - S
        phi_extra = 2 * math.pi * torch.rand(extra, device=device, dtype=dtype)
        theta_extra = torch.arccos(
            torch.rand(extra, device=device, dtype=dtype) * (1 - math.cos(max_theta)) + math.cos(max_theta)
        ).clamp(0, max_theta - epsilon)
        phi = torch.cat([phi, phi_extra], dim=0)
        theta = torch.cat([theta, theta_extra], dim=0)

    S_final = phi.shape[0]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    local_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)  # (S_final,3)
    local_dirs = local_dirs.unsqueeze(0).expand(Npix, -1, -1)  # (Npix,S_final,3)

    # 构造正交基 (tangent, bitangent, normal)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(normals)
    tangent = torch.cross(z_axis, normals)
    deg_mask = (tangent.norm(dim=-1) < 1e-6)
    if deg_mask.any():
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(tangent)
        tangent[deg_mask] = torch.cross(fallback[deg_mask], normals[deg_mask])
    tangent = F.normalize(tangent, dim=-1)
    bitangent = F.normalize(torch.cross(normals, tangent), dim=-1)
    T = torch.stack([tangent, bitangent, normals], dim=-1)  # (Npix,3,3)

    global_dirs = torch.matmul(local_dirs, T.transpose(1, 2))  # (Npix,S,3)
    global_dirs = F.normalize(global_dirs, dim=-1)
    global_dirs = global_dirs.reshape(H, W, S_final, 3).permute(2, 0, 1, 3).contiguous()  # (S,H,W,3)

    return global_dirs


def sample_hemisphere_directions_random(
    normal_map: torch.Tensor,
    n_samples: int,
    min_elevation_deg: float = 5.0,
) -> torch.Tensor:
    """为每个像素法线在其正向半球上生成 n_samples 个“覆盖性更强”的随机方向。

    设计目标：
    - 每次调用都使用随机的 (phi, theta)
    - 但相较纯 i.i.d. 随机，尽量覆盖整个半球（使用分层随机 / jittered stratified sampling）

    坐标/接口约定与 sample_hemisphere_directions 保持一致：
    - 输入 normal_map: (H, W, 3)，世界空间法线（不要求 contiguous）
    - 输出: (n_samples, H, W, 3)，世界空间单位向量方向

    采样分布：在以 +Z 为中心的局部半球帽上，按“均匀立体角”采样（cos(theta) 线性）。
    其中 theta ∈ [0, pi/2 - min_elevation]。
    """
    device = normal_map.device
    dtype = normal_map.dtype

    H, W, _ = normal_map.shape
    normals = F.normalize(normal_map.reshape(-1, 3), dim=-1)  # (Npix,3)
    Npix = normals.shape[0]

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    epsilon = 1e-6
    min_elevation_rad = float(min_elevation_deg) * math.pi / 180.0
    max_theta = (math.pi / 2) - min_elevation_rad
    if max_theta <= 0:
        raise ValueError(
            f"min_elevation_deg={min_elevation_deg} is too large; it must be < 90 degrees."
        )

    # Choose a 2D stratification grid close to n_samples.
    # Using near-square bins tends to yield better coverage.
    n_phi = int(math.ceil(math.sqrt(n_samples)))
    n_theta = int(math.ceil(n_samples / max(1, n_phi)))
    n_phi = max(1, n_phi)
    n_theta = max(1, n_theta)

    # Stratified jitter in [0,1) for (u,v) -> (phi, cos(theta))
    # u controls azimuth; v controls polar angle (via uniform solid-angle).
    u = (torch.arange(n_phi, device=device, dtype=dtype).unsqueeze(1) + torch.rand((n_phi, n_theta), device=device, dtype=dtype)) / float(n_phi)
    v = (torch.arange(n_theta, device=device, dtype=dtype).unsqueeze(0) + torch.rand((n_phi, n_theta), device=device, dtype=dtype)) / float(n_theta)

    phi = (2.0 * math.pi) * u.reshape(-1)

    # Uniform solid-angle over a spherical cap: cos(theta) ∈ [cos(max_theta), 1]
    cos_max = float(math.cos(max_theta))
    cos_theta = 1.0 - v.reshape(-1) * (1.0 - cos_max)
    cos_theta = cos_theta.clamp(min=cos_max, max=1.0)
    theta = torch.arccos(cos_theta)

    # Keep exactly n_samples directions.
    S = phi.shape[0]
    if S > n_samples:
        # Random subset to avoid always taking the same prefix.
        perm = torch.randperm(S, device=device)
        perm = perm[:n_samples]
        phi = phi[perm]
        theta = theta[perm]
    elif S < n_samples:
        extra = n_samples - S
        phi_extra = (2.0 * math.pi) * torch.rand(extra, device=device, dtype=dtype)
        cos_theta_extra = 1.0 - torch.rand(extra, device=device, dtype=dtype) * (1.0 - cos_max)
        theta_extra = torch.arccos(cos_theta_extra.clamp(min=cos_max, max=1.0 - epsilon))
        phi = torch.cat([phi, phi_extra], dim=0)
        theta = torch.cat([theta, theta_extra], dim=0)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    local_dirs = torch.stack(
        [
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            cos_theta,
        ],
        dim=-1,
    )  # (S,3)

    # Expand to per-pixel: (Npix,S,3)
    local_dirs = local_dirs.unsqueeze(0).expand(Npix, -1, -1)

    # Construct orthonormal basis per pixel: (tangent, bitangent, normal)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(normals)
    tangent = torch.cross(z_axis, normals)
    deg_mask = (tangent.norm(dim=-1) < 1e-6)
    if deg_mask.any():
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(tangent)
        tangent[deg_mask] = torch.cross(fallback[deg_mask], normals[deg_mask])
    tangent = F.normalize(tangent, dim=-1)
    bitangent = F.normalize(torch.cross(normals, tangent), dim=-1)
    T = torch.stack([tangent, bitangent, normals], dim=-1)  # (Npix,3,3)

    global_dirs = torch.matmul(local_dirs, T.transpose(1, 2))  # (Npix,S,3)
    global_dirs = F.normalize(global_dirs, dim=-1)
    global_dirs = global_dirs.reshape(H, W, n_samples, 3).permute(2, 0, 1, 3).contiguous()  # (S,H,W,3)
    return global_dirs


def sample_sphere_directions(normal_map: torch.Tensor, n_samples: int, min_elevation_deg=5.0) -> torch.Tensor:
    """根据每个像素法线在整球面上生成 n_samples 个方向。

    采样策略与 sample_hemisphere_directions 保持一致（规则网格 + 不足时随机补足）。
    区别在于：同时包含法线正向半球与反向半球（上下两个半球）。

    为避免贴近切平面的方向，沿用最小抬升角 min_elevation_deg：
    - 上半球：polar angle theta ∈ [0, pi/2 - min_elevation]
    - 下半球：等价于将上半球方向的 z 分量取负（相对法线反向），从而同样避开地平线附近。

    Args:
        normal_map: (H, W, 3) 世界空间法线
        n_samples: 采样方向数量
        min_elevation_deg: 最小抬升角（度），对上下半球都生效

    Returns:
        (n_samples, H, W, 3) 归一化世界空间方向
    """
    device = normal_map.device
    dtype = normal_map.dtype

    H, W, _ = normal_map.shape
    normals = F.normalize(normal_map.reshape(-1, 3), dim=-1)  # (Npix,3)
    Npix = normals.shape[0]

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    epsilon = 1e-6
    min_elevation_rad = float(min_elevation_deg) * math.pi / 180.0
    max_theta = (math.pi / 2) - min_elevation_rad
    if max_theta <= 0:
        raise ValueError(
            f"min_elevation_deg={min_elevation_deg} is too large; it must be < 90 degrees."
        )

    def _sample_local_hemisphere(sample_count: int) -> torch.Tensor:
        """Sample local hemisphere directions around +Z with min elevation."""
        sample_count = int(sample_count)
        if sample_count <= 0:
            return torch.empty((0, 3), device=device, dtype=dtype)

        n_phi = max(1, int(math.sqrt(sample_count * 2)))
        n_theta = max(1, sample_count // max(1, n_phi))
        if n_phi * n_theta == 0:
            n_phi, n_theta = 1, sample_count

        phi_vals = torch.linspace(0, 2 * math.pi - epsilon, n_phi, device=device, dtype=dtype)
        theta_vals = torch.linspace(0, max_theta - epsilon, n_theta, device=device, dtype=dtype)

        phi_grid, theta_grid = torch.meshgrid(phi_vals, theta_vals, indexing='ij')
        phi = phi_grid.reshape(-1)
        theta = theta_grid.reshape(-1)
        S = phi.shape[0]

        if S > sample_count:
            phi = phi[:sample_count]
            theta = theta[:sample_count]
        elif S < sample_count:
            extra = sample_count - S
            phi_extra = 2 * math.pi * torch.rand(extra, device=device, dtype=dtype)
            theta_extra = torch.arccos(
                torch.rand(extra, device=device, dtype=dtype) * (1 - math.cos(max_theta)) + math.cos(max_theta)
            ).clamp(0, max_theta - epsilon)
            phi = torch.cat([phi, phi_extra], dim=0)
            theta = torch.cat([theta, theta_extra], dim=0)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        return torch.stack([
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            cos_theta,
        ], dim=-1)  # (S,3)

    # Split samples across the two hemispheres.
    n_up = (n_samples + 1) // 2
    n_down = n_samples - n_up

    local_up = _sample_local_hemisphere(n_up)  # z > 0
    if n_down > 0:
        local_down = _sample_local_hemisphere(n_down)
        local_down = local_down.clone()
        local_down[:, 2] = -local_down[:, 2]
        local_dirs = torch.cat([local_up, local_down], dim=0)  # (n_samples,3)
    else:
        local_dirs = local_up

    # Expand to per-pixel: (Npix,S,3)
    local_dirs = local_dirs.unsqueeze(0).expand(Npix, -1, -1)

    # Construct orthonormal basis per pixel: (tangent, bitangent, normal)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(normals)
    tangent = torch.cross(z_axis, normals)
    deg_mask = (tangent.norm(dim=-1) < 1e-6)
    if deg_mask.any():
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(tangent)
        tangent[deg_mask] = torch.cross(fallback[deg_mask], normals[deg_mask])
    tangent = F.normalize(tangent, dim=-1)
    bitangent = F.normalize(torch.cross(normals, tangent), dim=-1)
    T = torch.stack([tangent, bitangent, normals], dim=-1)  # (Npix,3,3)

    global_dirs = torch.matmul(local_dirs, T.transpose(1, 2))  # (Npix,S,3)
    global_dirs = F.normalize(global_dirs, dim=-1)
    global_dirs = global_dirs.reshape(H, W, n_samples, 3).permute(2, 0, 1, 3).contiguous()
    return global_dirs
