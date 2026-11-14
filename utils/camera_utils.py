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

from scene.cameras import Camera
import numpy as np
import os, cv2, torch
import torch.nn.functional as F
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov

WARNED = False

def _load_material_gt(cam_info, names, original_size, resize_resolution, final_resolution):
    try:
        import pyexr
    except ImportError as exc:
        raise ImportError("Material GT loss requires pyexr to read EXR files") from exc

    dataset_root = os.path.dirname(os.path.dirname(cam_info.image_path))
    split = os.path.basename(os.path.dirname(cam_info.image_path))
    gt_root = os.path.join(dataset_root, "gt", split)
    assert os.path.isdir(gt_root), f"Material GT directory does not exist: {gt_root}"

    stem = cam_info.image_name
    mask_path = os.path.join(gt_root, "mask", stem + ".png")
    assert os.path.isfile(mask_path), f"Material GT mask does not exist: {mask_path}"
    mask_array = np.asarray(Image.open(mask_path).convert("L"))
    expected_shape = (original_size[1], original_size[0])
    assert mask_array.shape == expected_shape, (
        f"Material GT mask shape mismatch for {stem}: "
        f"{mask_array.shape} vs {expected_shape}"
    )
    mask = torch.from_numpy(mask_array.copy()).float()[None, None] / 255.0

    resize_hw = (resize_resolution[1], resize_resolution[0])
    mask = F.interpolate(mask, size=resize_hw, mode="nearest")[0]
    final_w, final_h = final_resolution
    mask = mask[:, :final_h, :final_w] > 0.5

    gt_maps = {}
    scalar_names = {"metallic", "roughness", "depth"}
    for name in names:
        path = os.path.join(gt_root, name, stem + ".exr")
        assert os.path.isfile(path), f"Enabled {name} GT does not exist: {path}"
        array = pyexr.open(path).get()
        if array.ndim == 2:
            array = array[..., None]
        channels = 1 if name in scalar_names else 3
        assert array.shape[:2] == expected_shape and array.shape[2] >= channels, (
            f"{name} GT shape mismatch for {stem}: got {array.shape}, "
            f"expected HxWx{channels} at {expected_shape}"
        )
        value = torch.from_numpy(
            np.ascontiguousarray(array[..., :channels], dtype=np.float32)
        ).permute(2, 0, 1)[None]
        value = F.interpolate(
            value, size=resize_hw, mode="bilinear", align_corners=False
        )[0, :, :final_h, :final_w]
        if name == "normal":
            # Blender camera space (Y up, Z back) -> Ref camera space (Y down, Z forward).
            value[1:3].neg_()
        gt_maps[name] = value

    return gt_maps, mask


def loadCam(args, id, cam_info, resolution_scale, load_material_gt=False):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = float(resolution_scale * args.resolution)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600 and False:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    K = None
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] = K[:2] / scale

    resize_resolution = resolution
    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # [UPSTREAM Ref-Gaussian] af7691a utils/camera_utils.py::loadCam
    # [REF-MV MODIFICATION] Crop only after Ref's original -r resize so DINO's
    # 14x14 patch embedding and the camera projection use one shared size.
    joint_patch_size = getattr(args, "joint_patch_size", 0)
    if joint_patch_size:
        width = (resolution[0] // joint_patch_size) * joint_patch_size
        height = (resolution[1] // joint_patch_size) * joint_patch_size
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Camera {cam_info.image_name} becomes empty when cropped to "
                f"a multiple of patch size {joint_patch_size}: {resolution}"
            )
        gt_image = gt_image[:, :height, :width]
        if loaded_mask is not None:
            loaded_mask = loaded_mask[:, :height, :width]
        resolution = (width, height)

    HWK = None if K is None else (resolution[1], resolution[0], K)
    fov_x, fov_y = cam_info.FovX, cam_info.FovY
    if joint_patch_size and K is not None:
        fov_x = focal2fov(K[0, 0], resolution[0])
        fov_y = focal2fov(K[1, 1], resolution[1])
    # #
    refl_path = os.path.join(
        os.path.dirname(os.path.dirname(cam_info.image_path)), 'image_msk')
    refl_path = os.path.join(refl_path, os.path.basename(cam_info.image_path))
    if not os.path.exists(refl_path):
        refl_path = refl_path.replace('.JPG', '.jpg')
    if os.path.exists(refl_path):
        refl_msk = cv2.imread(refl_path) != 0 # max == 1
        refl_msk = torch.tensor(refl_msk).permute(2,0,1).float()
    else: refl_msk = None

    material_gt = None
    material_gt_mask = None
    gt_names = getattr(args, "material_gt_names", ())
    if load_material_gt and gt_names:
        material_gt, material_gt_mask = _load_material_gt(
            cam_info,
            gt_names,
            (orig_w, orig_h),
            resize_resolution,
            resolution,
        )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=fov_x, FoVy=fov_y,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device, HWK=HWK, gt_refl_mask=refl_msk,
                  material_gt=material_gt, material_gt_mask=material_gt_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, load_material_gt=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, load_material_gt))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
