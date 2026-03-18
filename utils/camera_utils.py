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
import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, mode):

    # 原图尺寸
    orig_w, orig_h = cam_info.image.size
    orig_w_ori, orig_h_ori = cam_info.image_ori.size
    orig_w_depth, orig_h_depth = cam_info.depth.size
#     if cam_info.image is None:
#         orig_w = cam_info.width
#         orig_h = cam_info.height
#         gt_image = None
#         loaded_mask = None
#     else:
#         orig_w, orig_h = cam_info.image.size
#         resized_image_rgb = PILtoTorch(cam_info.image, (orig_w, orig_h))
#         gt_image = resized_image_rgb[:3, ...]
    
#     if cam_info.depth is None:
#         depth_tensor = None
#     else:
#         resized_depth = PILtoTorch(cam_info.depth, (orig_w, orig_h))
#         depth_tensor = resized_depth[:1, ...]
    
#     if cam_info.image_ori is None:
#         gt_image_ori = None
#     else:
#         orig_w_ori, orig_h_ori = cam_info.image_ori.size
#         resized_image_ori_rgb = PILtoTorch(cam_info.image_ori, (orig_w_ori, orig_h_ori))
#         gt_image_ori = resized_image_ori_rgb[:3, ...]


    # 测试分辨率直接使用训练分辨率，保证一致性
    # resolution_ori = resolution
    resolution = (orig_w, orig_h) 
    resolution_ori = (orig_w_ori, orig_h_ori)  # 强制保持原图尺寸
    resolution_depth = (orig_w_depth, orig_h_depth)

    # PIL 转 Torch
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)   
    resized_image_ori_rgb = PILtoTorch(cam_info.image_ori, resolution_ori)
    resized_depth = PILtoTorch(cam_info.depth,resolution_depth)


    gt_image = resized_image_rgb[:3, ...]
    gt_image_ori = resized_image_ori_rgb[:3, ...]
    depth_tensor = resized_depth[:1, ...]

    loaded_mask = None
    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # return Camera(
    #     colmap_id=cam_info.uid,
    #     R=cam_info.R,
    #     T=cam_info.T,
    #     FoVx=cam_info.FovX,
    #     FoVy=cam_info.FovY,
    #     image=gt_image,
    #     image_ori=gt_image_ori,
    #     gt_alpha_mask=loaded_mask,
    #     image_name=cam_info.image_name,
    #     uid=id,
    #     data_device=args.data_device
    # )
    camera = Camera(
        colmap_id=cam_info.uid,  #训练
        # colmap_id=getattr(cam_info, "uid", id),  # 如果 cam_info 有 uid 用它，否则用 id
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,          # LR
        image_ori=gt_image_ori,  # HR
        depth=depth_tensor,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,  #训练
        # image_name=getattr(cam_info, "image_name", f"cam_{id}"),   #渲染
        uid=id,
        data_device=args.data_device
    )
    # print("camera_utils_HR size:", image_ori.size)

#     # =============================
#     # 添加 SR 和 depth
#     # =============================

#     # HR 作为 SR 监督
#     camera.SR_image = gt_image_ori

#     # Depth
#     if hasattr(cam_info, "depth"):
#         depth_tensor = PILtoTorch(cam_info.depth, resolution)[:1, ...]
#         camera.gt_depth = depth_tensor
#     else:
#         camera.gt_depth = None

    return camera

def cameraList_from_camInfos(cam_infos, resolution_scale, args, mode):
    camera_list = []

    # for id, c in enumerate(cam_infos):
    #     camera_list.append(loadCam(args, id, c, resolution_scale, mode))
    for c in cam_infos:
        camera_list.append(loadCam(args, c.uid, c, resolution_scale, mode))
        print("Loading camera:", c.image_name)

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
