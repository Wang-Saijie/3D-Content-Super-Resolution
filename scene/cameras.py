#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#将LR图像渲染为HR图像
from PIL import Image
from models.network_swinir import SwinIR as net
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torchvision.transforms.functional as TF

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,image_ori, gt_alpha_mask,
                 image_name, uid, depth,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        # print("image_ori type:", type(image_ori))
        # print("image_ori shape:", image_ori.shape)
        
        super(Camera, self).__init__()
        # self.data_device = device
        # self.test_data_device = device
        # self.image = TF.to_tensor(image).float()           # LR 图
        # self.image_ori = TF.to_tensor(image_ori).float() if image_ori is not None else self.image  # HR 图
        # self.depth = TF.to_tensor(depth).float() if depth is not None else None
        self.image = image
        self.image_ori = image_ori
        self.depth = depth if depth is not None else None
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.SR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
        #                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        #                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        # param_key_g = 'params'
        # pretrained_model = torch.load("./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
        # self.SR_model.load_state_dict(
        #     pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
        #     strict=True)
        # self.SR_model.eval()
        # self.SR_model = self.SR_model.to(device)

        try:
            self.data_device = torch.device(data_device)
            self.test_data_device = torch.device("cpu")
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_test = image_ori.clamp(0.0, 1.0).to(self.test_data_device)

        # with torch.no_grad():
        # # pad input image to be a multiple of window_size
        #     window_size = 8
        #     self.SR_image = self.original_image.unsqueeze(dim=0)
        #     _, _, h_old, w_old = self.SR_image.size()
        #     h_pad = (h_old // window_size + 1) * window_size - h_old
        #     w_pad = (w_old // window_size + 1) * window_size - w_old
        #     self.SR_image = torch.cat([self.SR_image, torch.flip(self.SR_image, [2])], 2)[:, :, :h_old + h_pad, :]
        #     self.SR_image = torch.cat([self.SR_image, torch.flip(self.SR_image, [3])], 3)[:, :, :, :w_old + w_pad]
        #     self.SR_image = self.SR_model(self.SR_image)
        #     self.SR_image = self.SR_image[..., :h_old * 4, :w_old * 4]
        #     print(self.SR_image.shape)
        self.SR_image = image_ori.clamp(0.0, 1.0).unsqueeze(0).to("cpu")
        # print("cameras_HR size:", image_ori.size)
        
        self.image_width = self.SR_image.shape[3]
        self.image_height = self.SR_image.shape[2]
        # print("SR_image shape:", self.SR_image.shape)

        # if gt_alpha_mask is not None:
        #     self.image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.image *= torch.ones((1, self.image_height // 4, self.image_width // 4), device=self.data_device)
        #     # self.image_test *= torch.ones((1, self.image_height, self.image_width),
        #     #                                   device=self.test_data_device)
        #     # self.image_test *= torch.ones((1, self.image_height, self.image_width), device=self.image_test.device)
            # if hasattr(self, "image_ori"):
            #     _, h, w = self.image_ori.shape
            #     self.image_test = torch.ones((1, h, w), device=self.image_ori.device)
            # else:
            #     # 退化方案，保证不报错
            #     self.image_test = torch.ones((1, self.image_height, self.image_width), device=self.image_test.device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
def loadCam(cam_info, device="cuda"):
    """
    根据 COLMAP cam_info 创建 MiniCam 对象，用于 render()
    """
    # width = 342
    # height = 228
    width = cam_info.width
    height = cam_info.height
    # fovx = cam_info.FovX
    # fovy = cam_info.FovY

    fovx = getattr(cam_info, "FovX", getattr(cam_info, "fov_x", 60.0))
    fovy = getattr(cam_info, "FovY", getattr(cam_info, "fov_y", 60.0))
    znear = 0.01
    zfar = 100.0

    world_view = torch.tensor(getWorld2View2(cam_info.R, cam_info.T), dtype=torch.float32).transpose(0, 1).to(device)
    proj = torch.tensor(getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy), dtype=torch.float32).transpose(0, 1).to(device)
    full_proj = torch.matmul(world_view, proj)

    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj
    )
    