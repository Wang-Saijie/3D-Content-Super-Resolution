# 不改相机内参，渲染所有视角
import os
import torch
import torch.nn.functional as F
import numpy as np
import skimage.io
import argparse
import copy

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import readColmapCamerasNoImage
from scene.cameras import loadCam
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from models.network_swinir import SwinIR


# ================= SwinIR 初始化 =================
def load_swinir(device):

    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        depths=[6]*6,
        embed_dim=180,
        num_heads=[6]*6,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    pretrained = torch.load(
        "./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        map_location=device
    )

    model.load_state_dict(pretrained["params"])
    model.eval()

    return model


# ================= SwinIR 推理 =================
def run_swinir(sr_model, lr):

    lr = lr.unsqueeze(0)  # [1,3,H,W]
    _, _, h, w = lr.size()

    window_size = 8
    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w

    lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h + h_pad, :]
    lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w + w_pad]

    with torch.no_grad():
        sr = sr_model(lr)

    sr = sr[..., :h*4, :w*4]
    return sr.squeeze(0)


# ================= 主函数 =================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_dir = os.path.join(args.output, "LR")
    hr_dir = os.path.join(args.output, "HR_pseudoGT")
    depth_dir = os.path.join(args.output, "depth")

    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print("Loading COLMAP cameras...")
    extrinsics = read_extrinsics_binary(os.path.join(args.source, "sparse/0/images.bin"))
    intrinsics = read_intrinsics_binary(os.path.join(args.source, "sparse/0/cameras.bin"))
    cam_infos = readColmapCamerasNoImage(extrinsics, intrinsics)

    print("Total views:", len(cam_infos))

    print("Loading trained 3DGS model...")
    gaussians = GaussianModel(3)
    gaussians.load_ply(args.ply)

    print("Loading SwinIR...")
    sr_model = load_swinir(device)

    background = torch.tensor([0., 0., 0.], dtype=torch.float32, device=device)

    # 目标 LR 尺寸
    W_lr = 342
    H_lr = 228

    # 指定相机
    selected_ids = [6, 16, 22, 25, 39, 50, 51, 52, 66, 71]

    for idx in selected_ids:

        cam_info = copy.deepcopy(cam_infos[idx])
        
        scale = 0.95   # ⭐推荐 0.9 ~ 0.98
        cam_info.FovX *= scale
        cam_info.FovY *= scale
        
        print(f"[{idx}] Rendering {cam_info.image_name}")
#         # 渲染所有相机
#     for idx, cam_info in enumerate(cam_infos):

#         cam_info = copy.deepcopy(cam_info)
#         print(f"[{idx}] Rendering {cam_info.image_name}")

        camera = loadCam(cam_info, device=device)

        class DummyPipe:
            def __init__(self):
                self.debug = False
                self.compute_cov3D_python = False
                self.convert_SHs_python = False
                self.antialiasing = False

        dummy_pipe = DummyPipe()

        # ================= HR 渲染 =================
        with torch.no_grad():
            renders = render(camera, gaussians, dummy_pipe, background)

        hr = renders["render"].clamp(0, 1)  # [3, H_orig, W_orig]
        print("hr size:", hr.size())

        # ================= 生成 LR =================
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(H_lr, W_lr),
            # scale_factor=0.25,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        print("lr size:", lr.size())

        # 保存 LR
        lr_np = lr.permute(1, 2, 0).cpu().numpy() * 255.0
        skimage.io.imsave(
            os.path.join(lr_dir, cam_info.image_name),
            lr_np.astype(np.uint8)
        )

        # ================= SR =================
        torch.cuda.empty_cache()
        sr = run_swinir(sr_model, lr.to(device))
        print("sr size:", sr.size())
        sr_np = sr.permute(1, 2, 0).cpu().numpy() * 255.0

        skimage.io.imsave(
            os.path.join(hr_dir, cam_info.image_name),
            sr_np.astype(np.uint8)
        )

        # ================= Depth =================
        depth_hr = renders["depth"]  # [1, H_orig, W_orig]

        depth_lr = F.interpolate(
            depth_hr.unsqueeze(0),
            size=(H_lr, W_lr),
            # scale_factor=0.25,
            mode='bicubic',
            # mode='bilinear',
            align_corners=False
        ).squeeze(0)[0]  # [H_lr, W_lr]

        rendered_depth = depth_lr.cpu().numpy()

        # min-max 归一化 + 反转
        valid_mask = rendered_depth > 0

        if valid_mask.sum() > 0:
            depth_min = rendered_depth[valid_mask].min()
            depth_max = rendered_depth[valid_mask].max()

            rendered_depth_norm = (
                (rendered_depth - depth_min) /
                (depth_max - depth_min + 1e-8)
            )

            rendered_depth_norm = 1.0 - rendered_depth_norm
        else:
            rendered_depth_norm = rendered_depth

        depth_uint16 = (rendered_depth_norm * 65535).astype(np.uint16)

        depth_name = cam_info.image_name.replace(".JPG", "_depth.png") \
                                  .replace(".jpg", "_depth.png")

        skimage.io.imsave(
            os.path.join(depth_dir, depth_name),
            depth_uint16
        )

        print(f"[{idx}] Done.")

    print("Finished generating pseudo GT dataset.")


# ================= 入口 =================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, required=True,
                        help="COLMAP scene root folder")

    parser.add_argument("--ply", type=str, required=True,
                        help="Trained 3DGS point_cloud.ply path")

    parser.add_argument("--output", type=str, default="pseudo_dataset",
                        help="Output dataset folder")

    args = parser.parse_args()

    main(args)