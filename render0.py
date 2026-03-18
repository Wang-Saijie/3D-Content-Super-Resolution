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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import loadCam
from gaussian_renderer import GaussianModel
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    for idx, cam_info in enumerate(tqdm(views, desc="Rendering progress")):
        print(type(cam_info))
        view = loadCam(cam_info)
#         import numpy as np

#         if not hasattr(view, "world_view_transform"):
#             # world_view_transform: 这里需要 4x4 矩阵，或者训练简化版只要 3x4
#             Rt = np.eye(4)
#             Rt[:3,:3] = view.R
#             Rt[:3,3] = view.T
#             view.world_view_transform = np.linalg.inv(Rt)[:3,:4]  # 取 3x4

#         if not hasattr(view, "full_proj_transform"):
#             # 根据 FovX/FovY + width/height 构建投影矩阵
#             fovx = view.FovX
#             fovy = view.FovY
#             w, h = view.width, view.height
#             near, far = 0.1, 100.0
#             fx = 1 / np.tan(fovx/2)
#             fy = 1 / np.tan(fovy/2)
#             view.full_proj_transform = np.array([
#                 [fx, 0, 0, 0],
#                 [0, fy, 0, 0],
#                 [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
#                 [0, 0, -1, 0]
#             ], dtype=np.float32)

#         if not hasattr(view, "camera_center"):
#             view.camera_center = -view.R.T @ view.T
        
        rendering = render(view, gaussians, pipeline, background)["render"]
        # gt = view.image_test[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        if not hasattr(dataset, 'white_background'):
            dataset.white_background = True  # 或 False，取决于你想要背景白色还是黑色
        sh_degree = getattr(dataset, "sh_degree", 3)
        gaussians = GaussianModel(sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [0,0,0] if dataset.white_background else [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    if not hasattr(args, 'images'):
        args.images = None   # 没有图片
    if not hasattr(args, 'eval'):
        args.eval = False    
        
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)