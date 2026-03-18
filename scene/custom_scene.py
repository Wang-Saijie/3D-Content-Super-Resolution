import os
import torch
from scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera

class CustomScene:

    def __init__(self, args, gaussians: GaussianModel):

        self.model_path = args.model_path
        self.gaussians = gaussians

        # ---------- 1️⃣ 直接加载已有 PLY ----------
        print("Loading existing LR Gaussian ply...")
        self.gaussians.load_ply(args.ply_path)

        # ---------- 2️⃣ 构造相机 ----------
        self.train_cameras = self.load_cameras(args)

    def load_cameras(self, args):

        cameras = []

        pose_list = torch.load(args.pose_path)  
        # 假设 pose_path 里存的是：
        # 每个元素包含：
        # {
        #   "R": 3x3,
        #   "T": 3,
        #   "fx": float,
        #   "fy": float,
        #   "cx": float,
        #   "cy": float,
        #   "image_name": "0001.png"
        # }

        for pose in pose_list:

            cam = Camera(
                colmap_id=0,
                R=pose["R"],
                T=pose["T"],
                FoVx=None,
                FoVy=None,
                image=None,
                gt_alpha_mask=None,
                image_name=pose["image_name"],
                uid=0,
                data_device="cuda"
            )

            # ---------- 3️⃣ 加载 LR RGB ----------
            lr_path = os.path.join(args.lr_path, pose["image_name"])
            cam.original_image = self.load_image(lr_path)

            # ---------- 4️⃣ 加载 HR pseudo GT ----------
            hr_path = os.path.join(args.hr_path, pose["image_name"])
            cam.SR_image = self.load_image(hr_path)

            # ---------- 5️⃣ 加载 LR depth ----------
            depth_path = os.path.join(args.depth_path, pose["image_name"])
            cam.gt_depth = self.load_depth(depth_path)

            cameras.append(cam)

        return cameras

    def load_image(self, path):
        import imageio
        img = imageio.imread(path)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)
        return img.cuda()

    def load_depth(self, path):
        import imageio
        depth = imageio.imread(path)
        depth = torch.from_numpy(depth).float()
        return depth.cuda()

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras

    def getTestCameras(self, scale=1.0):
        return []