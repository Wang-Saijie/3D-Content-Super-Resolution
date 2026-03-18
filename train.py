# 加入伪视点
import os
import torch
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import numpy as np
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from models.network_swinir import SwinIR as net
from tqdm import tqdm
from utils.image_utils import psnr
from utils.depth_utils import estimate_depth
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def rgb_shuffle(img: torch.Tensor, p=0.5):
            if torch.rand(1) < p:
                idx = torch.randperm(3)
                img = img[idx, :, :]
            return img
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    os.makedirs(args.pc_path, exist_ok=True)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # scene = CustomScene(dataset, gaussians)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # SR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
    #             img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    #             mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    # param_key_g = 'params'
    # pretrained_model = torch.load("./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
    # SR_model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
    #                       strict=True)
    # SR_model.eval()
    # SR_model = SR_model.to(device)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    best_psnr = -1.0
    best_iteration = -1
    # target_indices = [6, 16, 22, 25, 39, 50, 51, 52, 66, 71]
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_stack = selected_cams.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if pseudo_stack is None or len(pseudo_stack) == 0:
            pseudo_stack = scene.getTrainCameras().copy()
            
        # # Pick a fixed Camera from target_indices
        # train_cams = scene.getTrainCameras()
        # if viewpoint_stack is None or len(viewpoint_stack) == 0:
        #     viewpoint_stack = [train_cams[idx] for idx in target_indices if idx < len(train_cams)]
        # viewpoint_cam = viewpoint_stack.pop(0)  # 顺序取第一个
        
        SR_H, SR_W = 912, 1368
        # LR_H, LR_W = 342, 228
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image1, viewspace_point_tensor, visibility_filter, radii, hr_depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        # RGBShuffle: 随机打乱通道
        # image1 = rgb_shuffle(image1)

        # print("train_image1 size:", image1.size())
        # print("train_hr_depth size:", hr_depth.size())
#         image1 = F.interpolate(
#             image_o.unsqueeze(0),   # [C,H,W] -> [1,C,H,W]
#             size=(SR_H, SR_W),
#             mode='bilinear',
#             align_corners=False
#         ).squeeze(0)  # [C,H,W]
#         # print("train_image1 size:", image1.size())
        
#         hr_depth = F.interpolate(
#             depth_o.unsqueeze(0),   # [C,H,W] -> [1,C,H,W]
#             size=(SR_H, SR_W),
#             mode='bilinear',
#             align_corners=False
#         ).squeeze(0)  # [C,H,W]
        # print("train_hr_depth size:", hr_depth.size())
        # Loss
        eps = 1e-6
        gt_image = viewpoint_cam.image.cuda()
        # gt_image = rgb_shuffle(gt_image)
        gt_depth = viewpoint_cam.depth.cuda()
        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + eps)
        
        # print("gt_depth size:", gt_depth.size())
        # print("gt_image size:", gt_image.size())
        # with torch.no_grad():
        #     # pad input image to be a multiple of window_size
        #     window_size = 8
        #     gt_image = gt_image.unsqueeze(dim=0)
        #     _, _, h_old, w_old = gt_image.size()
        #     h_pad = (h_old // window_size + 1) * window_size - h_old
        #     w_pad = (w_old // window_size + 1) * window_size - w_old
        #     gt_image = torch.cat([gt_image, torch.flip(gt_image, [2])], 2)[:, :, :h_old + h_pad, :]
        #     gt_image = torch.cat([gt_image, torch.flip(gt_image, [3])], 3)[:, :, :, :w_old + w_pad]
        #     SR_image = SR_model(gt_image)torch.cuda.empty_cache()
        #     SR_image = SR_image[..., :h_old * 4, :w_old * 4]
        #     print(SR_image.shape)

        SR_image = viewpoint_cam.image_ori.cuda()
        SR_image = SR_image.squeeze(dim=0)
        # SR_image = rgb_shuffle(SR_image)

        # print("train_HR size:", SR_image.size())
        Ll1_sr = l1_loss(image1, SR_image)   #1368×912
        loss_sr = (1.0 - opt.lambda_dssim) * Ll1_sr + opt.lambda_dssim * (1.0 - ssim(image1, SR_image))
        image1 = F.avg_pool2d(image1, kernel_size=(4, 4))
        # print("train_image1 size:", image1.size())
        Ll1_lr = l1_loss(image1, gt_image)
        loss_lr = (1.0 - opt.lambda_dssim) * Ll1_lr + opt.lambda_dssim * (1.0 - ssim(image1, gt_image))
        # loss = (1-alpha) * loss_sr + alpha * loss_lr
        loss_L = 0.6 * loss_sr + 0.4 * loss_lr
        # print("loss_L:", loss_L.item()) # 0.25
        
        lr_depth_pred = F.avg_pool2d(hr_depth, kernel_size=(4, 4))
        lr_depth_pred = (lr_depth_pred - lr_depth_pred.min()) / (lr_depth_pred.max() - lr_depth_pred.min() + eps)
        lr_depth_pred = 1.0 - lr_depth_pred
        loss_depth = F.l1_loss(lr_depth_pred, gt_depth)
        # print("loss_depth:", loss_depth.item()) # 0.18
        
        loss = loss_L + 0.2 * loss_depth
        
        # ==========================
        # pseudo view supervision
        # ==========================
        pseudo_loss = 0

        # 频繁计算，每次迭代或每 2 次迭代
        # if iteration > 3000 and iteration % 1 == 0:  
        if iteration % 1 == 0:  

            # 如果 pseudo_stack 空了，重新生成
            if pseudo_stack is None or len(pseudo_stack) == 0:
                pseudo_stack = scene.getTrainCameras().copy()

            # 每次采样 N 个伪视点
            num_pseudo = min(3, len(pseudo_stack))  # 可修改为更多
            sampled_indices = np.random.choice(len(pseudo_stack), num_pseudo, replace=False)
    
            for idx in sampled_indices:
                pseudo_cam = pseudo_stack[idx]

                render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                pseudo_rgb = render_pkg_pseudo["render"]
                pseudo_depth = render_pkg_pseudo["depth"][0]
                visibility_filter = render_pkg_pseudo["visibility_filter"]  # 可见性 mask

                with torch.no_grad():
                    midas_depth = estimate_depth(pseudo_rgb)

                # Depth尺度对齐：MinMax
                eps = 1e-6
                
                pseudo_depth = pseudo_depth.squeeze()
                midas_depth = midas_depth.squeeze()
                mask = (pseudo_depth > 0)
                pseudo_depth_vis = pseudo_depth[mask]
                midas_depth_vis = midas_depth[mask]
                # pseudo_depth_vis = pseudo_depth.reshape(-1)
                # midas_depth_vis = midas_depth.reshape(-1)
                if pseudo_depth_vis.numel() > 10:
                    pseudo_depth_vis = (pseudo_depth_vis - pseudo_depth_vis.mean()) / (pseudo_depth_vis.std() + 1e-6)
                    midas_depth_vis = (midas_depth_vis - midas_depth_vis.mean()) / (midas_depth_vis.std() + 1e-6)
                # pseudo_depth = (pseudo_depth - pseudo_depth.mean()) / (pseudo_depth.std() + 1e-6)
                # midas_depth = (midas_depth - midas_depth.mean()) / (midas_depth.std() + 1e-6)

                # Pearson correlation
                    corr = pearson_corrcoef(pseudo_depth_vis.reshape(-1,1), midas_depth_vis.reshape(-1,1))
                    depth_loss_pseudo = (1 - corr).mean()
                else:
                    depth_loss_pseudo = 0.0

                # RGB一致性 (可见点)
#                 pseudo_rgb_vis = pseudo_rgb[:, visibility_filter.view(-1)].reshape(3, -1)  # [C, N]
#                 gt_rgb_vis = pseudo_cam.image[:, visibility_filter.view(-1)].reshape(3, -1)

#                 rgb_l1 = F.l1_loss(pseudo_rgb_vis, gt_rgb_vis)
#                 rgb_ssim = 1.0 - ssim(pseudo_rgb_vis.reshape(1,3,1,-1), 
#                                       gt_rgb_vis.reshape(1,3,1,-1))  # 将 N 点当作 1D 图像计算 SSIM
#                 rgb_loss_pseudo = 0.5 * rgb_l1 + 0.5 * rgb_ssim
                # RGB一致性
                pseudo_rgb_lr = F.avg_pool2d(pseudo_rgb, kernel_size=4)
                pseudo_rgb_vis = pseudo_rgb_lr.unsqueeze(0)  # [1,3,H,W]
                gt_rgb_vis = pseudo_cam.image.unsqueeze(0)

                rgb_l1 = F.l1_loss(pseudo_rgb_vis, gt_rgb_vis)
                rgb_ssim = 1.0 - ssim(pseudo_rgb_vis, gt_rgb_vis, size_average=True)
                rgb_loss_pseudo = 0.5 * rgb_l1 + 0.5 * rgb_ssim

                # 综合伪视点损失
                pseudo_loss += 0.2 * depth_loss_pseudo + 0.3 * rgb_loss_pseudo  # 增大权重
                
        pseudo_loss /= num_pseudo
        loss += 0.05 * pseudo_loss
        
        loss.backward()

        iter_end.record()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration == first_iter:
                best_psnr_holder = {"value": -1.0}

            training_report(tb_writer, iteration, Ll1_sr, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background),
                            best_psnr_holder,
                            # target_indices,
                            scene.model_path)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                if gaussians.max_radii2D.numel() == 0:
                    gaussians.max_radii2D = torch.zeros(gaussians.get_xyz.shape[0], device=radii.device)
                # Keep track of max radii in image-space for pruning
                # 确保 max_radii2D 在和 radii 同一 device 上
                gaussians.max_radii2D = gaussians.max_radii2D.to(radii.device)
                visibility_filter = visibility_filter.to(radii.device)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    if args.log_dir is not None:
        tb_writer = SummaryWriter(args.log_dir)
    else:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene,
                    renderFunc, renderArgs,
                    best_psnr_holder,
                    # target_indices,
                    save_root):

    global_best_psnr = best_psnr_holder["value"]

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        cameras = scene.getTestCameras()
        if cameras is None or len(cameras) == 0:
            cameras = scene.getTrainCameras()

        psnr_total = 0.0

        for viewpoint in cameras:
            renders = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            image = torch.clamp(renders["render"], 0.0, 1.0)
            image_test = torch.clamp(viewpoint.image_test.to("cuda"), 0.0, 1.0)
            psnr_total += psnr(image, image_test).mean().double()

        psnr_avg = psnr_total / len(cameras)

        print(f"\n[ITER {iteration}] PSNR: {psnr_avg:.4f}")

        # 如果当前PSNR更好
        if psnr_avg > global_best_psnr:
            print("New best PSNR, saving ply and selected renders...")
            best_psnr_holder["value"] = psnr_avg

            # 1️⃣ 保存最佳ply
            save_ply_path = os.path.join(save_root, "best_model.ply")
            scene.gaussians.save_ply(save_ply_path)
            
            save_ply_path = os.path.join(args.pc_path, f"point_cloud_{iteration}.ply")
            scene.gaussians.save_ply(save_ply_path)

            # 2️⃣ 保存指定索引视角
            save_img_dir = os.path.join(save_root, "best_renders")
            os.makedirs(save_img_dir, exist_ok=True)

            test_cams = scene.getTestCameras()
            # test_cams = [c for c in scene.getTrainCameras() if c.uid in test_ids]
            
            for viewpoint in test_cams:
                renders = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                image = torch.clamp(renders["render"], 0.0, 1.0)
                depth = renders["depth"]

                # 保存 image
                image_np = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
                image_name = viewpoint.image_name + ".JPG"
                image_path = os.path.join(save_img_dir, image_name)

                from imageio import imwrite
                imwrite(image_path, image_np)

                # 保存 depth
                depth_np = depth.squeeze().cpu().numpy()
                depth_np = depth_np.astype(np.float32)
                depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
                depth_np = 1.0 - depth_np
#                 depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
#                 valid_mask = depth_np > 0
#                 if valid_mask.sum() > 0:
#                     depth_valid = depth_np[valid_mask]
                    
#                     # 百分位裁剪（关键）
#                     p2 = np.percentile(depth_valid, 2)
#                     p98 = np.percentile(depth_valid, 98)
#                     depth_np = np.clip(depth_np, p2, p98)
                    
#                     # 重新归一化
#                     depth_np = (depth_np - p2) / (p98 - p2 + 1e-6)

#                     # 反转（近处白）
#                     depth_np = 1.0 - depth_np
#                 else:
#                     depth_np = np.zeros_like(depth_np)
                depth_name = viewpoint.image_name + "_depth.png"
                depth_path = os.path.join(save_img_dir, depth_name)
                # depth_uint16 = (depth_np * 65535).astype(np.uint16)
                depth_uint16 = (depth_np * 65535).astype(np.uint16)
                imwrite(depth_path, depth_uint16)

#             for idx in train_cams:
#                 # print("idx:", idx)
#                 if idx >= len(train_cams):
#                     continue

#                 viewpoint = train_cams[idx]
#                 # print("viewpoint:", viewpoint)
#                 render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

#                 image = torch.clamp(render_pkg["render"], 0.0, 1.0)
#                 depth = render_pkg["depth"]

#                 # 保存 image
#                 image_np = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
#                 image_name = viewpoint.image_name + ".JPG"
#                 image_path = os.path.join(save_img_dir, image_name)

#                 from imageio import imwrite
#                 imwrite(image_path, image_np)

#                 # 保存 depth
#                 depth_np = depth.squeeze().cpu().numpy()
#                 depth_np = depth_np.astype(np.float32)
#                 depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
#                 depth_name = viewpoint.image_name + "_depth.png"
#                 depth_path = os.path.join(save_img_dir, depth_name)
#                 depth_uint16 = (depth_np * 65535).astype(np.uint16)
#                 # imwrite(depth_path, depth_np)
#                 imwrite(depth_path, depth_uint16)

            print("✅ Best ply and selected views saved.")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 5000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pc_path", type=str, default="./point_clouds", help="Path to save Gaussians point_cloud.ply")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")