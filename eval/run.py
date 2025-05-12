import os
import sys
import time
import numpy as np
import torch
from tqdm import trange
import json
import glob
import gs_render
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2
# import atexit

# Register the cleanup function to be called at exit
# atexit.register(gs_render.cleanup)


from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import plot

import matplotlib.pyplot as plt
import copy

calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()
calc_lpips = LearnedPerceptualImagePatchSimilarity().cuda()


def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    tstamp = [
        float(
            color_path.split("/")[-1]
            .replace("frame", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadTUM(path):
    if os.path.exists(os.path.join(path, "rgb3")):
        color_paths = sorted(glob.glob(os.path.join(path, "rgb3/*.png")))
    else:
        color_paths = sorted(glob.glob(os.path.join(path, "rgb/*.png")))
    tstamp = [
        float(
            color_path.split("/")[-1]
            .replace("frame", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    tstamp = np.loadtxt(
        os.path.join(path, "times.txt"), delimiter=" ", dtype=np.str_
    ).astype(np.float32)
    return color_paths, tstamp


def loadEuRoC(path):
    color_paths = sorted(glob.glob(os.path.join(path, "mav0/cam0/data/*.png")))
    tstamp = [np.float64(x.split("/")[-1][:-4]) / 1e9 for x in color_paths]
    return color_paths, tstamp


def associate_frames(tstamp_image, tstamp_pose, max_dt=0.1, slowdown_factor=1.0):
    """Pair images, depths, and poses, accounting for slowdown factor."""
    associations = []
    
    # If there's a slowdown factor, adjust the pose timestamps
    if slowdown_factor != 1.0:
        # Adjust estimated timestamps back to the original time scale
        # (start_time + (current_time - start_time) / slowdown)
        tstamp_pose_adjusted = tstamp_pose[0] + (tstamp_pose - tstamp_pose[0]) / slowdown_factor
    else:
        tstamp_pose_adjusted = tstamp_pose
    
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose_adjusted - t))
        if np.abs(tstamp_pose_adjusted[j] - t) < max_dt:
            associations.append((i, j))
            
    print(f"Associated {len(associations)} frames out of {len(tstamp_image)} ground truth frames")
    return associations

def load_slowdown_factor(result_path, default=1.0):
    """Load slowdown factor from result path if available, otherwise return default value."""
    slowdown_path = os.path.join(result_path, "slowdown_factor.txt")
    if os.path.exists(slowdown_path):
        try:
            with open(slowdown_path, "r") as f:
                slowdown_factor = float(f.readline().strip())
            print(f"Loaded slowdown factor: {slowdown_factor}x from {slowdown_path}")
            return slowdown_factor
        except (ValueError, IOError) as e:
            print(f"Error loading slowdown factor: {e}. Using default: {default}x")
            return default
    else:
        print(f"No slowdown factor file found at {slowdown_path}. Using default: {default}x")
        return default


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default=None)
    parser.add_argument("gt_path", type=str, default=None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--skip_trajectory_eval", action="store_true", 
                        help="Skip trajectory evaluation and use ground truth poses for rendering")
    args = parser.parse_args()
    dirs = os.listdir(args.result_path)
    # load model
    ts = []
    Rs = []
    width, height = 0, 0
    render_time = 0
    shutdown_name = None    
    for file_name in dirs:
        if ("shutdown" in file_name):
            shutdown_name = file_name
            break
    if shutdown_name is None:
        sys.exit("No shutdown dir found, exiting...")
    model_data_path = os.path.join(
        args.result_path,
        shutdown_name,
        "data",
            )
    print(model_data_path)
    with open(
        os.path.join(
            args.result_path, shutdown_name, "data", "cameras.json"
        ),
        "r",
    ) as fin:
        camera_paras = json.load(fin)
        print(os.path.join(
            args.result_path, shutdown_name, "data", "cameras.json"
        ))

    width, height = (
        camera_paras[0]["width"],
        camera_paras[0]["height"],
    )
    config_path = os.path.join(model_data_path, "gaussian_mapper_cfg.yaml")
    print("Using:", model_data_path, config_path)
    
    success = gs_render.initialize(config_path, model_data_path)
    render_time = np.loadtxt(
        os.path.join(args.result_path, shutdown_name, "render_time.txt"),
        delimiter=" ",
        dtype=np.str_,
    )
    render_time = render_time[:, 1].astype(np.float32)
    
    num_gaussians = np.loadtxt(
        os.path.join(args.result_path, shutdown_name, "gaussianCount.txt"),
        delimiter=" ",
        dtype=np.str_,
    )
    num_gaussians = int(num_gaussians.item())

    # load gt
    if "replica" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadReplica(args.gt_path)
    elif "kitti" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadKITTI(args.gt_path)
    elif "euroc" in args.gt_path.lower():
        print(args.gt_path)
        gt_color_paths, gt_tstamp = loadEuRoC(args.gt_path)
    else:
        gt_color_paths, gt_tstamp = loadTUM(args.gt_path)

    # Load ground truth poses for either trajectory evaluation or direct rendering
    if "kitti" in args.gt_path.lower():
        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, "poses/{}.txt".format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    # print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    # print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)
            return pose_quat

        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(
            positions_xyz=pose_quat[:, :3],
            orientations_quat_wxyz=pose_quat[:, 3:],
            timestamps=np.array(gt_tstamp),
        )
    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, "pose_TUM.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(
            args.gt_path, "mav0/state_groundtruth_estimate0/data.csv"
        )
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
        T_i_c0 = np.array(
            [
                [
                    0.0148655429818,
                    -0.999880929698,
                    0.00414029679422,
                    -0.0216401454975,
                ],
                [
                    0.999557249008,
                    0.0149672133247,
                    0.025715529948,
                    -0.064676986768,
                ],
                [
                    -0.0257744366974,
                    0.00375618835797,
                    0.999660727178,
                    0.00981073058949,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        traj_ref.transform(T_i_c0, True)
    else:
        gt_file = os.path.join(args.gt_path, "groundtruth.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        
    # Load slowdown factor (if exists)
    slowdown_factor = load_slowdown_factor(args.result_path)

    # If not skipping trajectory eval, load estimated poses and evaluate them
    if not args.skip_trajectory_eval:
        pose_path = os.path.join(args.result_path, "CameraTrajectory_TUM.txt")
        traj_est = file_interface.read_tum_trajectory_file(pose_path)
        
        # If there's a slowdown factor, adjust timestamps before association
        if slowdown_factor != 1.0:
            # Make a deep copy to avoid modifying the original
            traj_est_adjusted = copy.deepcopy(traj_est)
            # Adjust timestamps: start_time + (current_time - start_time) / slowdown
            start_time = traj_est_adjusted.timestamps[0]
            traj_est_adjusted.timestamps = start_time + (traj_est_adjusted.timestamps - start_time) / slowdown_factor
            # Use the adjusted trajectory for association
            traj_ref_sync, traj_est = sync.associate_trajectories(
                traj_ref, traj_est_adjusted, max_diff=0.1
            )
        else:
            # Use original trajectory
            traj_ref_sync, traj_est = sync.associate_trajectories(
                traj_ref, traj_est, max_diff=0.1
            )
        
        traj_ref_sync.align(traj_est, True)
        poses = traj_est.poses_se3
        tstamp = traj_est.timestamps
        
        result = main_ape.ape(
            traj_ref_sync,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=args.correct_scale,
        )
        result_rotation_part = main_ape.ape(
            traj_ref_sync,
            traj_est,
            est_name="rot",
            pose_relation=PoseRelation.rotation_part,
            align=True,
            correct_scale=args.correct_scale,
        )

        out_path = os.path.join(args.result_path, "metrics_traj.txt")
        with open(out_path, "w") as fp:
            fp.write(result.pretty_str())
            fp.write(result_rotation_part.pretty_str())
        print(result)

        if args.show_plot:
            traj_est_aligned = copy.deepcopy(traj_est)
            traj_est_aligned.align(traj_ref_sync, correct_scale=True)
            fig = plt.figure()
            traj_by_label = {
                "estimate (not aligned)": traj_est,
                "estimate (aligned)": traj_est_aligned,
                "reference": traj_ref_sync,
            }
            plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            plt.show()
    else:
        # When skipping trajectory eval, use ground truth poses directly
        print("Skipping trajectory evaluation, using ground truth poses for rendering")
        poses = traj_ref.poses_se3
        tstamp = traj_ref.timestamps
        
        # Write a note to metrics file
        out_path = os.path.join(args.result_path, "metrics_traj.txt")
        with open(out_path, "w") as fp:
            fp.write("Trajectory evaluation skipped - using ground truth poses for rendering\n")

    ## render and evaluation
    associations = associate_frames(tstamp, gt_tstamp)
    
    os.makedirs(os.path.join(args.result_path, "image"), exist_ok=True)
    if "_0" in args.result_path:
        os.makedirs(os.path.join(args.result_path, "gt"), exist_ok=True)
    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []
    for index in trange(
        len(associations),
        desc="rendering {}".format(args.result_path.split("/")[-1]),
    ):
        t_start = time.time()
        (result_indx, gt_indx) = associations[index]
        w2c = torch.tensor(np.linalg.inv(poses[result_indx]))
        t0 = time.time()
        render_image = gs_render.render_from_pose(w2c, width, height).clone().detach().to('cuda')
        t_render = time.time() - t0  
        
        t0 = time.time()
        render_image = render_image.permute(1, 2, 0)
        render_image = torch.clamp(render_image, 0.0, 1.0)
        render_image_torch = render_image.permute([2, 0, 1])[None]
    
        pil_image = Image.open(gt_color_paths[gt_indx])
        gt_image = np.array(pil_image).astype(np.float32) / 255.0
        gt_image_torch = torch.from_numpy(gt_image).float().permute(2, 0, 1).unsqueeze(0).to('cuda')
        t_process = time.time() - t0
        
        t0 = time.time()
        val_psnr = calc_psnr(render_image_torch, gt_image_torch).item()
        val_ssim = calc_ssim(render_image_torch, gt_image_torch).item()
        val_lpips = calc_lpips(render_image_torch, gt_image_torch).item()
        t_metrics = time.time() - t0

        t0 = time.time()
        if "_0" in args.result_path:
            # Convert floating point (0.0-1.0) to uint8 (0-255)
            gt_image_uint8 = np.uint8(gt_image * 255)
            gt_image_bgr = cv2.cvtColor(gt_image_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(
                    args.result_path,
                    "gt",
                    gt_color_paths[gt_indx].split("/")[-1],
                ),
                gt_image_bgr,  # Use uint8 version for saving
            )
        predict_image_np = render_image.detach().cpu().numpy()
        predict_image_img = np.uint8(predict_image_np * 255)
        predict_image_img = cv2.cvtColor(predict_image_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            os.path.join(
                args.result_path,
                "image",
                gt_color_paths[gt_indx].split("/")[-1],
            ),
            predict_image_img,
        )
        t_save = time.time() - t0

        psnr_list.append(val_psnr)
        ssim_list.append(val_ssim)
        lpips_list.append(val_lpips)
        time_list.append(t_render)
        
        t_total = time.time() - t_start
        # print(f"Render: {t_render*1000:.1f}ms, Process: {t_process*1000:.1f}ms, " 
        #     f"Metrics: {t_metrics*1000:.1f}ms, Save: {t_save*1000:.1f}ms, " 
        #     f"Total: {t_total*1000:.1f}ms")
        
    # print("Calling cleanup to properly release resources...")
    # try: 
    #     gs_render.cleanup()
    # except Exception as e:
    #     print(f"Error during cleanup: {e}")
    # print("Cleanup finished.")

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    time_list = np.array(time_list)
    np.savetxt(os.path.join(args.result_path, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(args.result_path, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(args.result_path, "lpips.txt"), lpips_list)

    # Handle tracking time evaluation if file exists
    tracking_time_path = os.path.join(args.result_path, "TrackingTime.txt")
    if os.path.exists(tracking_time_path):
        with open(tracking_time_path, "r") as fin:
            tracking_time = fin.readlines()
        if len(tracking_time) > 3:  # Check if there's enough data to process
            tracking_time = np.array(tracking_time[:-3]).astype(np.float32)
            tracking_fps = 1 / np.mean(tracking_time) if np.mean(tracking_time) > 0 else 0
        else:
            tracking_time = np.array([0])
            tracking_fps = 0
    else:
        tracking_time = np.array([0])
        tracking_fps = 0

    with open(os.path.join(args.result_path, "eval.txt"), "w") as fout:
        fout.write("psnr: {}\n".format(np.mean(psnr_list)))
        fout.write("ssim: {}\n".format(np.mean(ssim_list)))
        fout.write("lpips: {}\n".format(np.mean(lpips_list)))
        
        if not args.skip_trajectory_eval:
            fout.write("tracking s: {}\n".format(np.mean(tracking_time)))
            fout.write("tracking FPS: {}\n".format(tracking_fps))
        else:
            fout.write("tracking evaluation skipped (using ground truth poses)\n")

        fout.write("rendering ms: {}\n".format(np.mean(render_time)))
        fout.write("rendering FPS: {}\n".format(1000 / np.mean(render_time)))
        fout.write("num gaussians: {}\n".format(num_gaussians))