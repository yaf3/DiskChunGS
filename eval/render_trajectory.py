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
import re
import subprocess
from pathlib import Path
import tempfile
import shutil

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import plot

import matplotlib.pyplot as plt
import copy

def natural_sort_key(path):
    """
    Sort key function for filenames with numeric patterns.
    """
    # Extract the numeric part from filenames like '000112.png'
    match = re.search(r'(\d+)\.png$', str(path))
    if match:
        return int(match.group(1))
    return 0

def convert_images_to_video(input_dir, output_path, fps=30):
    """
    Convert a directory of images to a video file using ffmpeg.
    
    Args:
        input_dir (str): Directory containing the images
        output_path (str): Path for the output video file
        fps (int): Frames per second for the output video
    """
    # Convert to Path objects for better path handling
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all png files and sort them using the natural sort key
    files = sorted([f for f in input_dir.glob("*.png")], key=natural_sort_key)
    if not files:
        raise ValueError(f"No PNG files found in {input_dir}")
    
    print(f"First few files in order: {[f.name for f in files[:5]]}")
    print(f"Total number of files: {len(files)}")
    
    # Create a temporary directory for preparation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Check the first image for dimensions and ensure they're even
        first_image = Image.open(files[0])
        width, height = first_image.size
        first_image.close()
        
        # Create a padded version of images if needed
        need_padding = width % 2 != 0 or height % 2 != 0
        
        # Create input file list for ffmpeg
        input_list_file = temp_dir_path / "input_list.txt"
        
        with open(input_list_file, 'w') as f:
            for i, img_path in enumerate(files):
                print(f"Processing {img_path.name}")
                
                # If we need padding, create a copy with even dimensions
                if need_padding:
                    try:
                        img = Image.open(img_path)
                        new_width = width + (1 if width % 2 != 0 else 0)
                        new_height = height + (1 if height % 2 != 0 else 0)
                        
                        # Create a new image with even dimensions
                        new_img = Image.new("RGB", (new_width, new_height))
                        new_img.paste(img, (0, 0))
                        
                        # Save to temp directory
                        temp_img_path = temp_dir_path / f"frame_{i:06d}.png"
                        new_img.save(temp_img_path)
                        f.write(f"file '{temp_img_path.absolute()}'\n")
                        f.write(f"duration {1/fps}\n") 
                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {e}")
                else:
                    # Use original image
                    f.write(f"file '{img_path.absolute()}'\n")
                    f.write(f"duration {1/fps}\n")
            
            # Add the last frame again without duration to avoid cutting off
            if files:
                if need_padding:
                    f.write(f"file '{(temp_dir_path / f'frame_{len(files)-1:06d}.png').absolute()}'\n")
                else:
                    f.write(f"file '{files[-1].absolute()}'\n")
        
        # Build the ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                   # Overwrite output file
            '-f', 'concat',         # Use concat format
            '-safe', '0',           # Allow absolute file paths
            '-i', str(input_list_file),  # Input file list
            '-vsync', 'vfr',        # Variable frame rate (helps with timestamp issues)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            str(output_path)        # Output file
        ]
        
        # Execute ffmpeg command
        try:
            print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg error: {error_message}")

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

    
    width = 1920
    height = 1080
    config_path = os.path.join(model_data_path, "gaussian_mapper_cfg.yaml")
    print("Using:", model_data_path, config_path)
    
    success = gs_render.initialize(config_path, model_data_path)
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
    
    os.makedirs(os.path.join(args.result_path, "render"), exist_ok=True)
    for index in trange(
        len(associations),
        desc="rendering {}".format(args.result_path.split("/")[-1]),
    ):
        (result_indx, gt_indx) = associations[index]
        w2c = torch.tensor(np.linalg.inv(poses[result_indx]))
        render_image = gs_render.render_from_pose(w2c, width, height).clone().detach().to('cuda')
        
        render_image = render_image.permute(1, 2, 0)
        render_image = torch.clamp(render_image, 0.0, 1.0)           
        predict_np = render_image.detach().cpu().numpy()
        # If in [0,1] range, convert to [0,255]
        if predict_np.max() <= 1.0:
            predict_np = (predict_np * 255).astype(np.uint8)
        # Create PIL image and save
        predict_pil = Image.fromarray(predict_np)
        predict_pil.save(
            os.path.join(
                args.result_path,
                "render",
                gt_color_paths[gt_indx].split("/")[-1]
            )
        )
        
    convert_images_to_video(os.path.join(
                args.result_path,
                "render"), os.path.join(
                args.result_path, "rendered_video.mp4"))