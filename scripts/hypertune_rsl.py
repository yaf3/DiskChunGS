import optuna
import subprocess
import yaml
import numpy as np
import os
from pathlib import Path
from optuna.samplers import TPESampler
import shutil
import time
from pynput.keyboard import Key, Controller

def play_rosbag(file_path):
    # Import threading here since we'll need it
    import threading
    
    # Function to continuously read from pipes to prevent buffer filling
    def read_pipe(pipe, prefix):
        for line in iter(pipe.readline, b''):
            print(f"{prefix}: {line.decode('utf-8', errors='ignore').strip()}")
    
    # Start the rosbag play command
    print("Starting rosbag play...")
    process = subprocess.Popen(
        ["rosbag", "play", "--clock", "--pause", file_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start threads to read from stdout and stderr
    stdout_thread = threading.Thread(target=read_pipe, args=(process.stdout, "STDOUT"))
    stderr_thread = threading.Thread(target=read_pipe, args=(process.stderr, "STDERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Give the process time to start
    time.sleep(2)
    
    # Send space character to begin playback
    print("Sending space to begin playback...")
    process.stdin.write(b' ')
    process.stdin.flush()
    
    # Wait for process to complete with a timeout
    try:
        print("Waiting for rosbag to complete...")
        return_code = process.wait(timeout=200)  # Bag is 160s, adding margin
        print(f"Rosbag playback complete with return code: {return_code}")
        return True
    except subprocess.TimeoutExpired:
        print("Rosbag playback timed out")
        process.terminate()
        process.wait(timeout=5)
        print("Rosbag process terminated")
        return False

def objective(trial):
    # Create the YAML content as a string
    yaml_content = "%YAML:1.0\n\n"
    
    hyperparams = {
        # Model architecture parameters
        'Model.sh_degree': int(trial.suggest_int('sh_degree', 1, 3)),
        # 'Model.resolution': trial.suggest_float('resolution', 0.6, 2.0),
        
        # Camera/Geometry parameters
        # 'Camera.z_near': trial.suggest_float('z_near', 0.01, 1.0),
        # 'Camera.z_far': trial.suggest_float('z_far', 50.0, 200.0),
        # 'Monocular.inactive_geo_densify_max_pixel_dist': trial.suggest_float('inactive_geo_densify_max_pixel_dist', 0.5, 2.0),
        'Mapper.depth_densify_subsample_ratio': trial.suggest_categorical('depth_densify_subsample_ratio', [0, 0.0001, 0.001, .01, 0.05, 0.1, 0.5]),
        
        # Mapper parameters
        # 'Mapper.min_num_initial_map_kfs': int(trial.suggest_int('min_num_initial_map_kfs', 10, 30)),
        # 'Mapper.new_keyframe_times_of_use': int(trial.suggest_int('new_keyframe_times_of_use', 5, 50)),
        # 'Mapper.large_rotation_threshold': trial.suggest_float('large_rotation_threshold', 6.0, 30.0),
        # 'Mapper.large_translation_threshold': trial.suggest_float('large_translation_threshold', 0.05, 0.3),
        
        'External.min_keyframe_translation': trial.suggest_float('min_keyframe_translation', 0.1, 0.5),
        'External.min_keyframe_rotation': trial.suggest_float('min_keyframe_rotation', 0.1, 0.5),
        'External.min_keyframe_time': trial.suggest_float('min_keyframe_time', 0.1, 0.5),
        
        # GausPyramid parameters
        # 'GausPyramid.do': trial.suggest_categorical('do', [0, 1]),

        # 'GausPyramid.num_sub_levels': int(trial.suggest_int('num_sub_levels', 1, 4)),
        # 'GausPyramid.sub_level_times_of_use': int(trial.suggest_int('sub_level_times_of_use', 2, 16)),
        
        # Optimization parameters - Learning rates
        # 'Optimization.position_lr_init': trial.suggest_float('position_lr_init', 1e-7, 1e-2, log=True),
        # 'Optimization.position_lr_final': trial.suggest_float('position_lr_final', 1e-7, 1e-4, log=True),
        # 'Optimization.position_lr_delay_mult': trial.suggest_float('position_lr_delay_mult', 0.001, 0.05),
        'Optimization.position_lr_max_steps': int(trial.suggest_int('position_lr_max_steps', 1000, 30000, log=True)),
        # 'Optimization.feature_lr': trial.suggest_float('feature_lr', 0.0001, 0.1, log=True),
        # 'Optimization.opacity_lr': trial.suggest_float('opacity_lr', 0.0001, 0.1, log=True),
        # 'Optimization.scaling_lr': trial.suggest_float('scaling_lr', 0.0001, 0.1, log=True),
        # 'Optimization.rotation_lr': trial.suggest_float('rotation_lr', 0.0001, 0.1, log=True),
        
        # Optimization parameters - Densification
        'Optimization.percent_dense': trial.suggest_float('percent_dense', 0.01, 0.1, log=True),
        'Optimization.lambda_dssim': trial.suggest_float('lambda_dssim', 0.1, 0.4),
        # 'Optimization.densification_interval': int(trial.suggest_int('densification_interval', 300, 500)),
        'Optimization.densify_min_opacity': trial.suggest_float('densify_min_opacity', 0.01, 0.1, log=True),
        # 'Optimization.densify_from_iter': int(trial.suggest_int('densify_from_iter', 100, 800)),
        'Optimization.densify_grad_threshold': trial.suggest_float('densify_grad_threshold', 1e-4, 1e-2, log=True),
        'Optimization.opacity_reg': trial.suggest_float('opacity_reg', 1e-5, 1e-1, log=True),
        # 'Optimization.opacity_reset_interval': trial.suggest_categorical('opacity_reset_interval', [0, 100, 300, 500]),
        # 'Optimization.prune_big_point_after_iter': int(trial.suggest_categorical('prune_big_point_after_iter', [-1, 500, 2000])),
        # 'Optimization.max_num_iterations': int(trial.suggest_int('max_num_iterations', 5000, 40000)),
        # 'Chunking.chunk_size': trial.suggest_categorical('chunk_size', [20, 50, 100]),
        
         'Optimization.appearance_embedding': trial.suggest_categorical('appearance_embedding', [0, 1]),
        
    }
    
    # Load base config
    base_path = "/workspace/repo"
    config_path = f"{base_path}/cfg/gaussian_mapper/RGB-D/RSL/arche_train1.yaml"
    trial_config_path = f"{base_path}/cfg/gaussian_mapper/RGB-D/RSL/runs/run0/train1_trial_{trial.number}.yaml"
    gt_path = "/data/RSL/datasets/train1"
    orbslam_cfg_path = f"{base_path}/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml"
    orbslam_vocab_path = f"{base_path}/slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt"
    result_dir = f"{base_path}/results/RSL/runs/run0/train1_trial_{trial.number}"

    with open(config_path, 'r') as f:
        base_content = f.read()
    
    # Parse the base content
    for line in base_content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('%'):
            key = line.split(':')[0].strip()
            if key not in hyperparams:
                yaml_content += line + '\n'
    
    # Add our hyperparameters
    for key, value in hyperparams.items():
        yaml_content += f"{key}: {value}\n"
    
    # Save modified config
    os.makedirs(os.path.dirname(trial_config_path), exist_ok=True)
    with open(trial_config_path, 'w') as f:
        f.write(yaml_content)
        
    # Run the program in a non-blocking way
    try:
        # Start the gaussian_slam process in the background
        gaussian_process = subprocess.Popen([
            "/root/catkin_ws/devel/lib/lsgs_ros/lsgs_ros_node",
            "__name:=gaussian_slam",
            f"_vocabulary_path:={orbslam_vocab_path}",
            f"_orb_settings_path:={orbslam_cfg_path}",
            f"_gaussian_settings_path:={trial_config_path}",
            f"_output_directory:={result_dir}",
            f"_use_viewer:=false",
            f"_mode:=rgbd",
            f"_rgb_topic:=/left_camera_rgb",
            f"_depth_topic:=/zed2/zed_node/depth/depth_registered",
            f"_slam_mode:=external",
            f"_target_frame:=map",
            f"_source_frame:=zed2_left_camera_optical_frame"
        ])
        
        # Wait for gaussian_slam to initialize
        time.sleep(10)
        
        # Play the rosbag file (this function will block until playback completes)
        play_rosbag("/data/RSL/arche/train1.bag")
        
        # Now wait for gaussian_slam to finish
        gaussian_process.wait(timeout=1000)
        
    except subprocess.TimeoutExpired:
        # Make sure to terminate the process if it times out
        gaussian_process.terminate()
        print("Gaussian SLAM process timed out")
        return float('inf')
    
    print("Training done, now running eval")
    # Now evaluate
    os.system(
                f"python3 {base_path}/eval/run.py {} {} --skip_trajectory_eval".format(result_dir, gt_path)
            )
    
    score = 9999999999999
    PSNR, SSIM, LPIPS = (
        None,
        None,
        None,
    )
    if os.path.exists(os.path.join(result_dir, "eval.txt")):
        with open(os.path.join(result_dir, "eval.txt")) as fin:
            PSNR = float(fin.readline().split()[-1])
            SSIM = float(fin.readline().split()[-1])
            LPIPS = float(fin.readline().split()[-1])
    
    # score = (PSNR / 40.0) + (SSIM / 1.0) + (-LPIPS*10) + (0 * Num_Gaussians)
    score = LPIPS
    shutil.rmtree(result_dir)
    return score

if __name__ == "__main__":

    sampler = TPESampler(
        multivariate=True,
        n_startup_trials=100,
    )
    
    study = optuna.create_study(
        storage="sqlite:///rsl_hypertune_02_05.db",
        study_name="rsl_gaussian_optimization",
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=10000)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  LPIPS: {trial.value:.3f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")