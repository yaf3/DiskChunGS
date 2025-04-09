import optuna
import subprocess
import yaml
import numpy as np
import os
from pathlib import Path
from optuna.samplers import TPESampler

def objective(trial):
    # Create the YAML content as a string
    yaml_content = "%YAML:1.0\n\n"
    
    hyperparams = {
        # Model architecture parameters
        'Model.sh_degree': int(trial.suggest_int('sh_degree', 1, 3)),
        # 'Model.resolution': trial.suggest_float('resolution', 0.6, 2.0),
        
        # Camera/Geometry parameters
        'Camera.z_near': trial.suggest_float('z_near', 0.01, 1.0),
        'Camera.z_far': trial.suggest_float('z_far', 50.0, 200.0),
        # 'Monocular.inactive_geo_densify_max_pixel_dist': trial.suggest_float('inactive_geo_densify_max_pixel_dist', 0.5, 2.0),
        
        # Mapper parameters
        # 'Mapper.min_num_initial_map_kfs': int(trial.suggest_int('min_num_initial_map_kfs', 10, 30)),
        # 'Mapper.new_keyframe_times_of_use': int(trial.suggest_int('new_keyframe_times_of_use', 5, 50)),
        # 'Mapper.large_rotation_threshold': trial.suggest_float('large_rotation_threshold', 6.0, 30.0),
        # 'Mapper.large_translation_threshold': trial.suggest_float('large_translation_threshold', 0.05, 0.3),
        
        # GausPyramid parameters
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
        # 'Chunking.chunk_size': trial.suggest_categorical('chunk_size', [20, 50, 100])
        
    }
    
    # Load base config
    config_path = "cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml"
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
    trial_config_path = f"cfg/gaussian_mapper/Stereo/KITTI/runs/run1/KITTI_trial_{trial.number}.yaml"
    os.makedirs(os.path.dirname(trial_config_path), exist_ok=True)
    with open(trial_config_path, 'w') as f:
        f.write(yaml_content)
        
    gt_path = "/data/kitti/data_odometry_color/dataset/sequences_modified/00_500f"
    
    # Run the program
    result_dir = f"results/kitti/runs/run1/00_trial_{trial.number}"
    try:
        subprocess.run([
            "./bin/kitti_stereo",
            "third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt",
            "cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml",
            trial_config_path,
            gt_path,
            result_dir,
            "no_viewer"
        ], timeout=1000)
    except subprocess.TimeoutExpired:
        return -float('inf')
    
    print("Training done, now running eval")
    # Now evaluate
    os.system(
                "python3 eval/run.py {} {}".format(result_dir, gt_path)
            )
    
    score = 9999999999999
    PSNR, SSIM, LPIPS, Tracking_fps, Rendering_fps, Num_Gaussians = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if os.path.exists(os.path.join(result_dir, "eval.txt")):
        with open(os.path.join(result_dir, "eval.txt")) as fin:
            PSNR = float(fin.readline().split()[-1])
            SSIM = float(fin.readline().split()[-1])
            LPIPS = float(fin.readline().split()[-1])
            Tracking_time = float(fin.readline().split()[-1])
            Tracking_fps = float(fin.readline().split()[-1])
            Rendering_time = float(fin.readline().split()[-1])
            Rendering_fps = float(fin.readline().split()[-1])
            Num_Gaussians = int(fin.readline().split()[-1])
    
    # score = (PSNR / 40.0) + (SSIM / 1.0) + (-LPIPS*10) + (0 * Num_Gaussians)
    score = LPIPS
    return score

if __name__ == "__main__":

    sampler = TPESampler(
        multivariate=True,
        n_startup_trials=100,
    )
    
    study = optuna.create_study(
        storage="sqlite:///kitti_hypertune_09_04.db",
        study_name="kitti_gaussian_optimization",
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