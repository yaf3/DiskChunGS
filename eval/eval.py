"""This is a scipt to calcuate quantitative results stored.

in the result_main_folder for the evluation dataset.

Outputs are two files including "result_main_folder/log.txt" and
"result_main_folder/log.csv"
"""

import numpy as np
import os
import glob
import csv
import argparse
import chamfer_l2


parser = argparse.ArgumentParser(description="evaluation script")
parser.add_argument("-d", "--dataset_center_path", type=str, required=True)
parser.add_argument("-r", "--result_main_folder", type=str, required=True)
args = parser.parse_args()


dataset_center_path = args.dataset_center_path
result_main_folder = args.result_main_folder

gt_dataset = {
    "replica": {
        "path": os.path.join(dataset_center_path, "Replica"),
        "scenes": [
            "office0",
            "office1",
            "office2",
            "office3",
            "office4",
            "room0",
            "room1",
            "room2",
        ],
    },
    "tum": {
        "path": os.path.join(dataset_center_path, "TUM"),
        "scenes": [
            "rgbd_dataset_freiburg1_desk",
            "rgbd_dataset_freiburg2_xyz",
            "rgbd_dataset_freiburg3_long_office_household",
        ],
    },
    "kitti": {
        "path": os.path.join(dataset_center_path, "kitti/data_odometry_color/dataset/sequences"),
        "scenes": [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
                
        ],
    }
}

# path the all results
results = [
    m
    for m in sorted(os.listdir(result_main_folder))
    if os.path.isdir(os.path.join(result_main_folder, m))
]

for result in results:
    print("processing", result)
    # support datasetName_cameratype_xx
    gt_dataset_name = result.split("_")[0].lower()
    
    if gt_dataset_name not in gt_dataset:
        continue
    gt_dataset_path = gt_dataset[gt_dataset_name]["path"]
    gt_dataset_scenes = gt_dataset[gt_dataset_name]["scenes"]
    
    for scene in gt_dataset_scenes:
        result_path = os.path.join(result_main_folder, result, scene)
        if not os.path.isdir(result_path):
            continue
        gt_path = os.path.join(gt_dataset_path, scene)
        # if not os.path.exists(os.path.join(result_path, "eval.txt")):
        skip_tracking_eval = ""
        if "rsl" in result.lower():
            skip_tracking_eval = "--skip_trajectory_eval --skip_error_vis"
        if "mono" in result.lower():
            os.system(
                "python3 run.py {} {} --correct_scale {} --skip_error_vis".format(
                    result_path, gt_path, skip_tracking_eval
                )
            )
        else:
            os.system(
                "python3 run.py {} {} {} --skip_error_vis".format(result_path, gt_path, skip_tracking_eval)
            )


logs = []
camera_type = ["mono", "rgbd", "stereo"]
#### get the result file ####
for gt_dataset_name in gt_dataset:
    # mono
    scenes = gt_dataset[gt_dataset_name]["scenes"]
    for camera in camera_type:
        results = sorted(
            glob.glob(
                os.path.join(
                    result_main_folder,
                    "{}_{}*".format(gt_dataset_name, camera),
                )
            )
        )
        for result in results:
            print(result)
            logs.append(result + "\n")
            for scene in scenes:
                # T	R PSNR SSIM	LPIPS Tracking speed Rendering speed
                vram_usage = -1.0
                vram_usage_path = os.path.join(result, scene, "GpuPeakUsageMB.txt")
                if os.path.exists(vram_usage_path):
                    with open(vram_usage_path, "r") as f:
                        for line in f:
                            if "Peak allocated (MB):" in line:
                                vram_usage = float(line.split(":")[1].strip())
                                break
                        
                T, R, T_std = None, None, None
                if os.path.exists(
                    os.path.join(result, scene, "metrics_traj.txt")
                ):
                    with open(
                        os.path.join(result, scene, "metrics_traj.txt")
                    ) as fin:
                        lines = fin.readlines()
                        ape_T = lines[7].split()
                        assert ape_T[0] == "rmse", result
                        T = ape_T[-1]
                        T_std = lines[9].split()[-1]
                        ape_R = lines[17].split()
                        assert ape_R[0] == "rmse", result
                        R = ape_R[-1]
                PSNR, SSIM, LPIPS, Time, Rendering_fps, Num_Gaussians = (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                if os.path.exists(os.path.join(result, scene, "eval.txt")):
                    with open(os.path.join(result, scene, "eval.txt")) as fin:
                        PSNR = fin.readline().split()[-1]
                        SSIM = fin.readline().split()[-1]
                        LPIPS = fin.readline().split()[-1]
                        Time = fin.readline().split()[-1]
                        Rendering_time = fin.readline().split()[-1]
                        Rendering_fps = fin.readline().split()[-1]
                        Num_Gaussians = fin.readline().split()[-1]

                render_path = glob.glob(
                    os.path.join(result, scene, "*shutdown", "render_time.txt")
                )
                if len(render_path) > 0:
                    render_time = np.loadtxt(
                        render_path[0], delimiter=" ", dtype=np.str_
                    )
                    render_time = render_time[:, 1].astype(np.float32)
                    Rendering_fps = 1000 / np.mean(render_time)

                Chamfer = None
                mesh_metrics = {}
                if gt_dataset_name == "replica":
                    pred_mesh_matches = glob.glob(os.path.join(result, scene, "*_shutdown", "data", "mesh.off"))
                    gt_mesh   = os.path.join(gt_dataset[gt_dataset_name]["path"], f"{scene}_mesh.ply")
                    pred_mesh = pred_mesh_matches[0] if pred_mesh_matches else None
                    if pred_mesh and os.path.exists(pred_mesh) and os.path.exists(gt_mesh):
                        try:
                            mesh_metrics = chamfer_l2.compute_all(pred_mesh, gt_mesh)
                            Chamfer = mesh_metrics["chamfer_l2"]
                            print(f"  {scene}: Ch-L2={Chamfer:.6f}  Acc={mesh_metrics['accuracy_cm']:.2f}cm  "
                                  f"Comp={mesh_metrics['completion_cm']:.2f}cm  "
                                  f"CompR={mesh_metrics['completion_ratio_%']:.1f}%  "
                                  f"F={mesh_metrics['f_score_%']:.1f}%")
                        except Exception as e:
                            print(f"Chamfer failed for {scene}: {e}")

                result_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                    scene,
                    T,
                    R,
                    PSNR,
                    SSIM,
                    LPIPS,
                    Time,
                    Rendering_fps,
                    Num_Gaussians,
                    vram_usage,
                    Chamfer,
                    mesh_metrics.get("accuracy_cm"),
                    mesh_metrics.get("completion_cm"),
                    mesh_metrics.get("f_score_%"),
                )
                print(result_str)
                logs.append(result_str)
# rgbd
with open(os.path.join(result_main_folder, "log.txt"), "w") as out_file:
    for log in logs:
        out_file.write(log)


with open(os.path.join(result_main_folder, "log.csv"), "w") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(
        (
            "scene",
            "T",
            "R",
            "PSNR",
            "SSIM",
            "LPIPS",
            "Time",
            "Rendering FPS",
            "Num Gaussians",
            "VRAM Usage",
            "Chamfer-L2",
            "Accuracy (cm)",
            "Completion (cm)",
            "F-score (%)",
        )
    )
    for log in logs:
        writer.writerow(log.split())
