"""Navigation benchmark in Isaac Sim.

Spawns a simple robot in a SLAM-reconstructed mesh environment and navigates
between waypoints. Measures collision rate, path completion, and safety margins.

Compares navigation on the predicted mesh vs GT mesh to measure how much the
reconstruction quality affects downstream robot performance.

The meshes are gravity-aligned using camera poses so that physics simulation
works correctly even when the SLAM frame is tilted.

Usage (inside Isaac Sim container):
    cd /workspace/IsaacLab
    ./isaaclab.sh -p /workspace/repo/eval/navigation_benchmark.py \
        --pred_mesh /workspace/repo/results/mesh_test/room0/4108_shutdown/data/mesh.off \
        --gt_mesh /path/to/Replica/room0_mesh.ply \
        --est_traj /workspace/repo/results/mesh_test/room0/CameraTrajectory_TUM.txt \
        --gt_traj /workspace/repo/data/Replica/room0/pose_TUM.txt \
        --headless
"""

import argparse
import os
import sys
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Navigation benchmark on SLAM meshes")
parser.add_argument("--pred_mesh", type=str, required=True, help="Path to predicted mesh (OFF/PLY/OBJ)")
parser.add_argument("--gt_mesh", type=str, required=True, help="Path to GT mesh (PLY/OBJ)")
parser.add_argument("--est_traj", type=str, required=True, help="Estimated trajectory (TUM format) for gravity alignment")
parser.add_argument("--gt_traj", type=str, default=None, help="GT trajectory (TUM format) for GT mesh alignment")
parser.add_argument("--num_waypoints", type=int, default=10, help="Number of waypoints per trial")
parser.add_argument("--num_trials", type=int, default=5, help="Number of navigation trials")
parser.add_argument("--robot_radius", type=float, default=0.15, help="Robot collision radius (m)")
parser.add_argument("--robot_speed", type=float, default=0.5, help="Robot movement speed (m/s)")
parser.add_argument("--max_steps_per_wp", type=int, default=500, help="Max physics steps per waypoint")
parser.add_argument("--waypoint_tolerance", type=float, default=0.2, help="Distance to consider waypoint reached (m)")
parser.add_argument("--output", type=str, default=None, help="Output file path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import open3d as o3d
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

sys.path.insert(0, os.path.dirname(__file__))
import mesh_metrics


def gravity_align_mesh(mesh_path, traj_path):
    """Rotate mesh so SLAM gravity aligns with -Z. Returns path to aligned OBJ."""
    aligned_path = mesh_path.rsplit(".", 1)[0] + "_aligned.obj"
    if os.path.exists(aligned_path):
        return aligned_path

    gravity_dir = mesh_metrics.estimate_gravity_direction(traj_path)
    R = mesh_metrics.compute_gravity_rotation(gravity_dir)

    mesh = mesh_metrics._load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    verts_rotated = (R @ verts.T).T
    mesh.vertices = o3d.utility.Vector3dVector(verts_rotated)
    o3d.io.write_triangle_mesh(aligned_path, mesh)
    print(f"Gravity-aligned mesh saved to {aligned_path}")
    print(f"  Estimated gravity dir: {gravity_dir}")
    return aligned_path


def import_mesh_to_stage(obj_path, prim_path):
    """Import OBJ mesh into USD stage with triangle mesh collision."""
    usd_dir = obj_path.rsplit(".", 1)[0] + "_usd"
    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=usd_dir,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.MeshCollisionBaseCfg(
            mesh_approximation_name="none",
        ),
        make_instanceable=False,
    )
    converter = MeshConverter(cfg)
    sim_utils.create_prim(prim_path, usd_path=converter.usd_path)
    return converter.usd_path


def sample_navigable_points(obj_path, n_points, floor_height_percentile=10):
    """Sample navigable XY positions on the floor of the aligned mesh."""
    mesh = mesh_metrics._load_mesh(obj_path)
    verts = np.asarray(mesh.vertices)

    floor_z = np.percentile(verts[:, 2], floor_height_percentile)
    floor_verts = verts[np.abs(verts[:, 2] - floor_z) < 0.3]

    if len(floor_verts) < n_points:
        floor_verts = verts

    bbox_min = floor_verts.min(axis=0)
    bbox_max = floor_verts.max(axis=0)
    margin = 0.2 * (bbox_max - bbox_min)
    bbox_min[:2] += margin[:2]
    bbox_max[:2] -= margin[:2]

    points = np.zeros((n_points, 3))
    points[:, 0] = np.random.uniform(bbox_min[0], bbox_max[0], n_points)
    points[:, 1] = np.random.uniform(bbox_min[1], bbox_max[1], n_points)
    points[:, 2] = floor_z + 0.1

    return points


def create_robot(prim_path, position, radius):
    """Create a simple sphere robot with rigid body physics."""
    prim = sim_utils.create_prim(
        prim_path,
        prim_type="Sphere",
        translation=position.tolist(),
        attributes={"radius": radius},
    )
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(1.0)
    return prim


def get_prim_position(prim):
    """Get world position of a prim."""
    xform = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(0)
    pos = transform.ExtractTranslation()
    return np.array([pos[0], pos[1], pos[2]])


def apply_velocity(prim, direction, speed):
    """Apply velocity to robot toward target."""
    vel = direction * speed
    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.GetVelocityAttr().Set(Gf.Vec3f(float(vel[0]), float(vel[1]), float(vel[2])))


def run_navigation_trial(sim, robot_prim, waypoints, robot_speed,
                         max_steps_per_wp, waypoint_tolerance, physics_dt):
    """Run one navigation trial. Returns metrics dict."""
    total_collisions = 0
    waypoints_reached = 0
    total_distance = 0.0
    total_steps = 0

    prev_pos = get_prim_position(robot_prim)

    for wp_idx, target in enumerate(waypoints):
        reached = False
        for step in range(max_steps_per_wp):
            pos = get_prim_position(robot_prim)

            dist_to_target = np.linalg.norm(pos[:2] - target[:2])
            if dist_to_target < waypoint_tolerance:
                reached = True
                waypoints_reached += 1
                break

            direction = target - pos
            direction[2] = 0
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist

            apply_velocity(robot_prim, direction, robot_speed)

            sim.step()
            total_steps += 1

            new_pos = get_prim_position(robot_prim)
            step_dist = np.linalg.norm(new_pos - prev_pos)
            total_distance += step_dist

            vel_magnitude = step_dist / physics_dt if physics_dt > 0 else 0
            if vel_magnitude < robot_speed * 0.1 and dist_to_target > waypoint_tolerance:
                total_collisions += 1

            prev_pos = new_pos

    path_efficiency = total_distance / max(
        sum(np.linalg.norm(waypoints[i+1][:2] - waypoints[i][:2])
            for i in range(len(waypoints)-1)), 1e-6
    )

    return {
        "waypoints_reached": waypoints_reached,
        "total_waypoints": len(waypoints),
        "completion_rate_%": waypoints_reached / len(waypoints) * 100,
        "total_collisions": total_collisions,
        "collision_rate": total_collisions / max(total_steps, 1),
        "total_distance_m": total_distance,
        "path_efficiency": min(path_efficiency, 10.0),
        "total_steps": total_steps,
    }


def run_on_mesh(sim, mesh_obj_path, mesh_prim_path, waypoint_sets,
                robot_radius, robot_speed, max_steps_per_wp,
                waypoint_tolerance, physics_dt):
    """Run all navigation trials on one mesh."""
    stage = sim_utils.get_current_stage()

    if stage.GetPrimAtPath(mesh_prim_path):
        sim_utils.delete_prim(mesh_prim_path)
    if stage.GetPrimAtPath("/World/robot"):
        sim_utils.delete_prim("/World/robot")

    import_mesh_to_stage(mesh_obj_path, mesh_prim_path)

    all_trial_results = []

    for trial_idx, waypoints in enumerate(waypoint_sets):
        print(f"  Trial {trial_idx + 1}/{len(waypoint_sets)}")

        if stage.GetPrimAtPath("/World/robot"):
            sim_utils.delete_prim("/World/robot")

        start_pos = waypoints[0].copy()
        robot_prim = create_robot("/World/robot", start_pos, robot_radius)

        sim.reset()

        result = run_navigation_trial(
            sim, robot_prim, waypoints[1:],
            robot_speed, max_steps_per_wp, waypoint_tolerance, physics_dt
        )
        all_trial_results.append(result)

        for k, v in result.items():
            print(f"    {k}: {v}")

    sim_utils.delete_prim(mesh_prim_path)
    if stage.GetPrimAtPath("/World/robot"):
        sim_utils.delete_prim("/World/robot")

    return all_trial_results


def aggregate_results(trial_results):
    """Average metrics across trials."""
    if not trial_results:
        return {}

    keys = [k for k in trial_results[0].keys() if isinstance(trial_results[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in trial_results]
        agg[f"mean_{k}"] = float(np.mean(vals))
        agg[f"std_{k}"] = float(np.std(vals))
    return agg


def main():
    # Gravity-align pred mesh using estimated trajectory
    pred_aligned = gravity_align_mesh(args_cli.pred_mesh, args_cli.est_traj)

    # GT mesh: align using GT trajectory if provided, otherwise assume already level
    if args_cli.gt_traj:
        gt_aligned = gravity_align_mesh(args_cli.gt_mesh, args_cli.gt_traj)
    else:
        gt_obj = args_cli.gt_mesh.rsplit(".", 1)[0] + ".obj"
        if not os.path.exists(gt_obj):
            mesh = mesh_metrics._load_mesh(args_cli.gt_mesh)
            o3d.io.write_triangle_mesh(gt_obj, mesh)
        gt_aligned = gt_obj

    physics_dt = 1.0 / 120.0
    sim_cfg = SimulationCfg(dt=physics_dt, use_fabric=False)
    sim = SimulationContext(sim_cfg)

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground, translation=[0, 0, -20])

    sim.reset()

    # Waypoints sampled from GT mesh (fair comparison)
    np.random.seed(42)
    waypoint_sets = []
    for _ in range(args_cli.num_trials):
        wps = sample_navigable_points(gt_aligned, args_cli.num_waypoints)
        waypoint_sets.append(wps)

    print(f"\n{'='*60}")
    print("Navigation on PREDICTED mesh (your SLAM output)")
    print(f"{'='*60}")
    pred_results = run_on_mesh(
        sim, pred_aligned, "/World/env_mesh", waypoint_sets,
        args_cli.robot_radius, args_cli.robot_speed,
        args_cli.max_steps_per_wp, args_cli.waypoint_tolerance, physics_dt
    )

    print(f"\n{'='*60}")
    print("Navigation on GT mesh")
    print(f"{'='*60}")
    gt_results = run_on_mesh(
        sim, gt_aligned, "/World/env_mesh", waypoint_sets,
        args_cli.robot_radius, args_cli.robot_speed,
        args_cli.max_steps_per_wp, args_cli.waypoint_tolerance, physics_dt
    )

    pred_agg = aggregate_results(pred_results)
    gt_agg = aggregate_results(gt_results)

    out_path = args_cli.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args_cli.pred_mesh), "navigation_benchmark.txt")

    with open(out_path, "w") as f:
        f.write("[predicted_mesh]\n")
        for k, v in pred_agg.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n[gt_mesh]\n")
        for k, v in gt_agg.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n[delta (pred - gt)]\n")
        for k in pred_agg:
            if k in gt_agg:
                delta = pred_agg[k] - gt_agg[k]
                f.write(f"{k}: {delta:+.6f}\n")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nPredicted mesh:")
    for k, v in pred_agg.items():
        print(f"  {k}: {v:.4f}")
    print("\nGT mesh:")
    for k, v in gt_agg.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n-> {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
