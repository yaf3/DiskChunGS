"""Navigation benchmark in Isaac Sim.

Spawns a simple robot sphere in a SLAM-reconstructed mesh environment and
navigates between waypoints. Measures collision rate, path completion, and
path efficiency.

Compares navigation on the predicted mesh vs GT mesh to measure how much the
reconstruction quality affects downstream robot performance.

Meshes are gravity-aligned using camera poses, floor-flattened via RANSAC, and
ICP-aligned so both meshes share the same coordinate frame.

Usage (inside Isaac Sim container):
    cd /workspace/IsaacLab
    ./isaaclab.sh -p /workspace/repo/eval/navigation_benchmark.py \
        --pred_mesh /workspace/repo/results/mesh_test/room0/4108_shutdown/data/mesh.off \
        --gt_mesh /workspace/repo/data/Replica/room0_mesh.ply \
        --est_traj /workspace/repo/results/mesh_test/room0/CameraTrajectory_TUM.txt \
        --gt_traj /workspace/repo/data/Replica/room0/pose_TUM.txt
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
parser.add_argument("--force_align", action="store_true", help="Re-generate aligned meshes even if cached")
parser.add_argument("--flip_gt", action="store_true", help="Negate GT gravity direction (use for OpenGL-convention GT trajectories like Replica)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import open3d as o3d
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.assets import RigidObject, RigidObjectCfg

sys.path.insert(0, os.path.dirname(__file__))
import mesh_metrics


# ---------------------------------------------------------------------------
# Mesh alignment (shared with collision_fidelity.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Isaac Sim helpers
# ---------------------------------------------------------------------------

def import_mesh_to_stage(obj_path, prim_path):
    """Import OBJ mesh into USD stage with raw triangle collision."""
    usd_dir = obj_path.rsplit(".", 1)[0] + "_none_usd"
    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=usd_dir,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.MeshCollisionBaseCfg(
            mesh_approximation_name="none",
        ),
        make_instanceable=False,
        force_usd_conversion=True,
    )
    converter = MeshConverter(cfg)
    sim_utils.create_prim(prim_path, usd_path=converter.usd_path)
    return converter.usd_path


def sample_navigable_points(obj_path, n_points, robot_radius):
    """Sample navigable positions on the floor of the aligned mesh.

    After flatten_floor, the floor is at Z=0. Robot center goes at Z=robot_radius.
    Validates candidates with raycasting in 6 directions (up, down, and 4 horizontal)
    to reject points inside walls, under furniture, or outside the mesh.
    """
    mesh = mesh_metrics._load_mesh(obj_path)
    verts = np.asarray(mesh.vertices)

    # Floor is at Z≈0 after flatten_floor. Pick vertices near it.
    floor_mask = np.abs(verts[:, 2]) < 0.1
    floor_verts = verts[floor_mask]
    if len(floor_verts) < 100:
        floor_mask = verts[:, 2] < np.percentile(verts[:, 2], 15)
        floor_verts = verts[floor_mask]

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    robot_z = robot_radius + 0.01
    n_candidates = n_points * 20
    candidates = floor_verts[np.random.choice(len(floor_verts), size=n_candidates, replace=True)].copy()
    candidates[:, 2] = robot_z

    def cast(origins, direction):
        rays = np.zeros((len(origins), 6), dtype=np.float32)
        rays[:, :3] = origins
        rays[:, 3:] = direction
        return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()

    # Down: floor must be within robot_radius + small margin (robot sits on it)
    dist_down = cast(candidates, [0, 0, -1])
    valid = (dist_down > 0.001) & (dist_down < robot_radius + 0.05)

    # Up: need room-height clearance (at least 0.5m above robot center)
    dist_up = cast(candidates, [0, 0, 1])
    valid &= dist_up > 0.5

    # Horizontal: 4 cardinal directions, need at least robot_radius clearance
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        dist_h = cast(candidates, [dx, dy, 0])
        valid &= dist_h > robot_radius * 1.5

    valid_points = candidates[valid]
    print(f"  Navigable candidates: {len(valid_points)}/{n_candidates} passed validation")

    if len(valid_points) < n_points:
        print(f"  WARNING: only {len(valid_points)} navigable points found, need {n_points}")
        if len(valid_points) == 0:
            valid_points = candidates[:n_points]
        else:
            idx = np.random.choice(len(valid_points), size=n_points, replace=True)
            valid_points = valid_points[idx]
    else:
        # Spread points out: iteratively pick the farthest point from selected set
        selected = [0]
        for _ in range(n_points - 1):
            dists = np.min([np.linalg.norm(valid_points[:, :2] - valid_points[s, :2], axis=1)
                           for s in selected], axis=0)
            selected.append(np.argmax(dists))
        valid_points = valid_points[selected]

    print(f"  Sampled {len(valid_points)} navigable points at Z={valid_points[0, 2]:.3f}")
    return valid_points


def setup_robot(start_position, robot_radius):
    """Create robot as a RigidObject sphere for proper PhysX registration."""
    sim_utils.create_prim("/World/Robot0", "Xform", translation=tuple(start_position.tolist()))

    robot_cfg = RigidObjectCfg(
        prim_path="/World/Robot.*/sphere",
        spawn=sim_utils.SphereCfg(
            radius=robot_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    robot = RigidObject(cfg=robot_cfg)
    return robot


def teleport_robot(robot, position):
    """Move robot to a new position with zero velocity."""
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[0, :3] = torch.tensor(position, dtype=torch.float32, device=robot.device)
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    root_vel = robot.data.default_root_vel.torch.clone()
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
    robot.reset()


def set_robot_velocity(robot, direction, speed):
    """Set robot's linear velocity toward target (XY only, no vertical)."""
    vel = robot.data.default_root_vel.torch.clone()
    vel[0, 0] = direction[0] * speed
    vel[0, 1] = direction[1] * speed
    vel[0, 2] = 0.0
    robot.write_root_velocity_to_sim_index(root_velocity=vel)


def get_robot_position(robot):
    """Read robot position from PhysX tensor."""
    return robot.data.root_pos_w.torch[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Navigation logic
# ---------------------------------------------------------------------------

def run_navigation_trial(sim, robot, waypoints, robot_speed,
                         max_steps_per_wp, waypoint_tolerance):
    """Run one navigation trial. Returns metrics dict."""
    sim_dt = sim.get_physics_dt()
    total_collisions = 0
    waypoints_reached = 0
    total_distance = 0.0
    total_steps = 0

    prev_pos = get_robot_position(robot)

    for wp_idx, target in enumerate(waypoints):
        for step in range(max_steps_per_wp):
            pos = get_robot_position(robot)

            dist_to_target = np.linalg.norm(pos[:2] - target[:2])
            if dist_to_target < waypoint_tolerance:
                waypoints_reached += 1
                break

            direction = np.zeros(3)
            direction[:2] = target[:2] - pos[:2]
            dist = np.linalg.norm(direction[:2])
            if dist > 0:
                direction /= dist

            set_robot_velocity(robot, direction, robot_speed)

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)
            total_steps += 1

            new_pos = get_robot_position(robot)
            step_dist = np.linalg.norm(new_pos - prev_pos)
            total_distance += step_dist

            vel_magnitude = step_dist / sim_dt if sim_dt > 0 else 0
            if vel_magnitude < robot_speed * 0.1 and dist_to_target > waypoint_tolerance:
                total_collisions += 1

            prev_pos = new_pos

    ideal_distance = sum(
        np.linalg.norm(waypoints[i+1][:2] - waypoints[i][:2])
        for i in range(len(waypoints) - 1)
    )
    path_efficiency = total_distance / max(ideal_distance, 1e-6)

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


def run_on_mesh(sim, robot, mesh_obj_path, mesh_prim_path, waypoint_sets,
                robot_speed, max_steps_per_wp, waypoint_tolerance):
    """Run all navigation trials on one mesh."""
    stage = sim_utils.get_current_stage()

    if stage.GetPrimAtPath(mesh_prim_path):
        sim_utils.delete_prim(mesh_prim_path)

    import_mesh_to_stage(mesh_obj_path, mesh_prim_path)

    all_trial_results = []

    for trial_idx, waypoints in enumerate(waypoint_sets):
        print(f"  Trial {trial_idx + 1}/{len(waypoint_sets)}")

        teleport_robot(robot, waypoints[0])

        result = run_navigation_trial(
            sim, robot, waypoints[1:],
            robot_speed, max_steps_per_wp, waypoint_tolerance
        )
        all_trial_results.append(result)

        for k, v in result.items():
            print(f"    {k}: {v}")

    sim_utils.delete_prim(mesh_prim_path)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nPreparing meshes...")
    pred_aligned, gt_aligned = mesh_metrics.align_pred_to_gt(
        args_cli.pred_mesh, args_cli.gt_mesh, args_cli.est_traj, args_cli.gt_traj,
        force=args_cli.force_align, flip_gt=args_cli.flip_gt,
    )

    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground, translation=[0, 0, -20])

    # Waypoints sampled from GT mesh (fair comparison)
    np.random.seed(42)
    waypoint_sets = []
    for _ in range(args_cli.num_trials):
        wps = sample_navigable_points(gt_aligned, args_cli.num_waypoints, args_cli.robot_radius)
        waypoint_sets.append(wps)

    # Import a dummy mesh so the stage isn't empty at reset
    import_mesh_to_stage(gt_aligned, "/World/env_mesh")

    # Robot starts at the first waypoint of the first trial
    start_pos = waypoint_sets[0][0].copy()
    robot = setup_robot(start_pos, args_cli.robot_radius)

    sim.reset()

    # Sanity check: verify robot is alive
    pos_before = get_robot_position(robot)
    set_robot_velocity(robot, np.array([1.0, 0.0, 0.0]), 1.0)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    pos_after = get_robot_position(robot)
    moved = np.linalg.norm(pos_after - pos_before)
    print(f"\n  Robot sanity check: moved {moved:.4f}m in one step")
    if moved < 1e-4:
        print("  WARNING: Robot did not move! Physics may not be active.")
        print("  Aborting.")
        simulation_app.close()
        return

    # --- Run on predicted mesh ---
    print(f"\n{'='*60}")
    print("Navigation on PREDICTED mesh (your SLAM output)")
    print(f"{'='*60}")
    pred_results = run_on_mesh(
        sim, robot, pred_aligned, "/World/env_mesh", waypoint_sets,
        args_cli.robot_speed, args_cli.max_steps_per_wp, args_cli.waypoint_tolerance
    )

    # --- Run on GT mesh ---
    print(f"\n{'='*60}")
    print("Navigation on GT mesh")
    print(f"{'='*60}")
    gt_results = run_on_mesh(
        sim, robot, gt_aligned, "/World/env_mesh", waypoint_sets,
        args_cli.robot_speed, args_cli.max_steps_per_wp, args_cli.waypoint_tolerance
    )

    pred_agg = aggregate_results(pred_results)
    gt_agg = aggregate_results(gt_results)

    # --- Write results ---
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

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nPredicted mesh:")
    for k, v in pred_agg.items():
        print(f"  {k}: {v:.4f}")
    print("\nGT mesh:")
    for k, v in gt_agg.items():
        print(f"  {k}: {v:.4f}")
    print("\nDelta (pred - gt):")
    for k in pred_agg:
        if k in gt_agg:
            delta = pred_agg[k] - gt_agg[k]
            print(f"  {k}: {delta:+.4f}")

    print(f"\n-> {out_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
