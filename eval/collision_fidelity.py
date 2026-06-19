"""Collision fidelity evaluation in Isaac Sim.

Loads a predicted mesh, applies different collision approximation modes, and
measures how much each mode distorts the collision boundary compared to raw
triangle mesh collision ("none") on the SAME mesh.

The mesh is gravity-aligned using camera poses so that physics simulation
(probe drops) works correctly even when the SLAM frame is tilted.

Usage (inside Isaac Sim container):
    cd /workspace/IsaacLab
    ./isaaclab.sh -p /workspace/repo/eval/collision_fidelity.py \
        --pred_mesh /workspace/repo/results/mesh_test/room0/4108_shutdown/data/mesh.off \
        --est_traj /workspace/repo/results/mesh_test/room0/CameraTrajectory_TUM.txt
"""

import argparse
import os
import sys
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collision fidelity evaluation")
parser.add_argument("--pred_mesh", type=str, required=True, help="Path to predicted mesh (OFF/PLY/OBJ)")
parser.add_argument("--est_traj", type=str, required=True, help="Estimated trajectory (TUM format) for gravity alignment")
parser.add_argument("--num_probes", type=int, default=200, help="Number of probe spheres to drop")
parser.add_argument("--probe_radius", type=float, default=0.02, help="Probe sphere radius (m)")
parser.add_argument("--settle_steps", type=int, default=300, help="Physics steps to let probes settle")
parser.add_argument("--output", type=str, default=None, help="Output file path (default: next to pred_mesh)")
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


def perfect_floor_flattening(mesh_path, output_path):
    """Finds the dominant floor plane using RANSAC and rotates the mesh so the floor is perfectly flat."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.03, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    floor_normal = np.array([a, b, c])
    
    if floor_normal[2] < 0:
        floor_normal = -floor_normal
        
    print(f"  Detected floor plane normal: {floor_normal}")
    
    world_up = np.array([0.0, 0.0, 1.0])
    
    v = np.cross(floor_normal, world_up)
    cos_theta = np.dot(floor_normal, world_up)
    
    if np.linalg.norm(v) < 1e-6:
        R_flatten = np.eye(3)
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_flatten = np.eye(3) + kmat + kmat @ kmat * ((1 - cos_theta) / (s ** 2))

    mesh.rotate(R_flatten, center=(0, 0, 0))
    
    verts = np.asarray(mesh.vertices)
    avg_floor_z = np.mean(verts[inliers, 2])
    mesh.translate([0, 0, -avg_floor_z])
    
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"  Floor flattened and snapped to Z=0. Saving to {output_path}")
    return output_path


def gravity_align_mesh(mesh_path, traj_path):
    """Rotate mesh so SLAM gravity aligns with -Z, then micro-flattens the floor footprint."""
    aligned_path = mesh_path.rsplit(".", 1)[0] + "_perfectly_aligned.obj"
    if os.path.exists(aligned_path):
        return aligned_path

    gravity_dir = mesh_metrics.estimate_gravity_direction(traj_path)
    R = mesh_metrics.compute_gravity_rotation(gravity_dir)

    mesh = mesh_metrics._load_mesh(mesh_path)
    mesh.rotate(R, center=(0, 0, 0))
    
    temp_path = mesh_path.rsplit(".", 1)[0] + "_temp.obj"
    o3d.io.write_triangle_mesh(temp_path, mesh)

    perfect_floor_flattening(temp_path, aligned_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return aligned_path


def import_mesh_to_stage(obj_path, prim_path, collision_approximation="none"):
    """Import OBJ mesh into USD stage with specified collision mode."""
    usd_dir = obj_path.rsplit(".", 1)[0] + f"_{collision_approximation}_usd"
    cfg = MeshConverterCfg(
        asset_path=obj_path,
        usd_dir=usd_dir,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.MeshCollisionBaseCfg(
            mesh_approximation_name=collision_approximation,
        ),
        make_instanceable=False,
        force_usd_conversion=True,
    )
    converter = MeshConverter(cfg)
    sim_utils.create_prim(prim_path, usd_path=converter.usd_path)
    return converter.usd_path


def sample_probe_positions(mesh_path, n_points):
    """Sample probe positions safely inside the room by offsetting upward from the floor."""
    mesh = mesh_metrics._load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)

    floor_threshold = np.percentile(verts[:, 2], 15)
    floor_verts = verts[verts[:, 2] <= floor_threshold]

    if len(floor_verts) == 0:
        floor_verts = verts  # Fallback if mesh is completely flat

    replace = len(floor_verts) < n_points
    indices = np.random.choice(len(floor_verts), size=n_points, replace=replace)
    sampled_floor_points = floor_verts[indices].copy()

    positions = sampled_floor_points
    positions[:, 2] += 0.5  

    print(f"  Probes safely generated inside: 50cm above local floor surfaces.")
    return positions


def setup_probes(probe_positions, probe_radius):
    """Create probe origins and RigidObject for all probes."""
    n = len(probe_positions)
    for i, pos in enumerate(probe_positions):
        sim_utils.create_prim(f"/World/Probe{i}", "Xform", translation=tuple(pos.tolist()))

    probe_cfg = RigidObjectCfg(
        prim_path="/World/Probe.*/sphere",
        spawn=sim_utils.SphereCfg(
            radius=probe_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    probes = RigidObject(cfg=probe_cfg)
    return probes


def compute_collision_metrics(baseline_settled, test_settled):
    """Compare settled positions between two collision runs."""
    diffs = np.linalg.norm(baseline_settled - test_settled, axis=1)

    valid = (baseline_settled[:, 2] > -10) & (test_settled[:, 2] > -10)
    if valid.sum() == 0:
        return {"error": "all probes fell through"}

    diffs_valid = diffs[valid]

    return {
        "mean_deviation_cm": float(np.mean(diffs_valid)) * 100,
        "median_deviation_cm": float(np.median(diffs_valid)) * 100,
        "max_deviation_cm": float(np.max(diffs_valid)) * 100,
        "std_deviation_cm": float(np.std(diffs_valid)) * 100,
        "p90_deviation_cm": float(np.percentile(diffs_valid, 90)) * 100,
        "p99_deviation_cm": float(np.percentile(diffs_valid, 99)) * 100,
        "valid_probes": int(valid.sum()),
        "total_probes": len(valid),
        "fallthrough_rate_%": float(1.0 - valid.mean()) * 100,
    }


def run_test(sim, probes, probe_positions, settle_steps):
    """Reset probes to start positions, step physics, return settled positions."""
    sim_dt = sim.get_physics_dt()

    root_pose = probes.data.default_root_pose.torch.clone()
    origins = torch.tensor(probe_positions, dtype=torch.float32, device=probes.device)
    root_pose[:, :3] = origins
    probes.write_root_pose_to_sim_index(root_pose=root_pose)
    root_vel = probes.data.default_root_vel.torch.clone()
    probes.write_root_velocity_to_sim_index(root_velocity=root_vel)
    probes.reset()

    for step in range(settle_steps):
        probes.write_data_to_sim()
        sim.step()
        probes.update(sim_dt)
        if step == 0 or step == settle_steps - 1:
            pos = probes.data.root_pos_w.torch
            print(f"    Step {step}: probe_0 Z={pos[0, 2].item():.4f}, "
                  f"avg Z={pos[:, 2].mean().item():.4f}")

    settled = probes.data.root_pos_w.torch.cpu().numpy()
    return settled


def main():
    aligned_obj = gravity_align_mesh(args_cli.pred_mesh, args_cli.est_traj)

    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground, translation=[0, 0, -20])

    np.random.seed(42)
    probe_positions = sample_probe_positions(aligned_obj, args_cli.num_probes)

    import_mesh_to_stage(aligned_obj, "/World/mesh", "none")

    probes = setup_probes(probe_positions, args_cli.probe_radius)

    sim.reset()

    print(f"\n{'='*60}")
    print("Baseline: collision mode 'none' (raw triangles)")
    print(f"{'='*60}")
    baseline_settled = run_test(sim, probes, probe_positions, args_cli.settle_steps)

    initial_z = probe_positions[:, 2].mean()
    settled_z = baseline_settled[:, 2].mean()
    print(f"  Initial avg Z={initial_z:.3f}, Settled avg Z={settled_z:.3f}")
    
    baseline_valid = baseline_settled[:, 2] > -10
    baseline_valid_count = int(baseline_valid.sum())
    baseline_total_count = len(baseline_settled)
    baseline_fallthrough_rate = float(1.0 - baseline_valid.mean()) * 100.0
    
    print(f"  Baseline Valid Probes: {baseline_valid_count}/{baseline_total_count}")
    print(f"  Baseline Fallthrough Rate: {baseline_fallthrough_rate:.2f}%")

    if abs(initial_z - settled_z) < 0.01:
        print("  WARNING: Probes did not move! Physics may not be active.")
        print("  Aborting — fix physics before testing collision modes.")
        simulation_app.close()
        return

    test_modes = ["convexDecomposition", "convexHull", "meshSimplification"]
    results = {}

    for mode in test_modes:
        print(f"\n{'='*60}")
        print(f"Testing collision mode: {mode}")
        print(f"{'='*60}")

        try:
            sim_utils.delete_prim("/World/mesh")
            import_mesh_to_stage(aligned_obj, "/World/mesh", mode)

            test_settled = run_test(sim, probes, probe_positions, args_cli.settle_steps)
            metrics = compute_collision_metrics(baseline_settled, test_settled)
            results[mode] = metrics

            for k, v in metrics.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {"error": str(e)}

    out_path = args_cli.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args_cli.pred_mesh), "collision_fidelity.txt")

    with open(out_path, "w") as f:
        f.write("[baseline: none (raw triangles)]\n")
        f.write(f"valid_probes: {baseline_valid_count}\n")
        f.write(f"total_probes: {baseline_total_count}\n")
        f.write(f"fallthrough_rate_%: {baseline_fallthrough_rate:.6f}\n\n")

        for mode, metrics in results.items():
            f.write(f"[{mode} vs none]\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\n{'='*60}")
    print("SUMMARY (deviation from raw triangle collision)")
    print(f"{'='*60}")
    print(f"\n  [none (Base Mesh)]")
    print(f"    fallthrough_rate_%: {baseline_fallthrough_rate:.2f}%")
    print(f"    valid_probes: {baseline_valid_count}/{baseline_total_count}")
    
    for mode, metrics in results.items():
        print(f"\n  [{mode}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    print(f"\n-> {out_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
