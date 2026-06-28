"""Collision fidelity evaluation in Isaac Sim.

Two evaluation modes:
  1. Collision approximation: compares collision modes (convexDecomposition,
     convexHull, meshSimplification) against raw triangles on the SAME mesh.
  2. Pred vs GT (when --gt_mesh given): drops probes on both pred and GT meshes
     to measure how SLAM reconstruction distorts collision boundaries.

Meshes are gravity-aligned using camera poses so that physics simulation
works correctly even when the SLAM frame is tilted. When GT mesh is provided,
pred is ICP-aligned onto GT for a fair per-probe comparison.

Usage (inside Isaac Sim container):
    cd /workspace/IsaacLab
    ./isaaclab.sh -p /workspace/repo/eval/collision_fidelity.py \
        --pred_mesh /workspace/repo/results/mesh_test/room0/4108_shutdown/data/mesh.off \
        --est_traj /workspace/repo/results/mesh_test/room0/CameraTrajectory_TUM.txt \
        --gt_mesh /workspace/repo/data/Replica/room0_mesh.ply \
        --gt_traj /workspace/repo/data/Replica/room0/pose_TUM.txt
"""

import argparse
import os
import sys
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collision fidelity evaluation")
parser.add_argument("--pred_mesh", type=str, required=True, help="Path to predicted mesh (OFF/PLY/OBJ)")
parser.add_argument("--est_traj", type=str, required=True, help="Estimated trajectory (TUM format) for gravity alignment")
parser.add_argument("--gt_mesh", type=str, default=None, help="Path to GT mesh (PLY/OBJ) for pred-vs-GT comparison")
parser.add_argument("--gt_traj", type=str, default=None, help="GT trajectory (TUM format) for GT mesh gravity alignment")
parser.add_argument("--num_probes", type=int, default=200, help="Number of probe spheres to drop")
parser.add_argument("--probe_radius", type=float, default=0.02, help="Probe sphere radius (m)")
parser.add_argument("--settle_steps", type=int, default=300, help="Physics steps to let probes settle")
parser.add_argument("--output", type=str, default=None, help="Output file path (default: next to pred_mesh)")
parser.add_argument("--force_align", action="store_true", help="Re-generate aligned meshes even if cached")
parser.add_argument("--flip_gt", action="store_true", help="Negate GT gravity direction (use for OpenGL-convention GT trajectories like Replica)")
parser.add_argument("--gt_only", action="store_true", help="Skip pred-only tests (Parts 1+2), run only GT vs pred comparison")
parser.add_argument("--pause", action="store_true", help="Wait for Enter before each simulation phase (inspect meshes in viewer first)")
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


def gravity_align_mesh(mesh_path, traj_path, force=False):
    """Rotate mesh so SLAM gravity aligns with -Z, RANSAC-flatten floor, snap to Z=0."""
    aligned_path = mesh_path.rsplit(".", 1)[0] + "_aligned.obj"
    if os.path.exists(aligned_path) and not force:
        return aligned_path

    mesh = mesh_metrics._load_mesh(mesh_path)
    mesh_metrics.gravity_align(mesh, traj_path)

    o3d.io.write_triangle_mesh(aligned_path, mesh)
    print(f"  Aligned mesh saved to {aligned_path}")
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
    force = args_cli.force_align
    gt_only = args_cli.gt_only

    pred_aligned = None
    if not gt_only:
        pred_aligned = gravity_align_mesh(args_cli.pred_mesh, args_cli.est_traj, force=force)

    gt_aligned = None
    pred_gt_aligned = None
    if args_cli.gt_mesh:
        print("\nAligning pred to GT and gravity-aligning both...")
        pred_gt_aligned, gt_aligned = mesh_metrics.align_pred_to_gt(
            args_cli.pred_mesh, args_cli.gt_mesh, args_cli.est_traj, args_cli.gt_traj,
            force=force, flip_gt=args_cli.flip_gt,
        )

    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground, translation=[0, 0, -20])

    pred_baseline_settled = None
    pred_valid_count = 0
    pred_total_count = 0
    pred_fallthrough_rate = 0.0
    approx_results = {}

    # Use GT-aligned pred for probe setup when --gt_only
    probe_mesh_path = pred_gt_aligned if gt_only else pred_aligned
    np.random.seed(42)
    probe_positions = sample_probe_positions(probe_mesh_path, args_cli.num_probes)

    import_mesh_to_stage(probe_mesh_path, "/World/mesh", "none")

    probes = setup_probes(probe_positions, args_cli.probe_radius)

    sim.reset()

    if not gt_only:
        # --- Part 1: Pred mesh baseline (raw triangles) ---
        print(f"\n{'='*60}")
        print("Pred mesh baseline: collision mode 'none' (raw triangles)")
        print(f"{'='*60}")
        if args_cli.pause:
            input(">> Press Enter to drop probes on PRED mesh...")
        pred_baseline_settled = run_test(sim, probes, probe_positions, args_cli.settle_steps)

        initial_z = probe_positions[:, 2].mean()
        settled_z = pred_baseline_settled[:, 2].mean()
        print(f"  Initial avg Z={initial_z:.3f}, Settled avg Z={settled_z:.3f}")

        pred_valid = pred_baseline_settled[:, 2] > -10
        pred_valid_count = int(pred_valid.sum())
        pred_total_count = len(pred_baseline_settled)
        pred_fallthrough_rate = float(1.0 - pred_valid.mean()) * 100.0

        print(f"  Valid Probes: {pred_valid_count}/{pred_total_count}")
        print(f"  Fallthrough Rate: {pred_fallthrough_rate:.2f}%")

        if abs(initial_z - settled_z) < 0.01:
            print("  WARNING: Probes did not move! Physics may not be active.")
            print("  Aborting — fix physics before testing collision modes.")
            simulation_app.close()
            return

        # --- Part 2: Collision approximation modes on pred mesh ---
        test_modes = ["convexDecomposition", "convexHull", "meshSimplification"]

        for mode in test_modes:
            print(f"\n{'='*60}")
            print(f"Pred mesh collision mode: {mode}")
            print(f"{'='*60}")

            try:
                sim_utils.delete_prim("/World/mesh")
                import_mesh_to_stage(pred_aligned, "/World/mesh", mode)

                test_settled = run_test(sim, probes, probe_positions, args_cli.settle_steps)
                metrics = compute_collision_metrics(pred_baseline_settled, test_settled)
                approx_results[mode] = metrics

                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                approx_results[mode] = {"error": str(e)}

    # --- Part 3: Pred vs GT comparison ---
    gt_vs_pred = None
    gt_baseline_settled = None
    gt_valid_count = 0
    gt_total_count = 0
    gt_fallthrough_rate = 0.0

    if gt_aligned:
        gt_probe_positions = sample_probe_positions(gt_aligned, args_cli.num_probes)

        print(f"\n{'='*60}")
        print("GT mesh baseline: collision mode 'none' (raw triangles)")
        print(f"{'='*60}")

        sim_utils.delete_prim("/World/mesh")
        import_mesh_to_stage(gt_aligned, "/World/mesh", "none")

        if args_cli.pause:
            input(">> Press Enter to drop probes on GT mesh...")
        gt_baseline_settled = run_test(sim, probes, gt_probe_positions, args_cli.settle_steps)

        gt_valid = gt_baseline_settled[:, 2] > -10
        gt_valid_count = int(gt_valid.sum())
        gt_total_count = len(gt_baseline_settled)
        gt_fallthrough_rate = float(1.0 - gt_valid.mean()) * 100.0

        print(f"  Valid Probes: {gt_valid_count}/{gt_total_count}")
        print(f"  Fallthrough Rate: {gt_fallthrough_rate:.2f}%")

        print(f"\n{'='*60}")
        print("Pred (ICP-aligned) on GT probe positions")
        print(f"{'='*60}")

        sim_utils.delete_prim("/World/mesh")
        import_mesh_to_stage(pred_gt_aligned, "/World/mesh", "none")

        if args_cli.pause:
            input(">> Press Enter to drop probes on PRED (ICP-aligned to GT)...")
        pred_on_gt_settled = run_test(sim, probes, gt_probe_positions, args_cli.settle_steps)

        gt_vs_pred = compute_collision_metrics(gt_baseline_settled, pred_on_gt_settled)
        print(f"\n  Pred vs GT deviation:")
        for k, v in gt_vs_pred.items():
            print(f"    {k}: {v}")

    # --- Write results ---
    out_path = args_cli.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args_cli.pred_mesh), "collision_fidelity.txt")

    with open(out_path, "w") as f:
        if pred_baseline_settled is not None:
            f.write("[pred baseline: none (raw triangles)]\n")
            f.write(f"valid_probes: {pred_valid_count}\n")
            f.write(f"total_probes: {pred_total_count}\n")
            f.write(f"fallthrough_rate_%: {pred_fallthrough_rate:.6f}\n\n")

            for mode, metrics in approx_results.items():
                f.write(f"[pred {mode} vs none]\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

        if gt_baseline_settled is not None:
            f.write("[gt baseline: none (raw triangles)]\n")
            f.write(f"valid_probes: {gt_valid_count}\n")
            f.write(f"total_probes: {gt_total_count}\n")
            f.write(f"fallthrough_rate_%: {gt_fallthrough_rate:.6f}\n\n")

            f.write("[pred vs gt]\n")
            for k, v in gt_vs_pred.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if pred_baseline_settled is not None:
        print(f"\n  [pred none (baseline)]")
        print(f"    fallthrough_rate_%: {pred_fallthrough_rate:.2f}%")
        print(f"    valid_probes: {pred_valid_count}/{pred_total_count}")

        for mode, metrics in approx_results.items():
            print(f"\n  [pred {mode} vs none]")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

    if gt_vs_pred:
        print(f"\n  [gt none (baseline)]")
        print(f"    fallthrough_rate_%: {gt_fallthrough_rate:.2f}%")
        print(f"    valid_probes: {gt_valid_count}/{gt_total_count}")
        print(f"\n  [pred vs gt]")
        for k, v in gt_vs_pred.items():
            print(f"    {k}: {v}")

    print(f"\n-> {out_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
