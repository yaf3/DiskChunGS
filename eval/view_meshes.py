"""Quick mesh viewer using Open3D — no Isaac Sim required.

Runs the same gravity-align + ICP pipeline as collision_fidelity.py,
then shows the result in an Open3D window or saves a screenshot.

Usage:
    # Pred only (gravity-aligned):
    python3 eval/view_meshes.py \
        --pred_mesh results/mesh_test/room0/4108_shutdown/data/mesh.obj \
        --est_traj results/mesh_test/room0/KeyFrameTrajectory_TUM.txt

    # Pred + GT comparison:
    python3 eval/view_meshes.py \
        --pred_mesh results/mesh_test/room0/4108_shutdown/data/mesh.obj \
        --est_traj results/mesh_test/room0/KeyFrameTrajectory_TUM.txt \
        --gt_mesh data/Replica/room0_mesh.ply \
        --gt_traj data/Replica/room0/pose_TUM.txt \
        --flip_gt

    # Save screenshot instead of interactive viewer:
    python3 eval/view_meshes.py ... --screenshot out.png
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(__file__))
import mesh_metrics


def main():
    parser = argparse.ArgumentParser(description="Quick mesh viewer (Open3D)")
    parser.add_argument("--pred_mesh", type=str, required=True)
    parser.add_argument("--est_traj", type=str, required=True)
    parser.add_argument("--gt_mesh", type=str, default=None)
    parser.add_argument("--gt_traj", type=str, default=None)
    parser.add_argument("--flip_gt", action="store_true")
    parser.add_argument("--force_align", action="store_true")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Save screenshot to this path instead of interactive viewer")
    parser.add_argument("--no_color", action="store_true",
                        help="Color meshes by identity (blue=pred, green=GT) instead of vertex colors")
    args = parser.parse_args()

    geometries = []

    if args.gt_mesh and args.gt_traj:
        pred_out, gt_out = mesh_metrics.align_pred_to_gt(
            args.pred_mesh, args.gt_mesh, args.est_traj, args.gt_traj,
            force=args.force_align, flip_gt=args.flip_gt,
        )
        pred_mesh = mesh_metrics._load_mesh(pred_out)
        gt_mesh = mesh_metrics._load_mesh(gt_out)
        print(f"Pred mesh: {len(pred_mesh.vertices)} verts, {len(pred_mesh.triangles)} tris")
        print(f"GT mesh:   {len(gt_mesh.vertices)} verts, {len(gt_mesh.triangles)} tris")

        if args.no_color:
            pred_mesh.paint_uniform_color([0.3, 0.5, 1.0])
            gt_mesh.paint_uniform_color([0.3, 0.9, 0.4])
        else:
            if not pred_mesh.has_vertex_colors():
                pred_mesh.paint_uniform_color([0.3, 0.5, 1.0])
            if not gt_mesh.has_vertex_colors():
                gt_mesh.paint_uniform_color([0.3, 0.9, 0.4])

        pred_bb = pred_mesh.get_axis_aligned_bounding_box()
        gt_bb = gt_mesh.get_axis_aligned_bounding_box()
        print(f"Pred bounds: {pred_bb.min_bound} — {pred_bb.max_bound}")
        print(f"GT   bounds: {gt_bb.min_bound} — {gt_bb.max_bound}")

        geometries = [pred_mesh, gt_mesh]
    else:
        pred_mesh = mesh_metrics._load_mesh(args.pred_mesh)
        print(f"Pred mesh: {len(pred_mesh.vertices)} verts, {len(pred_mesh.triangles)} tris")
        mesh_metrics.gravity_align(pred_mesh, args.est_traj)
        if args.no_color or not pred_mesh.has_vertex_colors():
            pred_mesh.paint_uniform_color([0.3, 0.5, 1.0])
        geometries = [pred_mesh]

    pred_mesh.compute_vertex_normals()
    if args.gt_mesh:
        gt_mesh.compute_vertex_normals()

    # Add coordinate frame at origin for reference
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(axes)

    # Ground plane grid at Z=0 (matches Isaac Sim floor)
    all_verts = np.asarray(pred_mesh.vertices)
    if args.gt_mesh:
        all_verts = np.vstack([all_verts, np.asarray(gt_mesh.vertices)])
    margin = 1.0
    x_min, x_max = all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin
    y_min, y_max = all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin
    grid_lines = []
    grid_points = []
    step = 0.5
    idx = 0
    for x in np.arange(x_min, x_max + step, step):
        grid_points.extend([[x, y_min, 0.0], [x, y_max, 0.0]])
        grid_lines.append([idx, idx + 1])
        idx += 2
    for y in np.arange(y_min, y_max + step, step):
        grid_points.extend([[x_min, y, 0.0], [x_max, y, 0.0]])
        grid_lines.append([idx, idx + 1])
        idx += 2
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(grid_points)
    grid.lines = o3d.utility.Vector2iVector(grid_lines)
    grid.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(grid)

    if args.screenshot:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        for g in geometries:
            vis.add_geometry(g)
        vis.get_render_option().mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.screenshot)
        vis.destroy_window()
        print(f"Screenshot saved to {args.screenshot}")
    else:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Mesh Viewer",
            width=1920, height=1080,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()
