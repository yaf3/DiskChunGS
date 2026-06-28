"""Mesh geometry evaluation metrics.

Usage:
    python3 eval/mesh_metrics.py <pred_mesh.off> <gt_mesh.ply> [--n_points 200000] [--tau 0.05]

Importable:
    from mesh_metrics import compute, compute_all
    chamfer = compute("mesh.off", "mesh.ply")
    metrics = compute_all("mesh.off", "mesh.ply")
"""

import argparse
import json
import os
import struct
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

N_POINTS_DEFAULT = 200_000
TAU_DEFAULT = 0.05  # 5 cm threshold for completion ratio and F-score


# ---------------------------------------------------------------------------
# Mesh loaders
# ---------------------------------------------------------------------------

def _load_off(path):
    """Load OFF or COFF mesh, ignoring vertex colors."""
    with open(path, "r") as f:
        header = f.readline().strip()
        n_verts, n_faces, _ = map(int, f.readline().split())
        verts = np.zeros((n_verts, 3), dtype=np.float64)
        for i in range(n_verts):
            vals = f.readline().split()
            verts[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
        tris = []
        for _ in range(n_faces):
            vals = list(map(int, f.readline().split()))
            poly = vals[1: vals[0] + 1]
            for j in range(1, len(poly) - 1):   # fan triangulation
                tris.append([poly[0], poly[j], poly[j + 1]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    return mesh


_PLY_TYPE = {
    "char": ("b", 1), "uchar": ("B", 1), "int8": ("b", 1), "uint8": ("B", 1),
    "short": ("h", 2), "ushort": ("H", 2), "int16": ("h", 2), "uint16": ("H", 2),
    "int": ("i", 4), "uint": ("I", 4), "int32": ("i", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _load_ply(path):
    """Load PLY mesh (ASCII or binary, n-gon faces) with fan triangulation."""
    with open(path, "rb") as f:
        fmt = "ascii"
        vertex_props = []   # [(name, fmt_char, byte_size)]
        n_verts = n_faces = 0
        in_vertex = in_face = False
        face_list_type = ("I", 4)   # type of each index in a face list

        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            tokens = line.split()
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                in_face   = tokens[1] == "face"
                if in_vertex: n_verts = int(tokens[2])
                if in_face:   n_faces = int(tokens[2])
            elif tokens[0] == "property":
                if in_vertex and tokens[1] != "list":
                    fc, sz = _PLY_TYPE.get(tokens[1], ("f", 4))
                    vertex_props.append((tokens[2], fc, sz))
                elif in_face and tokens[1] == "list":
                    face_list_type = _PLY_TYPE.get(tokens[3], ("I", 4))

        endian = ">" if fmt == "binary_big_endian" else "<"
        is_binary = fmt != "ascii"

        prop_names = [p[0] for p in vertex_props]
        xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        v_fmt  = endian + "".join(p[1] for p in vertex_props)
        v_size = sum(p[2] for p in vertex_props)

        verts = np.zeros((n_verts, 3), dtype=np.float64)
        if is_binary:
            raw = f.read(v_size * n_verts)
            for i in range(n_verts):
                vals = struct.unpack_from(v_fmt, raw, i * v_size)
                verts[i] = [vals[xi], vals[yi], vals[zi]]
        else:
            for i in range(n_verts):
                vals = f.readline().decode().split()
                verts[i] = [float(vals[xi]), float(vals[yi]), float(vals[zi])]

        tris = []
        idx_fc, idx_sz = face_list_type
        for _ in range(n_faces):
            if is_binary:
                count = struct.unpack("B", f.read(1))[0]
                indices = struct.unpack(endian + idx_fc * count, f.read(idx_sz * count))
            else:
                vals = list(map(int, f.readline().decode().split()))
                count, indices = vals[0], vals[1: vals[0] + 1]
            for j in range(1, count - 1):   # fan triangulation
                tris.append([indices[0], indices[j], indices[j + 1]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    return mesh


def _load_mesh(path):
    path = str(path)
    if path.lower().endswith(".off"):
        return _load_off(path)
    if path.lower().endswith(".ply"):
        return _load_ply(path)
    if path.lower().endswith(".obj"):
        import open3d as o3d
        return o3d.io.read_triangle_mesh(path)
    raise ValueError(f"Unsupported mesh format: {path}")


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _load_tum_trajectory(path):
    """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw."""
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            poses.append([float(v) for v in vals[1:4]])
    return np.array(poses)


def _load_tum_trajectory_with_timestamps(path):
    """Load TUM trajectory returning (timestamps, positions)."""
    timestamps = []
    positions = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            timestamps.append(float(vals[0]))
            positions.append([float(v) for v in vals[1:4]])
    return np.array(timestamps), np.array(positions)


def _match_trajectories(est_traj_path, gt_traj_path):
    """Load two TUM trajectories and return timestamp-matched position pairs."""
    est_ts, est_pos = _load_tum_trajectory_with_timestamps(est_traj_path)
    gt_ts, gt_pos = _load_tum_trajectory_with_timestamps(gt_traj_path)
    matched_est = []
    matched_gt = []
    for i, t in enumerate(est_ts):
        j = np.argmin(np.abs(gt_ts - t))
        if abs(gt_ts[j] - t) < 2.0:
            matched_est.append(est_pos[i])
            matched_gt.append(gt_pos[j])
    return np.array(matched_est), np.array(matched_gt)


def _umeyama_alignment(src, dst):
    """Compute rigid SE(3) alignment (no scale) from src to dst point sets.
    Returns 4x4 transform T such that dst ≈ T @ src."""
    assert src.shape == dst.shape
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T
    t = dst_mean - R @ src_mean
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _umeyama_yaw(src, dst):
    """Umeyama alignment constrained to yaw rotation (Z-axis preserved).

    Computes the 2D rigid alignment directly from XY coordinates rather
    than extracting from a 3D Umeyama, which can introduce a spurious
    Z-flip when trajectory positions are nearly coplanar.
    """
    src_xy = src[:, :2]
    dst_xy = dst[:, :2]
    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    H = (src_xy - src_mean).T @ (dst_xy - dst_mean)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R2d = Vt.T @ np.diag([1.0, d]) @ U.T
    R_yaw = np.eye(3)
    R_yaw[:2, :2] = R2d
    t = dst.mean(axis=0) - R_yaw @ src.mean(axis=0)
    T_out = np.eye(4)
    T_out[:3, :3] = R_yaw
    T_out[:3, 3] = t
    return T_out


def _align(pred_pc, gt_pc, est_traj_path=None, gt_traj_path=None):
    """Align pred to GT. Uses trajectory-based Umeyama if paths given, else centroid+ICP."""
    if est_traj_path and gt_traj_path:
        est_pos = _load_tum_trajectory(est_traj_path)
        gt_pos = _load_tum_trajectory(gt_traj_path)
        n = min(len(est_pos), len(gt_pos))
        T = _umeyama_alignment(est_pos[:n], gt_pos[:n])
    else:
        pred_center = pred_pc.get_center()
        gt_center = gt_pc.get_center()
        T = np.eye(4)
        T[:3, 3] = gt_center - pred_center

    reg = o3d.pipelines.registration.registration_icp(
        pred_pc, gt_pc,
        0.2,
        T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    pred_pc.transform(reg.transformation)
    return pred_pc, reg.transformation


# ---------------------------------------------------------------------------
# Gravity estimation from camera poses
# ---------------------------------------------------------------------------

def estimate_gravity_direction(traj_path):
    """Estimate gravity direction in SLAM frame from camera poses.

    In OpenCV convention (ORB-SLAM), camera Y-axis points down.
    Rotating (0,1,0) by each pose's rotation gives the down direction
    in the SLAM world frame. Averaging across poses is robust to noise.

    Returns unit vector pointing in the gravity (down) direction.
    """
    down_vectors = []
    with open(traj_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            q = [float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])]
            R = Rotation.from_quat(q).as_matrix()
            down_vectors.append(R @ np.array([0.0, 1.0, 0.0]))
    avg_down = np.mean(down_vectors, axis=0)
    return avg_down / np.linalg.norm(avg_down)


def compute_gravity_rotation(gravity_dir, target_down=None):
    """Compute rotation matrix that aligns gravity_dir with target_down.

    Default target_down is (0, 0, -1) for Z-up convention (Isaac Sim / USD).
    Returns 3x3 rotation matrix.
    """
    if target_down is None:
        target_down = np.array([0.0, 0.0, -1.0])
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    target_down = target_down / np.linalg.norm(target_down)

    v = np.cross(gravity_dir, target_down)
    c = np.dot(gravity_dir, target_down)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # Anti-parallel: 180° rotation around any perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(gravity_dir[0]) < 0.9 else np.array([0., 1., 0.])
        axis = np.cross(gravity_dir, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    angle = np.arctan2(s, c)
    axis = v / s
    return Rotation.from_rotvec(angle * axis).as_matrix()


# ---------------------------------------------------------------------------
# Floor flattening + gravity alignment (shared by all eval scripts)
# ---------------------------------------------------------------------------

def flatten_floor(mesh, R_floor=None, floor_z=None):
    """RANSAC floor detection + rotation to make floor perfectly flat at Z=0.

    If R_floor and floor_z are provided, applies those instead of detecting.
    Returns (R_floor, floor_z) so the same transform can be reused.
    """
    if R_floor is None:
        pcd = o3d.geometry.PointCloud(mesh.vertices)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.03, ransac_n=3, num_iterations=1000
        )
        floor_normal = np.array(plane_model[:3])
        verts = np.asarray(mesh.vertices)
        plane_d = plane_model[3]
        signed_dists = verts @ floor_normal + plane_d
        if np.sum(signed_dists > 0) < np.sum(signed_dists < 0):
            floor_normal = -floor_normal
        print(f"  Floor plane normal: {floor_normal}")

        world_up = np.array([0.0, 0.0, 1.0])
        v = np.cross(floor_normal, world_up)
        c = np.dot(floor_normal, world_up)
        if np.linalg.norm(v) < 1e-6:
            R_floor = np.eye(3)
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_floor = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

        mesh.rotate(R_floor, center=(0, 0, 0))
        verts = np.asarray(mesh.vertices)
        floor_z = np.mean(verts[inliers, 2])
    else:
        mesh.rotate(R_floor, center=(0, 0, 0))

    mesh.translate([0, 0, -floor_z])
    return R_floor, floor_z


def gravity_align(mesh, traj_path, sign=1):
    """Gravity-align a mesh using camera poses and RANSAC floor detection.

    sign=1 for OpenCV trajectories (ORB-SLAM), sign=-1 for OpenGL (Replica GT).
    Returns (R_grav, R_floor, floor_z) so trajectory positions can be
    transformed through the same alignment.
    """
    gravity_dir = sign * estimate_gravity_direction(traj_path)
    R_grav = compute_gravity_rotation(gravity_dir)
    mesh.rotate(R_grav, center=(0, 0, 0))
    R_floor, floor_z = flatten_floor(mesh)

    if (R_floor @ np.array([0.0, 0.0, -1.0]))[2] > 0:
        print("  flatten_floor found ceiling — keeping tilt, undoing flip")
        flip_x = np.diag([1.0, -1.0, -1.0])
        mesh.rotate(flip_x, center=(0, 0, 0))
        R_floor = flip_x @ R_floor
        verts = np.asarray(mesh.vertices)
        floor_z = np.percentile(verts[:, 2], 2)
        mesh.translate([0, 0, -floor_z])

    return R_grav, R_floor, floor_z


def transform_positions(positions, R_grav, R_floor, floor_z):
    """Transform trajectory positions through the same gravity alignment."""
    aligned = (R_floor @ R_grav @ positions.T).T
    aligned[:, 2] -= floor_z
    return aligned


def align_pred_to_gt(pred_mesh_path, gt_mesh_path, est_traj_path, gt_traj_path,
                     force=False, flip_gt=False):
    """Independently gravity-align each mesh, then ICP-align pred onto GT.

    Returns (pred_out_path, gt_out_path) of saved aligned meshes.
    Caches results; pass force=True to regenerate.
    """
    pred_out = pred_mesh_path.rsplit(".", 1)[0] + "_gt_aligned.obj"
    gt_out = gt_mesh_path.rsplit(".", 1)[0] + "_aligned.obj"
    if os.path.exists(pred_out) and os.path.exists(gt_out) and not force:
        return pred_out, gt_out

    est_matched, gt_matched = _match_trajectories(est_traj_path, gt_traj_path)
    print(f"  Matched {len(est_matched)} trajectory poses by timestamp")

    pred_mesh = _load_mesh(pred_mesh_path)
    R_gp, R_fp, zp = gravity_align(pred_mesh, est_traj_path)
    est_aligned = transform_positions(est_matched, R_gp, R_fp, zp)

    gt_sign = -1 if flip_gt else 1
    gt_mesh = _load_mesh(gt_mesh_path)
    R_gg, R_fg, zg = gravity_align(gt_mesh, gt_traj_path, sign=gt_sign)
    gt_aligned = transform_positions(gt_matched, R_gg, R_fg, zg)

    T_init = _umeyama_yaw(est_aligned, gt_aligned)
    pred_pc = o3d.geometry.PointCloud(pred_mesh.vertices).voxel_down_sample(0.1)
    gt_pc = o3d.geometry.PointCloud(gt_mesh.vertices).voxel_down_sample(0.1)
    reg = o3d.pipelines.registration.registration_icp(
        pred_pc, gt_pc, 1.0, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    print(f"  Coarse ICP fitness={reg.fitness:.4f}")

    pred_pc = o3d.geometry.PointCloud(pred_mesh.vertices).voxel_down_sample(0.05)
    gt_pc = o3d.geometry.PointCloud(gt_mesh.vertices).voxel_down_sample(0.05)
    reg = o3d.pipelines.registration.registration_icp(
        pred_pc, gt_pc, 0.2, reg.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    print(f"  Fine ICP fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.4f}")
    pred_mesh.transform(reg.transformation)

    for mesh, path in [(gt_mesh, gt_out), (pred_mesh, pred_out)]:
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"  Saved {path}")

    return pred_out, gt_out


# ---------------------------------------------------------------------------
# View-based culling
# ---------------------------------------------------------------------------

def _load_tum_poses(path):
    """Load TUM trajectory as list of 4x4 world-to-camera matrices."""
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals = line.strip().split()
            if len(vals) < 8:
                continue
            t = np.array([float(vals[1]), float(vals[2]), float(vals[3])])
            q = [float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])]
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(np.linalg.inv(T))
    return poses


def _cull_gt_points(gt_points, gt_traj_path, cam_params_path, max_depth=10.0,
                    stride=10):
    """Keep only GT points visible from at least one camera pose."""
    w2c_list = _load_tum_poses(gt_traj_path)
    with open(cam_params_path) as f:
        cam = json.load(f)["camera"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    w, h = cam["w"], cam["h"]

    visible = np.zeros(len(gt_points), dtype=bool)

    for i in range(0, len(w2c_list), stride):
        w2c = w2c_list[i]
        pts_h = np.hstack([gt_points, np.ones((len(gt_points), 1))])
        pts_cam = (w2c @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        valid_depth = (z > 0) & (z < max_depth)
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy
        in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible |= valid_depth & in_frame

    return visible


# ---------------------------------------------------------------------------
# Core distance computation
# ---------------------------------------------------------------------------

def _sample_and_distances(pred_path, gt_path, n_points=N_POINTS_DEFAULT, align=True,
                          est_traj=None, gt_traj=None, cam_params=None,
                          return_clouds=False):
    pred_mesh = _load_mesh(pred_path)
    gt_mesh   = _load_mesh(gt_path)

    pred_mesh.compute_vertex_normals()
    gt_mesh.compute_vertex_normals()
    pred_pc = pred_mesh.sample_points_uniformly(n_points, use_triangle_normal=True)
    gt_pc   = gt_mesh.sample_points_uniformly(n_points, use_triangle_normal=True)

    if align:
        pred_pc, T = _align(pred_pc, gt_pc, est_traj, gt_traj)
        R = T[:3, :3]
        pred_normals = np.asarray(pred_pc.normals) @ R.T
        pred_pc.normals = o3d.utility.Vector3dVector(pred_normals)

    if gt_traj and cam_params:
        gt_pts = np.asarray(gt_pc.points)
        gt_mask = _cull_gt_points(gt_pts, gt_traj, cam_params)
        gt_pc = gt_pc.select_by_index(np.where(gt_mask)[0])
        print(f"View culling GT: kept {int(np.sum(gt_mask))}/{len(gt_mask)}")

        pred_pts = np.asarray(pred_pc.points)
        pred_mask = _cull_gt_points(pred_pts, gt_traj, cam_params)
        pred_pc = pred_pc.select_by_index(np.where(pred_mask)[0])
        print(f"View culling pred: kept {int(np.sum(pred_mask))}/{len(pred_mask)}")

    d_pred_to_gt = np.asarray(pred_pc.compute_point_cloud_distance(gt_pc))
    d_gt_to_pred = np.asarray(gt_pc.compute_point_cloud_distance(pred_pc))

    if return_clouds:
        return d_pred_to_gt, d_gt_to_pred, pred_pc, gt_pc
    return d_pred_to_gt, d_gt_to_pred


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute(pred_path, gt_path, n_points=N_POINTS_DEFAULT, align=True,
            est_traj=None, gt_traj=None, cam_params=None):
    """Chamfer-L2 (backward compatible)."""
    d_pred_to_gt, d_gt_to_pred = _sample_and_distances(
        pred_path, gt_path, n_points, align, est_traj, gt_traj, cam_params)
    return float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))


def compute_all(pred_path, gt_path, n_points=N_POINTS_DEFAULT, tau=TAU_DEFAULT,
                align=True, est_traj=None, gt_traj=None, cam_params=None):
    """All standard geometry metrics. Returns dict."""
    d_pred_to_gt, d_gt_to_pred, pred_pc, gt_pc = _sample_and_distances(
        pred_path, gt_path, n_points, align, est_traj, gt_traj, cam_params,
        return_clouds=True)

    accuracy   = float(np.mean(d_pred_to_gt))
    completion = float(np.mean(d_gt_to_pred))
    chamfer_l1 = accuracy + completion
    chamfer_l2 = float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))

    hausdorff = float(max(np.max(d_pred_to_gt), np.max(d_gt_to_pred)))

    completion_ratio = float(np.mean(d_gt_to_pred < tau)) * 100.0

    precision = float(np.mean(d_pred_to_gt < tau))
    recall    = float(np.mean(d_gt_to_pred < tau))
    f_score   = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f_score  *= 100.0

    pred_normals = np.asarray(pred_pc.normals)
    gt_normals = np.asarray(gt_pc.normals)
    gt_tree = o3d.geometry.KDTreeFlann(gt_pc)
    normal_dots = np.zeros(len(pred_normals))
    for i in range(len(pred_normals)):
        _, idx, _ = gt_tree.search_knn_vector_3d(pred_pc.points[i], 1)
        normal_dots[i] = abs(np.dot(pred_normals[i], gt_normals[idx[0]]))
    normal_consistency = float(np.mean(normal_dots))

    return {
        "accuracy_cm":        accuracy * 100.0,
        "completion_cm":      completion * 100.0,
        "chamfer_l1_cm":      chamfer_l1 * 100.0,
        "chamfer_l2":         chamfer_l2,
        "hausdorff_cm":       hausdorff * 100.0,
        "completion_ratio_%": completion_ratio,
        "precision_%":        precision * 100.0,
        "recall_%":           recall * 100.0,
        "f_score_%":          f_score,
        "normal_consistency": normal_consistency,
        "tau_m":              tau,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_mesh")
    parser.add_argument("gt_mesh")
    parser.add_argument("--n_points", type=int, default=N_POINTS_DEFAULT)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT,
                        help="threshold in meters for completion ratio / F-score (default: 0.05)")
    parser.add_argument("--no_align", action="store_true",
                        help="skip ICP alignment (use if meshes are already in the same frame)")
    parser.add_argument("--est_traj", type=str, default=None,
                        help="estimated trajectory (TUM format) for Umeyama alignment")
    parser.add_argument("--gt_traj", type=str, default=None,
                        help="GT trajectory (TUM format) for Umeyama alignment")
    parser.add_argument("--cam_params", type=str, default=None,
                        help="camera params JSON for view-based GT culling")
    args = parser.parse_args()

    metrics = compute_all(args.pred_mesh, args.gt_mesh, args.n_points, args.tau,
                          align=not args.no_align,
                          est_traj=args.est_traj, gt_traj=args.gt_traj,
                          cam_params=args.cam_params)
    out_path = os.path.join(os.path.dirname(args.pred_mesh), "mesh_metrics.txt")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            line = f"{k}: {v:.6f}"
            f.write(line + "\n")
            print(line)
    print(f"\n→  {out_path}")
