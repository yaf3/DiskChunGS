"""Compute bidirectional Chamfer-L2 between a predicted mesh and a GT mesh.

Ch-L2 = mean_{p in P} min_{q in Q} ||p-q||^2  +  mean_{q in Q} min_{p in P} ||q-p||^2

Usage:
    python3 eval/chamfer_l2.py <pred_mesh.off> <gt_mesh.ply> [--n_points 200000]

Importable:
    from chamfer_l2 import compute
    value = compute("mesh.off", "mesh.ply")
"""

import argparse
import os
import struct
import numpy as np
import open3d as o3d

N_POINTS_DEFAULT = 200_000


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
    raise ValueError(f"Unsupported mesh format: {path}")


# ---------------------------------------------------------------------------
# Chamfer-L2
# ---------------------------------------------------------------------------

def compute(pred_path, gt_path, n_points=N_POINTS_DEFAULT):
    pred_mesh = _load_mesh(pred_path)
    gt_mesh   = _load_mesh(gt_path)

    pred_pc = pred_mesh.sample_points_uniformly(n_points)
    gt_pc   = gt_mesh.sample_points_uniformly(n_points)

    d_pred_to_gt = np.asarray(pred_pc.compute_point_cloud_distance(gt_pc))
    d_gt_to_pred = np.asarray(gt_pc.compute_point_cloud_distance(pred_pc))

    return float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_mesh")
    parser.add_argument("gt_mesh")
    parser.add_argument("--n_points", type=int, default=N_POINTS_DEFAULT)
    args = parser.parse_args()

    value = compute(args.pred_mesh, args.gt_mesh, args.n_points)
    out_path = os.path.join(os.path.dirname(args.pred_mesh), "chamfer.txt")
    with open(out_path, "w") as f:
        f.write(f"{value}\n")
    print(f"Ch-L2: {value:.6f}  →  {out_path}")
