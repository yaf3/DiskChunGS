#!/usr/bin/env python3
"""
Generate ablation experiment YAML configs from base configs.
Edit cfg/triangle_mapper/experiments/experiments.yaml to add/modify variants, then run:
    python3 scripts/ablations/generate_configs.py

Output: cfg/triangle_mapper/experiments/{replica,tum}_rgbd_<name>.yaml
"""

import json
import os
import re
import yaml

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR      = os.path.join(REPO_ROOT, "cfg", "triangle_mapper", "experiments")
OUT_DIR      = os.path.join(EXP_DIR, "generated")
EXPERIMENTS_YAML = os.path.join(EXP_DIR, "experiments.yaml")

BASE_CONFIGS = {
    "replica": os.path.join(REPO_ROOT, "cfg/triangle_mapper/RGB-D/Replica/replica_rgbd.yaml"),
    "tum":     os.path.join(REPO_ROOT, "cfg/triangle_mapper/RGB-D/TUM/tum_rgbd.yaml"),
}

with open(EXPERIMENTS_YAML) as f:
    _exp_data = yaml.safe_load(f)

ALWAYS = {str(k): v for k, v in _exp_data.get("always", {}).items()}
EXPERIMENTS = [
    (e["name"], e["description"], {str(k): v for k, v in e["overrides"].items()})
    for g in _exp_data["groups"]
    for e in g["experiments"]
]


def format_val(val):
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == 0.0:
            return "0.0"
        if abs(val) >= 1e-3:
            return f"{val:g}"
        return f"{val:.1e}"
    return str(val)


def apply_overrides(content, overrides):
    lines = content.split("\n")
    applied = set()
    result = []
    for line in lines:
        replaced = False
        for key, val in overrides.items():
            if re.match(r"^" + re.escape(key) + r"\s*:", line):
                result.append(f"{key}: {format_val(val)}")
                applied.add(key)
                replaced = True
                break
        if not replaced:
            result.append(line)
    for key, val in overrides.items():
        if key not in applied:
            result.append(f"{key}: {format_val(val)}")
    return "\n".join(result)


os.makedirs(OUT_DIR, exist_ok=True)

base_contents = {}
for dataset, path in BASE_CONFIGS.items():
    with open(path) as f:
        base_contents[dataset] = f.read()

generated = []
for name, desc, overrides in EXPERIMENTS:
    all_overrides = {**ALWAYS, **overrides}
    for dataset in ("replica", "tum"):
        out_name = f"{dataset}_rgbd_{name}.yaml"
        out_path = os.path.join(OUT_DIR, out_name)
        content = apply_overrides(base_contents[dataset], all_overrides)
        lines = content.split("\n")
        lines.insert(1, f"# {desc}")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        generated.append(out_name)

groups_data = _exp_data["groups"]
manifest = {
    "groups": [
        {
            "name": g["name"],
            "experiments": [
                {"name": f"abl_{e['name']}", "label": e["description"]}
                for e in g["experiments"]
            ],
        }
        for g in groups_data
    ]
}
manifest_path = os.path.join(EXP_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(generated)} configs in {OUT_DIR}/")
for name in generated:
    print(f"  {name}")
print(f"Wrote manifest: {manifest_path}")
