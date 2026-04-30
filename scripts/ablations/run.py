#!/usr/bin/env python3
"""
Run ablation experiments (1 trial, scene subsets for speed).
Usage: python3 scripts/ablations/run.py [<group>|all] [--data DATA_ROOT]

Groups and experiments are driven by cfg/triangle_mapper/experiments/experiments.yaml.
Replica subset: office3, office4, room1  (~10 min/variant)
TUM subset:     fr1_desk, fr3_office     (~2.5 min/variant)

Evaluate with: ./scripts/ablations/eval.sh
"""

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR       = os.path.join(REPO_ROOT, "cfg", "triangle_mapper", "experiments")
GENERATED_DIR = os.path.join(EXP_DIR, "generated")
MANIFEST      = os.path.join(EXP_DIR, "manifest.json")


def generate_configs():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "ablations", "generate_configs.py")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"Config generation failed:\n{result.stderr}")
    n = sum(1 for line in result.stdout.splitlines() if line.strip().endswith(".yaml"))
    print(f"Generated {n} configs from experiments.yaml")


def run_one(exp_name, dataset, cfg_path, data_root):
    log_dir  = os.path.join(REPO_ROOT, "results", exp_name)
    log_path = os.path.join(log_dir, f"{dataset}_run.log")
    script   = os.path.join(REPO_ROOT, "scripts", "ablations", f"{dataset}_subset.sh")
    os.makedirs(log_dir, exist_ok=True)

    print(f"[{exp_name}  {dataset}]  running...  (log: results/{exp_name}/{dataset}_run.log)")
    t0 = time.time()

    with open(log_path, "w") as log:
        result = subprocess.run(
            ["bash", script, exp_name, "1", cfg_path, data_root],
            stdout=log, stderr=log, cwd=REPO_ROOT
        )

    elapsed = int(time.time() - t0)
    if result.returncode != 0:
        print(f"[{exp_name}  {dataset}]  FAILED after {elapsed}s — last lines of log:", file=sys.stderr)
        with open(log_path) as f:
            print("".join(f.readlines()[-20:]), file=sys.stderr)
        sys.exit(1)

    print(f"[{exp_name}  {dataset}]  done in {elapsed}s")


def run_group(group, data_root):
    print(f"\n=== {group['name']} ablation ===")
    for exp in group["experiments"]:
        raw = exp["name"][len("abl_"):]
        run_one(exp["name"], "replica", os.path.join(GENERATED_DIR, f"replica_rgbd_{raw}.yaml"), data_root)
        run_one(exp["name"], "tum",     os.path.join(GENERATED_DIR, f"tum_rgbd_{raw}.yaml"),     data_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", nargs="?", default="all", help="Group to run, or 'all' (default)")
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "data"), help="Data root directory")
    args = parser.parse_args()

    generate_configs()
    atexit.register(shutil.rmtree, GENERATED_DIR, True)

    with open(MANIFEST) as f:
        groups = json.load(f)["groups"]

    group_names = [g["name"] for g in groups]

    if args.group == "all":
        for g in groups:
            run_group(g, args.data)
    elif args.group in group_names:
        run_group(next(g for g in groups if g["name"] == args.group), args.data)
    else:
        sys.exit(f"Unknown group '{args.group}'. Available: all {' '.join(group_names)}")

    print("\n=== Done. Run ./scripts/ablations/eval.sh to evaluate. ===")


if __name__ == "__main__":
    main()
