"""
run_all.py
==========
Master script to run the full Cross-Channel Mule Detection pipeline.
Executes all steps in sequence with proper error handling.

Usage:
    python scripts/run_all.py                    # full pipeline
    python scripts/run_all.py --skip-enrichment  # skip data enrichment
    python scripts/run_all.py --skip-graph       # skip graph building
"""

import os
import sys
import subprocess
import time
import argparse

SCRIPTS_DIR = os.path.dirname(__file__)

def run_step(name, cmd, cwd=None):
    """Run a pipeline step and report status."""
    print(f"\n{'#'*70}")
    print(f"# STEP: {name}")
    print(f"# CMD: {cmd}")
    print(f"{'#'*70}\n")

    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd or os.path.join(SCRIPTS_DIR, ".."))
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"  [ERROR] Exit code: {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=100)
    args = parser.parse_args()

    start_total = time.time()
    steps_ok = 0
    steps_total = 0

    # Step 1: Enrichment
    if not args.skip_enrichment:
        steps_total += 1
        if run_step("Cross-Channel Enrichment",
                     "python scripts/generate_cross_channel.py"):
            steps_ok += 1

    # Step 2: Graph building
    if not args.skip_graph:
        steps_total += 1
        if run_step("Build Entity Graph (train)",
                     "python scripts/build_graph.py --split train"):
            steps_ok += 1

        steps_total += 1
        if run_step("Build Entity Graph (val)",
                     "python scripts/build_graph.py --split val"):
            steps_ok += 1

        steps_total += 1
        if run_step("Build Entity Graph (test)",
                     "python scripts/build_graph.py --split test"):
            steps_ok += 1

    # Step 3: Baseline models
    if not args.skip_baseline:
        steps_total += 1
        if run_step("Train Baseline Models",
                     "python scripts/train_baseline.py"):
            steps_ok += 1

    # Step 4: GNN models
    if not args.skip_gnn:
        steps_total += 1
        if run_step("Train GNN Models",
                     f"python scripts/train_gnn.py --epochs {args.gnn_epochs}"):
            steps_ok += 1

    # Step 5: Streaming demo
    steps_total += 1
    if run_step("Streaming Demo",
                 "python scripts/streaming_demo.py --limit 500"):
        steps_ok += 1

    # Step 6: Explainability
    steps_total += 1
    if run_step("Explainability Reports",
                 "python scripts/explainability.py --top-k 20"):
        steps_ok += 1

    elapsed_total = time.time() - start_total
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE: {steps_ok}/{steps_total} steps succeeded in {elapsed_total:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
