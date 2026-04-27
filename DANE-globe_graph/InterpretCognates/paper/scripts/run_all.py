#!/usr/bin/env python3
"""Run the complete analysis pipeline: figures then stats."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(name):
    path = os.path.join(SCRIPT_DIR, name)
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] {name} exited with code {result.returncode}")
        return False
    return True


def main():
    ok = True
    ok = run_script('precompute_paper_data.py') and ok
    ok = run_script('generate_figures.py') and ok
    ok = run_script('generate_stats.py') and ok

    print(f"\n{'='*60}")
    if ok:
        print("All scripts completed successfully.")
    else:
        print("Some scripts had errors — check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
