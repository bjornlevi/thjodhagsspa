#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py  – run all extraction scripts in pipeline/

Assumes:
  - project/
      run.py
      pipeline/
         extract_spa_2025_mars_vor_tables.py
         extract_spa_2025_juli_sumar_tables.py
         ...
"""

import os
import sys
import subprocess
import glob

PIPELINE_DIR = "pipeline"

def main():
    if not os.path.isdir(PIPELINE_DIR):
        print(f"[!] Missing folder: {PIPELINE_DIR}")
        sys.exit(1)

    # find all extraction scripts
    scripts = sorted(glob.glob(os.path.join(PIPELINE_DIR, "extract_*.py")))
    if not scripts:
        print("[!] No extract_*.py scripts found in pipeline/")
        sys.exit(0)

    print(f"[+] Found {len(scripts)} extraction scripts")

    for script in scripts:
        name = os.path.basename(script)
        print(f"\n=== Running {name} ===")
        try:
            # run in the same Python interpreter as this script
            result = subprocess.run(
                [sys.executable, script],
                check=False
            )
        except Exception as e:
            print(f"[!] Failed to launch {name}: {e}")
            sys.exit(1)

        if result.returncode != 0:
            print(f"[!] {name} exited with code {result.returncode}")
            # stop at first failure
            sys.exit(result.returncode)

    print("\n[✓] All extraction scripts completed successfully.")

if __name__ == "__main__":
    main()
