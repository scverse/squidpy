#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

EPILOG = """
Examples:
  python run_notebooks.py docs/notebooks
  python run_notebooks.py /path/to/notebooks --kernel my-kernel
"""


def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run Jupyter notebooks in specified directories using jupytext",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    parser.add_argument("base_directory", help="Base directory containing notebook subdirectories")

    parser.add_argument(
        "-k", "--kernel", default="squidpy", help="Jupyter kernel to use for execution (default: squidpy)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show which notebooks would be run without executing them"
    )

    args = parser.parse_args()

    # Base directory for notebooks
    base_dir = args.base_directory

    # Define notebook directories or patterns
    notebook_patterns = [
        f"{base_dir}/examples/tools/*.ipynb",
        f"{base_dir}/examples/plotting/*.ipynb",
        f"{base_dir}/examples/image/*.ipynb",
        f"{base_dir}/examples/graph/*.ipynb",
        # f"{base_dir}/tutorials/*.ipynb"  # don't include because it contains many external modules
    ]

    # Initialize a list to hold valid notebook paths
    valid_notebooks = []

    # Gather all valid notebook files from the patterns
    print("Gathering notebooks...")
    for pattern in notebook_patterns:
        for nb_path in glob.glob(pattern):
            if os.path.isfile(nb_path):  # Check if the file exists
                valid_notebooks.append(nb_path)  # Add to the list of valid notebooks

    # Check if we have any notebooks to run
    if len(valid_notebooks) == 0:
        print("No notebooks found to run.")
        sys.exit(1)

    # Echo the notebooks that will be run for clarity
    print("Preparing to run the following notebooks:")
    for nb in valid_notebooks:
        print(f"  {nb}")

    # If dry run, exit here
    if args.dry_run:
        print(f"\nDry run complete. Would execute {len(valid_notebooks)} notebooks with kernel '{args.kernel}'.")
        return

    # Initialize a flag to track the success of all commands
    all_success = True

    # Execute all valid notebooks
    print(f"\nExecuting notebooks with kernel '{args.kernel}'...")
    for nb in valid_notebooks:
        print(f"Running {nb}")
        try:
            subprocess.run(["jupytext", "-k", args.kernel, "--execute", nb], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to run {nb}")
            all_success = False

    # Check if any executions failed
    if not all_success:
        print("One or more notebooks failed to execute.")
        sys.exit(1)

    print("All notebooks executed successfully.")


if __name__ == "__main__":
    main()
