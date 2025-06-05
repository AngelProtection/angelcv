#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run_command(cmd, cwd=None, env=None, check=True):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True, capture_output=True)  # noqa: S603
    return result


def create_test_script(venv_path):
    """Create a test script to verify the package works."""
    test_script = venv_path / "test_import.py"
    with open(test_script, "w") as f:
        f.write("""
import torch
import numpy as np
from angelcv.utils.env_utils import is_debug_mode
from angelcv.utils.block_utils import get_block_name_to_impl_dict, create_activation_function
from angelcv.utils.logging_manager import get_logger

# Test basic functionality
print("Testing AngelCV package...")

# Test logger
logger = get_logger("TestLogger")
logger.info("Logger initialized successfully")

# Test activation function creation
relu = create_activation_function("ReLU")
print(f"Created activation: {relu}")

# Test block utils
block_map = get_block_name_to_impl_dict()
print(f"Found {len(block_map)} block implementations")

# Test debug mode detection
debug_mode = is_debug_mode()
print(f"Debug mode: {debug_mode}")

print("All basic tests passed!")
""")
    return test_script


def main():
    # Store the workspace directory (being executed from there)
    workspace_dir = Path.cwd()

    # Create a temporary directory for our test
    temp_dir = Path(tempfile.gettempdir()) / "test_build"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    dist_path = workspace_dir / "dist"

    # Step 1: Build the package
    print("\n=== Building package ===")
    # Clean up previous build artifacts
    if dist_path.exists():
        print(f"Removing {dist_path}")
        shutil.rmtree(dist_path)

    # Also remove the .egg-info directory
    egg_info_dirs = list(workspace_dir.glob("*.egg-info"))
    for egg_info_dir in egg_info_dirs:
        print(f"Removing {egg_info_dir}")
        shutil.rmtree(egg_info_dir)

    # Try with uv first, fall back to pip if not available
    result = run_command(["uv", "build", "."], cwd=workspace_dir)
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    # Check if build was successful
    wheel_files = list(dist_path.glob("*.whl"))
    if not wheel_files:
        print("No wheel file found in dist directory")
        return 1

    wheel_path = wheel_files[0]
    print(f"Built wheel: {wheel_path}")

    # Step 2: Create a virtual environment using uv
    print("\n=== Creating virtual environment with uv ===")
    venv_path = temp_dir / ".venv"
    # Create venv with uv
    result = run_command(["uv", "venv", "--python", "3.12"], cwd=temp_dir)
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    # Get the activate script path
    activate_script = venv_path / "bin" / "activate"

    # Step 3: Install the wheel in the virtual environment using uv with activated environment
    print("\n=== Installing wheel in virtual environment ===")
    install_cmd = f"source {activate_script} && uv pip install {wheel_path}"
    result = subprocess.run(
        install_cmd, shell=True, cwd=temp_dir, check=True, text=True, capture_output=True, executable="/bin/bash"
    )
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    # Step 4: Create and run a test script
    print("\n=== Testing package functionality ===")
    test_script = create_test_script(temp_dir)
    run_cmd = f"source {activate_script} && python {test_script}"
    result = subprocess.run(
        run_cmd, shell=True, cwd=temp_dir, check=True, text=True, capture_output=True, executable="/bin/bash"
    )
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    print("\n=== All tests completed successfully! ===")

    """
    # Clean up
    shutil.rmtree(temp_dir)
    """
    return 0


if __name__ == "__main__":
    sys.exit(main())
