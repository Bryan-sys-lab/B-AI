import subprocess
import tempfile
import os
from typing import List
from .config import ARTIFACT_DIR

def capture_artifacts(container_id: str) -> List[str]:
    """
    Capture artifacts from the container's artifact directory.
    Copies files from container:/artifacts to a temporary host directory
    and returns the list of captured file paths.
    """
    artifacts = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy artifacts from container
            subprocess.run([
                "docker", "cp", f"{container_id}:{ARTIFACT_DIR}/.", temp_dir
            ], check=True, capture_output=True)

            # List captured files
            for root, dirs, filenames in os.walk(temp_dir):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), temp_dir)
                    artifacts.append(rel_path)

    except subprocess.CalledProcessError:
        # If copy fails, return empty list
        pass

    return artifacts

def prepare_workspace() -> str:
    """
    Prepare a temporary workspace directory for the execution.
    Returns the path to the workspace directory.
    """
    return tempfile.mkdtemp(prefix="sandbox_workspace_")

def cleanup_workspace(workspace_path: str):
    """
    Clean up the workspace directory after execution.
    """
    import shutil
    try:
        shutil.rmtree(workspace_path)
    except Exception:
        pass  # Ignore cleanup errors