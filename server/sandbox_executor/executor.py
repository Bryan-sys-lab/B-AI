from fastapi import FastAPI
import subprocess
import os
import tempfile
import shutil
from typing import List, Dict, Optional
from pydantic import BaseModel
try:
    # Prefer package-qualified imports when running tests (PYTHONPATH=$(pwd))
    from sandbox_executor.config import (
        CPU_SHARES,
        MEMORY,
        MEMORY_SWAP,
        CPU_PERIOD,
        CPU_QUOTA,
        NETWORK_MODE,
        SECCOMP_PROFILE,
        BASE_IMAGE,
        DEFAULT_TIMEOUT,
        MAX_TIMEOUT,
        ARTIFACT_DIR,
    )
    from sandbox_executor.utils import prepare_workspace, cleanup_workspace
except Exception:
    # When running inside container as /app/executor.py the package name
    # may not be available; fall back to local module imports.
    from config import (
        CPU_SHARES,
        MEMORY,
        MEMORY_SWAP,
        CPU_PERIOD,
        CPU_QUOTA,
        NETWORK_MODE,
        SECCOMP_PROFILE,
        BASE_IMAGE,
        DEFAULT_TIMEOUT,
        MAX_TIMEOUT,
        ARTIFACT_DIR,
    )
    from utils import prepare_workspace, cleanup_workspace
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES
from common.utils import is_running_in_container
from common.models import ShellExecRequest, ShellExecResponse

app = FastAPI(title="Sandbox Executor")

@app.get("/health")
def health():
    return {"status": "ok"}

def execute_locally(command: str, args: List[str], working_dir: str, env: Dict[str, str], timeout: int) -> ShellExecResponse:
    """
    Execute a command directly on the local system with artifact capture.
    """
    workspace = prepare_workspace()
    artifacts = []

    try:
        # Create artifacts directory in workspace
        artifacts_dir = os.path.join(workspace, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Prepare environment variables
        env_vars = os.environ.copy()
        env_vars.update(env)
        env_vars['ARTIFACTS_DIR'] = artifacts_dir  # Make artifacts dir available to commands

        # Build the full command
        full_command = [command] + args

        # Execute with timeout
        try:
            result = subprocess.run(
                full_command,
                cwd=working_dir or workspace,
                env=env_vars,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            # Capture artifacts from the artifacts directory
            if os.path.exists(artifacts_dir):
                for root, dirs, filenames in os.walk(artifacts_dir):
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), artifacts_dir)
                        artifacts.append(rel_path)

                        # Copy artifact to generated directory if it exists
                        generated_dir = os.path.join(os.getcwd(), "generated")
                        os.makedirs(generated_dir, exist_ok=True)
                        shutil.copy2(os.path.join(root, filename), os.path.join(generated_dir, filename))

        except subprocess.TimeoutExpired:
            return ShellExecResponse(stdout="", stderr="Command timed out", exit_code=1, artifacts=[], error="Timeout")

    except Exception as e:
        return ShellExecResponse(stdout="", stderr="", exit_code=1, artifacts=[], error=str(e))
    finally:
        cleanup_workspace(workspace)

    return ShellExecResponse(stdout=stdout, stderr=stderr, exit_code=exit_code, artifacts=artifacts)

def execute_in_sandbox(command: str, args: List[str], working_dir: str, env: Dict[str, str], timeout: int) -> ShellExecResponse:
    """
    Execute a command either in a Docker sandbox or locally, depending on the environment.
    """
    if is_running_in_container():
        # Execute in Docker sandbox with resource limits and security controls
        workspace = prepare_workspace()
        try:
            # Prepare environment variables
            env_vars = os.environ.copy()
            env_vars.update(env)

            # Build docker run command
            docker_cmd = [
                "docker", "run", "--rm",
                "--cpu-shares", str(CPU_SHARES),
                "--memory", MEMORY,
                "--memory-swap", MEMORY_SWAP,
                "--cpu-period", str(CPU_PERIOD),
                "--cpu-quota", str(CPU_QUOTA),
                "--network", NETWORK_MODE,
                "--read-only",
                "--tmpfs", "/tmp",
                "--tmpfs", "/var/tmp",
                "--cap-drop", "all",
                "--security-opt", "no-new-privileges",
                "-v", f"{workspace}:/workspace",
                "-w", working_dir,
            ]

            # Add environment variables
            for key, value in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

            # Add the base image
            docker_cmd.append(BASE_IMAGE)

            # Add the command
            full_command = [command] + args
            docker_cmd.extend(["sh", "-c", " ".join(full_command)])

            # Execute with timeout
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode
                artifacts = []  # For now, no artifacts
            except subprocess.TimeoutExpired:
                return ShellExecResponse(stdout="", stderr="Command timed out", exit_code=1, artifacts=[], error="Timeout")

        except Exception as e:
            return ShellExecResponse(stdout="", stderr="", exit_code=1, artifacts=[], error=str(e))
        finally:
            cleanup_workspace(workspace)

        return ShellExecResponse(stdout=stdout, stderr=stderr, exit_code=exit_code, artifacts=artifacts)
    else:
        # Execute locally with artifact capture
        return execute_locally(command, args, working_dir, env, timeout)

@app.post("/execute", response_model=ShellExecResponse)
def execute(request: ShellExecRequest):
    # Validate timeout
    if request.timeout > MAX_TIMEOUT:
        request.timeout = MAX_TIMEOUT

    result = execute_in_sandbox(
        request.command,
        request.args,
        request.working_dir,
        request.env,
        request.timeout
    )
    return result


@app.get("/about")
def about(detail: str = "short"):
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}
    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {"level": level, "response": resp}