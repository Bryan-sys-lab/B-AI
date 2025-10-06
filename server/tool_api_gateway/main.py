from fastapi import FastAPI, HTTPException
from .models import (
    GitReadFileRequest,
    GitReadFileResponse,
    GitWriteFileRequest,
    GitWriteFileResponse,
    ListFilesRequest,
    ListFilesResponse,
    RunTestsRequest,
    RunTestsResponse,
    ShellExecRequest,
    ShellExecResponse,
    CreatePrRequest,
    CreatePrResponse,
    ScanVulnRequest,
    ScanVulnResponse,
    SearchDocsRequest,
    SearchDocsResponse,
    FetchMetricsRequest,
    FetchMetricsResponse,
    RunCodeRequest,
    RunCodeResponse,
)
from .security import validate_request
import requests
import subprocess
import tempfile
import os
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

app = FastAPI(title="Tool API Gateway")

@app.get("/health")
def health():
    return {"status": "ok"}

SANDBOX_EXECUTOR_URL = os.environ.get("SANDBOX_EXECUTOR_URL", "http://localhost:8002")

@app.post("/git_read_file", response_model=GitReadFileResponse)
def git_read_file(request: GitReadFileRequest):
    if not validate_request("git_read_file", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repo
            subprocess.run(["git", "clone", "--branch", request.branch, "--depth", "1", request.repo_url, temp_dir], check=True, capture_output=True)
            file_path = os.path.join(temp_dir, request.file_path)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                return GitReadFileResponse(content=content)
            else:
                return GitReadFileResponse(content="", error="File not found")
    except subprocess.CalledProcessError as e:
        return GitReadFileResponse(content="", error=str(e))
    except Exception as e:
        return GitReadFileResponse(content="", error=str(e))

@app.post("/git_write_file", response_model=GitWriteFileResponse)
def git_write_file(request: GitWriteFileRequest):
    if not validate_request("git_write_file", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", request.repo_url, temp_dir], check=True, capture_output=True)
            subprocess.run(["git", "checkout", request.branch], cwd=temp_dir, check=True)
            file_path = os.path.join(temp_dir, request.file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(request.content)
            subprocess.run(["git", "add", request.file_path], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", request.commit_message], cwd=temp_dir, check=True)
            subprocess.run(["git", "push"], cwd=temp_dir, check=True)
            return GitWriteFileResponse(success=True)
    except subprocess.CalledProcessError as e:
        return GitWriteFileResponse(success=False, error=str(e))
    except Exception as e:
        return GitWriteFileResponse(success=False, error=str(e))

@app.post("/list_files", response_model=ListFilesResponse)
def list_files(request: ListFilesRequest):
    if not validate_request("list_files", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", "--branch", request.branch, "--depth", "1", request.repo_url, temp_dir], check=True, capture_output=True)
            path = os.path.join(temp_dir, request.path)
            if os.path.exists(path):
                files = []
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        files.append(os.path.relpath(os.path.join(root, filename), path))
                return ListFilesResponse(files=files)
            else:
                return ListFilesResponse(files=[], error="Path not found")
    except subprocess.CalledProcessError as e:
        return ListFilesResponse(files=[], error=str(e))
    except Exception as e:
        return ListFilesResponse(files=[], error=str(e))

@app.post("/run_tests", response_model=RunTestsResponse)
def run_tests(request: RunTestsRequest):
    if not validate_request("run_tests", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        # Use sandbox for running tests
        exec_request = ShellExecRequest(command=request.test_command, working_dir="/workspace", timeout=300)
        response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=exec_request.model_dump())
        if response.status_code == 200:
            result = response.json()
            success = result.get("exit_code", 1) == 0
            return RunTestsResponse(output=result.get("stdout", ""), success=success)
        else:
            return RunTestsResponse(output="", success=False, error="Sandbox execution failed")
    except Exception as e:
        return RunTestsResponse(output="", success=False, error=str(e))

@app.post("/shell_exec", response_model=ShellExecResponse)
def shell_exec(request: ShellExecRequest):
    if not validate_request("shell_exec", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=request.model_dump())
        if response.status_code == 200:
            return ShellExecResponse(**response.json())
        else:
            return ShellExecResponse(stdout="", stderr="", exit_code=1, error="Sandbox execution failed")
    except Exception as e:
        return ShellExecResponse(stdout="", stderr="", exit_code=1, error=str(e))

@app.post("/run_code", response_model=RunCodeResponse)
def run_code(request: RunCodeRequest):
    if not validate_request("run_code", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        # Map language to command
        lang_commands = {
            "python": ("python3", "code.py"),
            "javascript": ("node", "code.js"),
            "bash": ("bash", "code.sh"),
        }
        if request.language not in lang_commands:
            return RunCodeResponse(stdout="", stderr="", exit_code=1, error=f"Unsupported language: {request.language}")

        command, filename = lang_commands[request.language]

        # Create shell exec request
        exec_request = ShellExecRequest(
            command=command,
            args=[filename],
            working_dir="/workspace",
            env={"CODE": request.code},  # Pass code via env if needed, but actually write to file
            timeout=request.timeout
        )

        # For simplicity, use a fixed code writing approach
        # Write code to file first
        write_code_request = ShellExecRequest(
            command="sh",
            args=["-c", f"echo '{request.code.replace("'", "\\'")}' > /workspace/{filename}"],
            working_dir="/workspace",
            timeout=10
        )
        write_response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=write_code_request.model_dump())
        if write_response.status_code != 200 or write_response.json().get("exit_code") != 0:
            return RunCodeResponse(stdout="", stderr="", exit_code=1, error="Failed to write code")

        # Then execute
        response = requests.post(f"{SANDBOX_EXECUTOR_URL}/execute", json=exec_request.model_dump())
        if response.status_code == 200:
            result = response.json()
            return RunCodeResponse(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                exit_code=result.get("exit_code", 0),
                error=None
            )
        else:
            return RunCodeResponse(stdout="", stderr="", exit_code=1, error="Sandbox execution failed")
    except Exception as e:
        return RunCodeResponse(stdout="", stderr="", exit_code=1, error=str(e))

@app.post("/create_pr", response_model=CreatePrResponse)
def create_pr(request: CreatePrRequest):
    if not validate_request("create_pr", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    # Mock implementation - in real, use GitHub API or similar
    return CreatePrResponse(pr_url=f"https://github.com/{request.repo_url.split('/')[-2]}/{request.repo_url.split('/')[-1]}/pulls/1", error=None)

@app.post("/scan_vuln", response_model=ScanVulnResponse)
def scan_vuln(request: ScanVulnRequest):
    if not validate_request("scan_vuln", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    # Mock implementation
    return ScanVulnResponse(vulnerabilities=[], error=None)

@app.post("/search_docs", response_model=SearchDocsResponse)
def search_docs(request: SearchDocsRequest):
    if not validate_request("search_docs", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    # Mock implementation
    return SearchDocsResponse(results=[], error=None)

@app.post("/fetch_metrics", response_model=FetchMetricsResponse)
def fetch_metrics(request: FetchMetricsRequest):
    if not validate_request("fetch_metrics", request.model_dump()):
        raise HTTPException(status_code=403, detail="Access denied")
    # Mock implementation
    return FetchMetricsResponse(metrics={}, error=None)


@app.get("/about")
def about(detail: str = "short"):
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}
    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {"level": level, "response": resp}