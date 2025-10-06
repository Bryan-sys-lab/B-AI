from pydantic import BaseModel
from typing import Optional, List, Dict
from common.models import ShellExecRequest, ShellExecResponse

class GitReadFileRequest(BaseModel):
    repo_url: str
    file_path: str
    branch: str = "main"

class GitReadFileResponse(BaseModel):
    content: str
    error: Optional[str] = None

class GitWriteFileRequest(BaseModel):
    repo_url: str
    file_path: str
    content: str
    branch: str = "main"
    commit_message: str = "Update file"

class GitWriteFileResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class ListFilesRequest(BaseModel):
    repo_url: str
    path: str = "."
    branch: str = "main"

class ListFilesResponse(BaseModel):
    files: List[str]
    error: Optional[str] = None

class RunTestsRequest(BaseModel):
    repo_url: str
    test_command: str
    branch: str = "main"

class RunTestsResponse(BaseModel):
    output: str
    success: bool
    error: Optional[str] = None


class CreatePrRequest(BaseModel):
    repo_url: str
    title: str
    body: str
    head_branch: str
    base_branch: str = "main"

class CreatePrResponse(BaseModel):
    pr_url: str
    error: Optional[str] = None

class ScanVulnRequest(BaseModel):
    repo_url: str
    branch: str = "main"

class ScanVulnResponse(BaseModel):
    vulnerabilities: List[Dict]
    error: Optional[str] = None

class SearchDocsRequest(BaseModel):
    query: str
    repo_url: Optional[str] = None

class SearchDocsResponse(BaseModel):
    results: List[Dict]
    error: Optional[str] = None

class FetchMetricsRequest(BaseModel):
    repo_url: str
    metric_type: str

class FetchMetricsResponse(BaseModel):
    metrics: Dict
    error: Optional[str] = None

class RunCodeRequest(BaseModel):
    code: str
    language: str
    timeout: int = 30

class RunCodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None