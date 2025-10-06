"""Common Pydantic models shared across services."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class ShellExecRequest(BaseModel):
    command: str
    args: List[str] = []
    working_dir: str = "/"
    env: Dict[str, str] = {}
    timeout: int = 30


class ShellExecResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    artifacts: List[str] = []
    error: Optional[str] = None