import requests
import json
from typing import Dict, List, Any
from pydantic import BaseModel

class StyleCheckResult(BaseModel):
    tool: str
    findings: List[Dict[str, Any]]
    error_count: int
    warning_count: int
    success: bool
    error: str = ""

class StyleChecker:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_executor_url = sandbox_executor_url

    def run_eslint(self, code_path: str, config_path: str = None) -> StyleCheckResult:
        """Run ESLint on JavaScript/TypeScript code."""
        try:
            # Prepare eslint command
            command = "eslint"
            args = ["--format", "json"]

            if config_path:
                args.extend(["--config", config_path])

            args.append(code_path)

            # Execute via sandbox executor
            payload = {
                "command": command,
                "args": args,
                "working_dir": "/workspace",
                "env": {},
                "timeout": 300  # 5 minutes timeout
            }

            response = requests.post(f"{self.sandbox_executor_url}/execute", json=payload, timeout=310)
            response.raise_for_status()

            result = response.json()

            if result["exit_code"] == 0 or result["exit_code"] == 1:  # ESLint returns 1 when linting errors are found
                # Parse ESLint JSON output
                try:
                    eslint_output = json.loads(result["stdout"])
                    findings = []
                    error_count = 0
                    warning_count = 0

                    for file_result in eslint_output:
                        for message in file_result.get("messages", []):
                            findings.append({
                                "file": file_result.get("filePath", ""),
                                "line": message.get("line", 0),
                                "column": message.get("column", 0),
                                "rule": message.get("ruleId", ""),
                                "severity": message.get("severity", 1),  # 1=warning, 2=error
                                "message": message.get("message", "")
                            })

                            if message.get("severity") == 2:
                                error_count += 1
                            elif message.get("severity") == 1:
                                warning_count += 1

                    return StyleCheckResult(
                        tool="eslint",
                        findings=findings,
                        error_count=error_count,
                        warning_count=warning_count,
                        success=True
                    )
                except json.JSONDecodeError:
                    return StyleCheckResult(
                        tool="eslint",
                        findings=[],
                        error_count=0,
                        warning_count=0,
                        success=False,
                        error="Failed to parse ESLint output"
                    )
            else:
                return StyleCheckResult(
                    tool="eslint",
                    findings=[],
                    error_count=0,
                    warning_count=0,
                    success=False,
                    error=result["stderr"]
                )

        except Exception as e:
            return StyleCheckResult(
                tool="eslint",
                findings=[],
                error_count=0,
                warning_count=0,
                success=False,
                error=str(e)
            )

    def run_prettier(self, code_path: str, check_only: bool = True) -> StyleCheckResult:
        """Run Prettier to check code formatting."""
        try:
            # Prepare prettier command
            command = "prettier"
            args = ["--check"]

            if not check_only:
                args = ["--write"]

            args.append(code_path)

            # Execute via sandbox executor
            payload = {
                "command": command,
                "args": args,
                "working_dir": "/workspace",
                "env": {},
                "timeout": 300  # 5 minutes timeout
            }

            response = requests.post(f"{self.sandbox_executor_url}/execute", json=payload, timeout=310)
            response.raise_for_status()

            result = response.json()

            findings = []
            if result["exit_code"] != 0:
                # Prettier found formatting issues
                lines = result["stderr"].split('\n')
                for line in lines:
                    if line.strip():
                        findings.append({
                            "file": code_path,
                            "message": line.strip(),
                            "type": "formatting"
                        })

            return StyleCheckResult(
                tool="prettier",
                findings=findings,
                error_count=len(findings) if result["exit_code"] != 0 else 0,
                warning_count=0,
                success=result["exit_code"] == 0
            )

        except Exception as e:
            return StyleCheckResult(
                tool="prettier",
                findings=[],
                error_count=0,
                warning_count=0,
                success=False,
                error=str(e)
            )

    def check(self, code_path: str, language: str = "javascript") -> List[StyleCheckResult]:
        """Perform style checking based on code language."""
        results = []

        if language.lower() in ["javascript", "typescript", "js", "ts"]:
            results.append(self.run_eslint(code_path))
            results.append(self.run_prettier(code_path))

        return results