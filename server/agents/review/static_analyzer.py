import requests
import json
from typing import Dict, List, Any
from pydantic import BaseModel

class StaticAnalysisResult(BaseModel):
    tool: str
    findings: List[Dict[str, Any]]
    severity_counts: Dict[str, int]
    success: bool
    error: str = ""

class StaticAnalyzer:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_executor_url = sandbox_executor_url

    def run_semgrep(self, code_path: str, rules_path: str = None) -> StaticAnalysisResult:
        """Run Semgrep static analysis on the given code path."""
        try:
            # Prepare semgrep command
            command = "semgrep"
            args = ["--json"]

            if rules_path:
                args.extend(["--config", rules_path])
            else:
                # Use default rules
                args.append("--config=auto")

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

            if result["exit_code"] == 0:
                # Parse semgrep JSON output
                try:
                    semgrep_output = json.loads(result["stdout"])
                    findings = semgrep_output.get("results", [])
                    severity_counts = self._count_severities(findings)
                    return StaticAnalysisResult(
                        tool="semgrep",
                        findings=findings,
                        severity_counts=severity_counts,
                        success=True
                    )
                except json.JSONDecodeError:
                    return StaticAnalysisResult(
                        tool="semgrep",
                        findings=[],
                        severity_counts={},
                        success=False,
                        error="Failed to parse semgrep output"
                    )
            else:
                return StaticAnalysisResult(
                    tool="semgrep",
                    findings=[],
                    severity_counts={},
                    success=False,
                    error=result["stderr"]
                )

        except Exception as e:
            return StaticAnalysisResult(
                tool="semgrep",
                findings=[],
                severity_counts={},
                success=False,
                error=str(e)
            )

    def _count_severities(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count findings by severity level."""
        counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        for finding in findings:
            severity = finding.get("extra", {}).get("severity", "INFO").upper()
            if severity in counts:
                counts[severity] += 1
            else:
                counts["INFO"] += 1  # Default to INFO for unknown severities
        return counts

    def analyze(self, code_path: str, rules_path: str = None) -> StaticAnalysisResult:
        """Perform static analysis using semgrep."""
        return self.run_semgrep(code_path, rules_path)