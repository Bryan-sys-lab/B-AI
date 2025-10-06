import requests
import json
from typing import Dict, List, Any
from pydantic import BaseModel

class SecurityScanResult(BaseModel):
    tool: str
    findings: List[Dict[str, Any]]
    severity_counts: Dict[str, int]
    success: bool
    error: str = ""

class SecurityScanner:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_executor_url = sandbox_executor_url

    def run_bandit(self, code_path: str) -> SecurityScanResult:
        """Run Bandit security scanner on Python code."""
        try:
            # Prepare bandit command
            command = "bandit"
            args = ["-f", "json", "-r", code_path]

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

            if result["exit_code"] == 0 or result["exit_code"] == 1:  # Bandit returns 1 when findings are found
                # Parse bandit JSON output
                try:
                    bandit_output = json.loads(result["stdout"])
                    findings = bandit_output.get("results", [])
                    severity_counts = self._count_severities(findings)
                    return SecurityScanResult(
                        tool="bandit",
                        findings=findings,
                        severity_counts=severity_counts,
                        success=True
                    )
                except json.JSONDecodeError:
                    return SecurityScanResult(
                        tool="bandit",
                        findings=[],
                        severity_counts={},
                        success=False,
                        error="Failed to parse bandit output"
                    )
            else:
                return SecurityScanResult(
                    tool="bandit",
                    findings=[],
                    severity_counts={},
                    success=False,
                    error=result["stderr"]
                )

        except Exception as e:
            return SecurityScanResult(
                tool="bandit",
                findings=[],
                severity_counts={},
                success=False,
                error=str(e)
            )

    def run_checkov(self, code_path: str, framework: str = "terraform") -> SecurityScanResult:
        """Run Checkov security scanner on infrastructure code."""
        try:
            # Prepare checkov command
            command = "checkov"
            args = ["-f", code_path, "--framework", framework, "-o", "json"]

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

            if result["exit_code"] == 0 or result["exit_code"] == 1:  # Checkov returns 1 when findings are found
                # Parse checkov JSON output
                try:
                    checkov_output = json.loads(result["stdout"])
                    findings = checkov_output.get("results", {}).get("failed_checks", [])
                    severity_counts = self._count_checkov_severities(findings)
                    return SecurityScanResult(
                        tool="checkov",
                        findings=findings,
                        severity_counts=severity_counts,
                        success=True
                    )
                except json.JSONDecodeError:
                    return SecurityScanResult(
                        tool="checkov",
                        findings=[],
                        severity_counts={},
                        success=False,
                        error="Failed to parse checkov output"
                    )
            else:
                return SecurityScanResult(
                    tool="checkov",
                    findings=[],
                    severity_counts={},
                    success=False,
                    error=result["stderr"]
                )

        except Exception as e:
            return SecurityScanResult(
                tool="checkov",
                findings=[],
                severity_counts={},
                success=False,
                error=str(e)
            )

    def _count_severities(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count bandit findings by severity level."""
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for finding in findings:
            severity = finding.get("issue_severity", "LOW").upper()
            if severity in counts:
                counts[severity] += 1
            else:
                counts["LOW"] += 1  # Default to LOW for unknown severities
        return counts

    def _count_checkov_severities(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count checkov findings by severity level."""
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for finding in findings:
            severity = finding.get("check_result", {}).get("severity", "LOW").upper()
            if severity in counts:
                counts[severity] += 1
            else:
                counts["LOW"] += 1  # Default to LOW for unknown severities
        return counts

    def scan(self, code_path: str, language: str = "python") -> List[SecurityScanResult]:
        """Perform security scanning based on code language."""
        results = []

        if language.lower() == "python":
            results.append(self.run_bandit(code_path))
        elif language.lower() in ["terraform", "cloudformation", "kubernetes"]:
            results.append(self.run_checkov(code_path, language.lower()))

        return results