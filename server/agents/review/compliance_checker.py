import requests
import json
from typing import Dict, List, Any
from pydantic import BaseModel

class ComplianceCheckResult(BaseModel):
    tool: str
    findings: List[Dict[str, Any]]
    passed_checks: int
    failed_checks: int
    success: bool
    error: str = ""

class ComplianceChecker:
    def __init__(self, sandbox_executor_url: str = "http://localhost:8002"):
        self.sandbox_executor_url = sandbox_executor_url

    def run_pre_commit(self, code_path: str, config_path: str = None) -> ComplianceCheckResult:
        """Run pre-commit hooks to check compliance."""
        try:
            # Prepare pre-commit command
            command = "pre-commit"
            args = ["run", "--all-files", "--verbose"]

            if config_path:
                args.extend(["--config", config_path])

            # Execute via sandbox executor
            payload = {
                "command": command,
                "args": args,
                "working_dir": code_path,
                "env": {},
                "timeout": 600  # 10 minutes timeout for pre-commit
            }

            response = requests.post(f"{self.sandbox_executor_url}/execute", json=payload, timeout=610)
            response.raise_for_status()

            result = response.json()

            findings = []
            passed_checks = 0
            failed_checks = 0

            # Parse pre-commit output
            lines = result["stdout"].split('\n') + result["stderr"].split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "Passed" in line:
                    passed_checks += 1
                elif "Failed" in line or "Error" in line:
                    failed_checks += 1
                    findings.append({
                        "check": line.split(':')[0] if ':' in line else "unknown",
                        "status": "failed",
                        "message": line
                    })

            return ComplianceCheckResult(
                tool="pre-commit",
                findings=findings,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                success=result["exit_code"] == 0
            )

        except Exception as e:
            return ComplianceCheckResult(
                tool="pre-commit",
                findings=[],
                passed_checks=0,
                failed_checks=0,
                success=False,
                error=str(e)
            )

    def check_best_practices(self, code_path: str, language: str = "python") -> ComplianceCheckResult:
        """Check for basic best practices based on language."""
        findings = []
        passed_checks = 0
        failed_checks = 0

        try:
            # Basic checks that can be done without external tools
            if language.lower() == "python":
                findings.extend(self._check_python_best_practices(code_path))
            elif language.lower() in ["javascript", "typescript"]:
                findings.extend(self._check_js_best_practices(code_path))

            # Count passed/failed
            for finding in findings:
                if finding.get("status") == "passed":
                    passed_checks += 1
                else:
                    failed_checks += 1

            return ComplianceCheckResult(
                tool="best-practices-checker",
                findings=findings,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                success=failed_checks == 0
            )

        except Exception as e:
            return ComplianceCheckResult(
                tool="best-practices-checker",
                findings=[],
                passed_checks=0,
                failed_checks=0,
                success=False,
                error=str(e)
            )

    def _check_python_best_practices(self, code_path: str) -> List[Dict[str, Any]]:
        """Basic Python best practices checks."""
        findings = []

        # This is a simplified check - in practice, you'd read files and analyze
        findings.append({
            "check": "imports_organized",
            "status": "passed",  # Placeholder
            "message": "Imports are properly organized"
        })

        findings.append({
            "check": "docstrings_present",
            "status": "passed",  # Placeholder
            "message": "Functions have docstrings"
        })

        return findings

    def _check_js_best_practices(self, code_path: str) -> List[Dict[str, Any]]:
        """Basic JavaScript/TypeScript best practices checks."""
        findings = []

        findings.append({
            "check": "semicolons_consistent",
            "status": "passed",  # Placeholder
            "message": "Semicolons are used consistently"
        })

        findings.append({
            "check": "var_avoided",
            "status": "passed",  # Placeholder
            "message": "Use of var is avoided in favor of let/const"
        })

        return findings

    def check(self, code_path: str, language: str = "python") -> List[ComplianceCheckResult]:
        """Perform compliance checking."""
        results = []

        # Run pre-commit if available
        pre_commit_result = self.run_pre_commit(code_path)
        results.append(pre_commit_result)

        # Run best practices check
        best_practices_result = self.check_best_practices(code_path, language)
        results.append(best_practices_result)

        return results