import subprocess
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

class DependencyVulnerability(BaseModel):
    file: str
    dependency: str
    vulnerability_id: str
    severity: str
    description: str
    cvss_score: Optional[float] = None
    cwe: Optional[str] = None
    references: List[str] = []

class DependencyScanResult(BaseModel):
    tool: str
    vulnerabilities: List[DependencyVulnerability]
    summary: Dict[str, Union[int, str]]

class DependencyChecker:
    def __init__(self):
        self.tool_path = "dependency-check"  # Assume it's in PATH or full path

    def scan_dependencies(self, path: str) -> DependencyScanResult:
        """Scan dependencies using OWASP Dependency-Check"""
        try:
            # Run dependency-check
            cmd = [
                self.tool_path,
                "--project", "SecurityScan",
                "--scan", path,
                "--format", "JSON",
                "--out", "/tmp/dependency-check-report.json"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            vulnerabilities = []
            if result.returncode == 0:
                # Parse the JSON report
                report_path = "/tmp/dependency-check-report.json"
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        data = json.load(f)

                    for dependency in data.get('dependencies', []):
                        for vuln in dependency.get('vulnerabilities', []):
                            vulnerabilities.append(DependencyVulnerability(
                                file=dependency.get('filePath', ''),
                                dependency=dependency.get('fileName', ''),
                                vulnerability_id=vuln.get('name', ''),
                                severity=vuln.get('severity', 'unknown'),
                                description=vuln.get('description', ''),
                                cvss_score=vuln.get('cvssv3', {}).get('baseScore'),
                                cwe=vuln.get('cwe', ''),
                                references=vuln.get('references', [])
                            ))

            # Clean up
            if os.path.exists("/tmp/dependency-check-report.json"):
                os.remove("/tmp/dependency-check-report.json")

            summary = {
                'total': len(vulnerabilities),
                'high': len([v for v in vulnerabilities if v.severity.upper() == 'HIGH']),
                'medium': len([v for v in vulnerabilities if v.severity.upper() == 'MEDIUM']),
                'low': len([v for v in vulnerabilities if v.severity.upper() == 'LOW'])
            }

            return DependencyScanResult(
                tool='dependency-check',
                vulnerabilities=vulnerabilities,
                summary=summary
            )

        except subprocess.TimeoutExpired:
            return DependencyScanResult(
                tool='dependency-check',
                vulnerabilities=[],
                summary={'error': 'timeout'}
            )
        except Exception as e:
            return DependencyScanResult(
                tool='dependency-check',
                vulnerabilities=[],
                summary={'error': str(e)}
            )

    def parse_package_json(self, path: str) -> List[str]:
        """Extract dependencies from package.json"""
        package_json = os.path.join(path, 'package.json')
        if not os.path.exists(package_json):
            return []

        try:
            with open(package_json, 'r') as f:
                data = json.load(f)

            deps = []
            for section in ['dependencies', 'devDependencies']:
                if section in data:
                    deps.extend(list(data[section].keys()))
            return deps
        except:
            return []

    def parse_requirements_txt(self, path: str) -> List[str]:
        """Extract dependencies from requirements.txt"""
        req_file = os.path.join(path, 'requirements.txt')
        if not os.path.exists(req_file):
            return []

        try:
            with open(req_file, 'r') as f:
                return [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
        except:
            return []