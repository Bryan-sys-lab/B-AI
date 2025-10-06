import os
import re
import ast
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

class PolicyViolation(BaseModel):
    rule: str
    severity: str
    file: str
    line: Optional[int] = None
    description: str
    recommendation: str

class PolicyCheckResult(BaseModel):
    violations: List[PolicyViolation]
    summary: Dict[str, Union[int, str]]

class PolicyEnforcer:
    def __init__(self):
        self.policies = {
            'hardcoded_secrets': self._check_hardcoded_secrets,
            'sql_injection': self._check_sql_injection,
            'xss_vulnerable': self._check_xss_vulnerable,
            'weak_crypto': self._check_weak_crypto,
            'insecure_random': self._check_insecure_random,
            'path_traversal': self._check_path_traversal
        }

    def enforce_policies(self, path: str, language: str = 'python') -> PolicyCheckResult:
        """Enforce security policies on codebase"""
        violations = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if self._should_check_file(file, language):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = content.split('\n')

                        for rule_name, check_func in self.policies.items():
                            rule_violations = check_func(content, lines, filepath)
                            violations.extend(rule_violations)
                    except Exception as e:
                        print(f"Error checking {filepath}: {e}")

        summary = {
            'total': len(violations),
            'high': len([v for v in violations if v.severity == 'high']),
            'medium': len([v for v in violations if v.severity == 'medium']),
            'low': len([v for v in violations if v.severity == 'low'])
        }

        return PolicyCheckResult(violations=violations, summary=summary)

    def _should_check_file(self, filename: str, language: str) -> bool:
        """Check if file should be analyzed"""
        if language == 'python':
            return filename.endswith(('.py', '.pyw'))
        elif language == 'javascript':
            return filename.endswith(('.js', '.jsx', '.ts', '.tsx'))
        return False

    def _check_hardcoded_secrets(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for hardcoded secrets like API keys, passwords"""
        violations = []

        # Patterns for potential secrets
        patterns = [
            (r'api[_-]?key\s*[:=]\s*["\'][^"\']+["\']', 'API key'),
            (r'password\s*[:=]\s*["\'][^"\']+["\']', 'Password'),
            (r'secret[_-]?key\s*[:=]\s*["\'][^"\']+["\']', 'Secret key'),
            (r'token\s*[:=]\s*["\'][^"\']+["\']', 'Token'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, secret_type in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(PolicyViolation(
                        rule='hardcoded_secrets',
                        severity='high',
                        file=filepath,
                        line=i,
                        description=f'Potential hardcoded {secret_type} found',
                        recommendation='Use environment variables or secure credential storage'
                    ))

        return violations

    def _check_sql_injection(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for potential SQL injection vulnerabilities"""
        violations = []

        # Simple pattern for string concatenation in SQL
        sql_patterns = [
            r'execute\s*\(\s*["\'].*?\+\s*.*?\s*["\']',
            r'cursor\.execute\s*\(\s*["\'].*?\+\s*.*?\s*["\']',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(PolicyViolation(
                        rule='sql_injection',
                        severity='high',
                        file=filepath,
                        line=i,
                        description='Potential SQL injection vulnerability',
                        recommendation='Use parameterized queries or prepared statements'
                    ))

        return violations

    def _check_xss_vulnerable(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for potential XSS vulnerabilities"""
        violations = []

        # Check for direct HTML output without escaping
        xss_patterns = [
            r'innerHTML\s*[:=]\s*.*?\+\s*.*',
            r'document\.write\s*\(\s*.*?\+\s*.*',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in xss_patterns:
                if re.search(pattern, line):
                    violations.append(PolicyViolation(
                        rule='xss_vulnerable',
                        severity='high',
                        file=filepath,
                        line=i,
                        description='Potential XSS vulnerability',
                        recommendation='Use proper output encoding or templating engines'
                    ))

        return violations

    def _check_weak_crypto(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for weak cryptographic practices"""
        violations = []

        weak_algorithms = ['md5', 'sha1', 'des', 'rc4']

        for i, line in enumerate(lines, 1):
            for algo in weak_algorithms:
                if re.search(rf'\b{algo}\b', line, re.IGNORECASE):
                    violations.append(PolicyViolation(
                        rule='weak_crypto',
                        severity='medium',
                        file=filepath,
                        line=i,
                        description=f'Weak cryptographic algorithm {algo} detected',
                        recommendation='Use stronger algorithms like SHA-256, AES-256'
                    ))

        return violations

    def _check_insecure_random(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for insecure random number generation"""
        violations = []

        if 'random.' in content and 'import random' in content:
            for i, line in enumerate(lines, 1):
                if 'random.random()' in line or 'random.randint(' in line:
                    violations.append(PolicyViolation(
                        rule='insecure_random',
                        severity='medium',
                        file=filepath,
                        line=i,
                        description='Using insecure random number generation',
                        recommendation='Use secrets module for cryptographic purposes'
                    ))

        return violations

    def _check_path_traversal(self, content: str, lines: List[str], filepath: str) -> List[PolicyViolation]:
        """Check for path traversal vulnerabilities"""
        violations = []

        for i, line in enumerate(lines, 1):
            if '../' in line or '..\\' in line:
                # Check if it's in file operations
                if any(func in line for func in ['open(', 'read(', 'write(', 'os.path.join']):
                    violations.append(PolicyViolation(
                        rule='path_traversal',
                        severity='high',
                        file=filepath,
                        line=i,
                        description='Potential path traversal vulnerability',
                        recommendation='Validate and sanitize file paths'
                    ))

        return violations