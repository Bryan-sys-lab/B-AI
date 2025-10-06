from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from .vulnerability_scanner import Vulnerability, ScanResult
from .dependency_checker import DependencyVulnerability, DependencyScanResult
from .policy_enforcer import PolicyViolation, PolicyCheckResult

class SecurityFix(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    risk_score: float
    priority: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'vulnerability', 'dependency', 'policy'
    file: Optional[str] = None
    line: Optional[int] = None
    cwe: Optional[str] = None
    cvss_score: Optional[float] = None
    exploitability: str
    impact: str
    remediation: str
    references: List[str] = []

class RiskAssessment(BaseModel):
    fixes: List[SecurityFix]
    summary: Dict[str, Union[int, str]]
    risk_level: str  # 'critical', 'high', 'medium', 'low'

class RiskAssessor:
    def __init__(self):
        self.severity_weights = {
            'critical': 10,
            'high': 7,
            'medium': 4,
            'low': 2,
            'info': 1
        }

    def assess_and_prioritize(
        self,
        vuln_results: List[ScanResult],
        dep_results: DependencyScanResult,
        policy_results: PolicyCheckResult,
        thresholds: Optional[Dict[str, int]] = None
    ) -> RiskAssessment:
        """Assess risk and prioritize security fixes"""

        if thresholds is None:
            thresholds = {
                'critical': 5,
                'high': 10,
                'medium': 20
            }

        fixes = []

        # Process vulnerability scan results
        for result in vuln_results:
            for vuln in result.vulnerabilities:
                fix = self._create_fix_from_vulnerability(vuln, result.tool)
                fixes.append(fix)

        # Process dependency scan results
        for vuln in dep_results.vulnerabilities:
            fix = self._create_fix_from_dependency_vuln(vuln)
            fixes.append(fix)

        # Process policy violations
        for violation in policy_results.violations:
            fix = self._create_fix_from_policy_violation(violation)
            fixes.append(fix)

        # Sort by risk score (descending)
        fixes.sort(key=lambda x: x.risk_score, reverse=True)

        # Assign priorities based on thresholds
        for fix in fixes:
            if fix.risk_score >= thresholds['critical']:
                fix.priority = 'critical'
            elif fix.risk_score >= thresholds['high']:
                fix.priority = 'high'
            elif fix.risk_score >= thresholds['medium']:
                fix.priority = 'medium'
            else:
                fix.priority = 'low'

        # Calculate overall risk level
        risk_level = self._calculate_overall_risk(fixes)

        summary = {
            'total_fixes': len(fixes),
            'critical': len([f for f in fixes if f.priority == 'critical']),
            'high': len([f for f in fixes if f.priority == 'high']),
            'medium': len([f for f in fixes if f.priority == 'medium']),
            'low': len([f for f in fixes if f.priority == 'low'])
        }

        return RiskAssessment(fixes=fixes, summary=summary, risk_level=risk_level)

    def _create_fix_from_vulnerability(self, vuln: Vulnerability, tool: str) -> SecurityFix:
        """Create a security fix from a vulnerability"""
        risk_score = self._calculate_vuln_risk_score(vuln)

        return SecurityFix(
            id=f"{tool}_{vuln.id}",
            title=f"Vulnerability: {vuln.id}",
            description=vuln.description,
            severity=vuln.severity,
            risk_score=risk_score,
            priority='medium',  # Will be reassigned
            category='vulnerability',
            file=vuln.file,
            line=vuln.line,
            cwe=vuln.cwe,
            cvss_score=vuln.cvss_score,
            exploitability=self._assess_exploitability(vuln),
            impact=self._assess_impact(vuln),
            remediation=self._suggest_remediation(vuln),
            references=[]
        )

    def _create_fix_from_dependency_vuln(self, vuln: DependencyVulnerability) -> SecurityFix:
        """Create a security fix from a dependency vulnerability"""
        risk_score = self._calculate_dep_risk_score(vuln)

        return SecurityFix(
            id=f"dep_{vuln.vulnerability_id}",
            title=f"Dependency Vulnerability: {vuln.vulnerability_id}",
            description=vuln.description,
            severity=vuln.severity,
            risk_score=risk_score,
            priority='medium',  # Will be reassigned
            category='dependency',
            file=vuln.file,
            cwe=vuln.cwe,
            cvss_score=vuln.cvss_score,
            exploitability='high',  # Dependencies are often easily exploitable
            impact='high',  # Can affect entire application
            remediation=f"Update {vuln.dependency} to a secure version",
            references=vuln.references
        )

    def _create_fix_from_policy_violation(self, violation: PolicyViolation) -> SecurityFix:
        """Create a security fix from a policy violation"""
        risk_score = self._calculate_policy_risk_score(violation)

        return SecurityFix(
            id=f"policy_{violation.rule}",
            title=f"Policy Violation: {violation.rule}",
            description=violation.description,
            severity=violation.severity,
            risk_score=risk_score,
            priority='medium',  # Will be reassigned
            category='policy',
            file=violation.file,
            line=violation.line,
            exploitability='medium',
            impact='medium',
            remediation=violation.recommendation,
            references=[]
        )

    def _calculate_vuln_risk_score(self, vuln: Vulnerability) -> float:
        """Calculate risk score for a vulnerability"""
        base_score = self.severity_weights.get(vuln.severity.lower(), 1)

        # Factor in CVSS score if available
        if vuln.cvss_score:
            cvss_factor = vuln.cvss_score / 10.0  # Normalize to 0-1
            base_score *= (1 + cvss_factor)

        return min(base_score, 10.0)  # Cap at 10

    def _calculate_dep_risk_score(self, vuln: DependencyVulnerability) -> float:
        """Calculate risk score for a dependency vulnerability"""
        base_score = self.severity_weights.get(vuln.severity.lower(), 1)

        # Dependencies are often more critical
        base_score *= 1.5

        if vuln.cvss_score:
            cvss_factor = vuln.cvss_score / 10.0
            base_score *= (1 + cvss_factor)

        return min(base_score, 10.0)

    def _calculate_policy_risk_score(self, violation: PolicyViolation) -> float:
        """Calculate risk score for a policy violation"""
        return self.severity_weights.get(violation.severity.lower(), 1)

    def _assess_exploitability(self, vuln: Vulnerability) -> str:
        """Assess exploitability of a vulnerability"""
        severity = vuln.severity.lower()
        if severity in ['critical', 'high']:
            return 'high'
        elif severity == 'medium':
            return 'medium'
        else:
            return 'low'

    def _assess_impact(self, vuln: Vulnerability) -> str:
        """Assess impact of a vulnerability"""
        severity = vuln.severity.lower()
        if severity in ['critical', 'high']:
            return 'high'
        elif severity == 'medium':
            return 'medium'
        else:
            return 'low'

    def _suggest_remediation(self, vuln: Vulnerability) -> str:
        """Suggest remediation for a vulnerability"""
        if vuln.cwe:
            cwe_num = vuln.cwe.replace('CWE-', '')
            return f"Follow CWE-{cwe_num} guidelines for remediation"
        else:
            return "Apply security patches and follow best practices"

    def _calculate_overall_risk(self, fixes: List[SecurityFix]) -> str:
        """Calculate overall risk level"""
        if any(f.priority == 'critical' for f in fixes):
            return 'critical'
        elif any(f.priority == 'high' for f in fixes):
            return 'high'
        elif any(f.priority == 'medium' for f in fixes):
            return 'medium'
        else:
            return 'low'

    def should_block_merge(self, assessment: RiskAssessment, thresholds: Optional[Dict[str, int]] = None) -> bool:
        """Determine if merge should be blocked based on risk assessment"""
        if thresholds is None:
            thresholds = {
                'max_critical': 0,  # No critical issues allowed
                'max_high': 5,
                'max_medium': 10
            }

        summary = assessment.summary
        if isinstance(summary.get('critical'), int) and summary['critical'] > thresholds['max_critical']:
            return True
        if isinstance(summary.get('high'), int) and summary['high'] > thresholds['max_high']:
            return True
        if isinstance(summary.get('medium'), int) and summary['medium'] > thresholds['max_medium']:
            return True

        return False