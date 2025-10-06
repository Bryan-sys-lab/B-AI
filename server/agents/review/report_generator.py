import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from providers.nim_adapter import NIMAdapter

class ReviewReport(BaseModel):
    timestamp: str
    code_path: str
    language: str
    static_analysis: Dict[str, Any]
    security_scan: Dict[str, Any]
    style_check: Dict[str, Any]
    compliance_check: Dict[str, Any]
    risk_score: float
    approval_status: str
    summary: str
    recommendations: List[str]
    detailed_findings: Dict[str, List[Dict[str, Any]]]

class ReportGenerator:
    def __init__(self, adapter: Optional[NIMAdapter] = None):
        self.adapter = adapter or NIMAdapter()

    def generate_report(
        self,
        code_path: str,
        language: str,
        static_results: List[Any],
        security_results: List[Any],
        style_results: List[Any],
        compliance_results: List[Any]
    ) -> ReviewReport:
        """Generate a comprehensive review report."""

        # Aggregate findings
        detailed_findings = {
            "static_analysis": self._extract_findings(static_results),
            "security": self._extract_findings(security_results),
            "style": self._extract_findings(style_results),
            "compliance": self._extract_findings(compliance_results)
        }

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            static_results, security_results, style_results, compliance_results
        )

        # Determine approval status
        approval_status = self._determine_approval_status(risk_score)

        # Generate summary and recommendations using Mistral
        summary = self._generate_summary(detailed_findings, risk_score)
        recommendations = self._generate_recommendations(detailed_findings, risk_score)

        return ReviewReport(
            timestamp=datetime.utcnow().isoformat(),
            code_path=code_path,
            language=language,
            static_analysis=self._summarize_results(static_results),
            security_scan=self._summarize_results(security_results),
            style_check=self._summarize_results(style_results),
            compliance_check=self._summarize_results(compliance_results),
            risk_score=risk_score,
            approval_status=approval_status,
            summary=summary,
            recommendations=recommendations,
            detailed_findings=detailed_findings
        )

    def _extract_findings(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Extract findings from result objects."""
        findings = []
        for result in results:
            if hasattr(result, 'findings') and result.findings:
                findings.extend(result.findings)
            elif hasattr(result, 'success') and not result.success and hasattr(result, 'error'):
                findings.append({
                    "type": "error",
                    "tool": getattr(result, 'tool', 'unknown'),
                    "message": result.error
                })
        return findings

    def _summarize_results(self, results: List[Any]) -> Dict[str, Any]:
        """Summarize results for a category."""
        if not results:
            return {"status": "not_run", "tools": []}

        summary = {
            "status": "completed",
            "tools": [],
            "total_findings": 0,
            "errors": 0
        }

        for result in results:
            tool_summary = {
                "name": getattr(result, 'tool', 'unknown'),
                "success": getattr(result, 'success', False),
                "findings_count": len(getattr(result, 'findings', []))
            }

            if hasattr(result, 'severity_counts'):
                tool_summary["severity_counts"] = result.severity_counts
            elif hasattr(result, 'error_count'):
                tool_summary["error_count"] = result.error_count
                tool_summary["warning_count"] = getattr(result, 'warning_count', 0)

            summary["tools"].append(tool_summary)
            summary["total_findings"] += tool_summary["findings_count"]

            if not result.success:
                summary["errors"] += 1

        return summary

    def _calculate_risk_score(self, static_results, security_results, style_results, compliance_results) -> float:
        """Calculate overall risk score from 0.0 (low risk) to 1.0 (high risk)."""
        score = 0.0
        weights = {
            "security": 0.4,    # Security issues are most critical
            "static": 0.3,      # Static analysis issues
            "compliance": 0.2,  # Compliance issues
            "style": 0.1        # Style issues are least critical
        }

        # Security score
        security_score = self._calculate_category_score(security_results, severity_weights={"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.2})
        score += security_score * weights["security"]

        # Static analysis score
        static_score = self._calculate_category_score(static_results, severity_weights={"ERROR": 0.8, "WARNING": 0.4, "INFO": 0.1})
        score += static_score * weights["static"]

        # Compliance score
        compliance_score = self._calculate_category_score(compliance_results, severity_weights={"failed": 0.7, "passed": 0.0})
        score += compliance_score * weights["compliance"]

        # Style score
        style_score = self._calculate_category_score(style_results, severity_weights={"error": 0.3, "warning": 0.1})
        score += style_score * weights["style"]

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_category_score(self, results: List[Any], severity_weights: Dict[str, float]) -> float:
        """Calculate score for a category of results."""
        if not results:
            return 0.0

        total_findings = 0
        weighted_score = 0.0

        for result in results:
            findings = getattr(result, 'findings', [])
            total_findings += len(findings)

            for finding in findings:
                severity = "unknown"
                if hasattr(result, 'severity_counts'):
                    # For results with severity counts
                    for sev, count in result.severity_counts.items():
                        if count > 0:
                            severity = sev.upper()
                            break
                elif 'severity' in finding:
                    severity = str(finding['severity']).upper()
                elif 'status' in finding:
                    severity = finding['status'].lower()

                weight = severity_weights.get(severity, 0.1)
                weighted_score += weight

        if total_findings == 0:
            return 0.0

        return weighted_score / max(total_findings, 1)

    def _determine_approval_status(self, risk_score: float) -> str:
        """Determine approval status based on risk score."""
        if risk_score < 0.2:
            return "approved"
        elif risk_score < 0.5:
            return "approved_with_warnings"
        elif risk_score < 0.8:
            return "requires_attention"
        else:
            return "rejected"

    def _generate_summary(self, detailed_findings: Dict[str, List], risk_score: float) -> str:
        """Generate a human-readable summary using Mistral."""
        try:
            findings_summary = json.dumps(detailed_findings, indent=2)

            prompt = f"""
            Based on the following code review findings, provide a concise summary of the code quality and any issues found.
            Risk score: {risk_score:.2f}

            Findings:
            {findings_summary}

            Provide a brief summary (2-3 sentences) highlighting the main issues and overall assessment.
            """

            messages = [{"role": "user", "content": prompt}]
            response = self.adapter.call_model(messages)
            return response.text.strip()

        except Exception as e:
            return f"Code review completed with risk score {risk_score:.2f}. Error generating Aetherium summary: {str(e)}"

    def _generate_recommendations(self, detailed_findings: Dict[str, List], risk_score: float) -> List[str]:
        """Generate recommendations using Mistral."""
        try:
            findings_summary = json.dumps(detailed_findings, indent=2)

            prompt = f"""
            Based on the following code review findings, provide specific recommendations to improve code quality.
            Risk score: {risk_score:.2f}

            Findings:
            {findings_summary}

            Provide 3-5 actionable recommendations, prioritized by importance. Be specific about what needs to be fixed.
            """

            messages = [{"role": "user", "content": prompt}]
            response = self.adapter.call_model(messages)
            recommendations = [line.strip() for line in response.text.split('\n') if line.strip()]
            return recommendations[:5]  # Limit to 5 recommendations

        except Exception as e:
            return [f"Address high-priority security and static analysis findings. Error generating Aetherium recommendations: {str(e)}"]