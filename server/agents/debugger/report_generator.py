from typing import List, Dict, Any

class DebugReport:
    def __init__(self, root_cause: str, analysis: str, recommendations: List[str]):
        self.root_cause = root_cause
        self.analysis = analysis
        self.recommendations = recommendations

class ReportGenerator:
    def generate_report(self, analysis: str) -> DebugReport:
        # Simple parsing of analysis to extract components
        lines = analysis.split('\n')
        root_cause = ""
        recommendations = []
        in_recommendations = False

        for line in lines:
            if "root cause" in line.lower():
                root_cause = line
            elif "potential fixes" in line.lower() or "recommendations" in line.lower():
                in_recommendations = True
            elif in_recommendations and line.strip():
                recommendations.append(line.strip())

        return DebugReport(
            root_cause=root_cause,
            analysis=analysis,
            recommendations=recommendations
        )