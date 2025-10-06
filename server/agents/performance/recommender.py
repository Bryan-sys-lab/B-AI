from providers.nim_adapter import NIMAdapter
from typing import Dict, Any, List

class PerformanceRecommender:
    def __init__(self):
        self.adapter = NIMAdapter()

    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance recommendations using Mistral Aetherium."""
        try:
            # Prepare context for Mistral
            context = self._prepare_context(analysis_results)

            messages = [
                {
                    "role": "system",
                    "content": "You are a performance optimization expert. Analyze the provided profiling data and generate actionable recommendations for improving code performance, including caching strategies, query optimizations, scaling policies, and infrastructure recommendations."
                },
                {
                    "role": "user",
                    "content": f"Analyze this performance data and provide recommendations:\n\n{context}"
                }
            ]

            response = self.adapter.call_model(messages)

            recommendations = self._parse_recommendations(response.text)

            return {
                "recommendations": recommendations,
                "ai_analysis": response.text,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def _prepare_context(self, results: Dict[str, Any]) -> str:
        """Prepare analysis results for Aetherium processing."""
        context_parts = []

        if "cprofile" in results:
            cp = results["cprofile"]
            if cp.get("success"):
                context_parts.append(f"cProfile Analysis:\nTotal Time: {cp.get('total_time', 0)}s")
                context_parts.append("Hotspots:")
                for hotspot in cp.get("hotspots", [])[:5]:
                    context_parts.append(f"- {hotspot['function']}: {hotspot['cumulative_time']}s cumulative")

        if "pyspy" in results:
            ps = results["pyspy"]
            if ps.get("success"):
                context_parts.append(f"Py-Spy Analysis:\nTotal Samples: {ps.get('total_samples', 0)}")
                for hotspot in ps.get("hotspots", [])[:5]:
                    context_parts.append(f"- {hotspot['function']}: {hotspot['samples']} samples")

        if "load_test" in results:
            lt = results["load_test"]
            if lt.get("success"):
                context_parts.append(f"Load Test Results:\nRequests/sec: {lt.get('requests_per_second', 0)}")
                context_parts.append(f"Failures: {lt.get('failures', 0)}")

        return "\n".join(context_parts)

    def _parse_recommendations(self, ai_response: str) -> Dict[str, List[str]]:
        """Parse Aetherium response into structured recommendations."""
        recommendations = {
            "caching": [],
            "query_optimization": [],
            "scaling": [],
            "infrastructure": [],
            "code_improvements": []
        }

        # Simple parsing based on keywords
        lines = ai_response.lower().split('\n')
        current_category = None

        for line in lines:
            line = line.strip()
            if "cach" in line:
                current_category = "caching"
            elif "query" in line or "database" in line:
                current_category = "query_optimization"
            elif "scal" in line:
                current_category = "scaling"
            elif "infrastruct" in line or "server" in line:
                current_category = "infrastructure"
            elif "code" in line or "function" in line:
                current_category = "code_improvements"

            if current_category and line.startswith(('-', '*', '•')):
                recommendations[current_category].append(line[1:].strip())

        # Fallback: if no structured parsing, put everything in code_improvements
        if not any(recommendations.values()):
            recommendations["code_improvements"] = [ai_response]

        return recommendations

    def get_preset_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate preset recommendations based on common patterns."""
        recommendations = {
            "caching": [],
            "query_optimization": [],
            "scaling": [],
            "infrastructure": [],
            "code_improvements": []
        }

        # Check for high call counts
        if "cprofile" in analysis_results:
            cp = analysis_results["cprofile"]
            for hotspot in cp.get("hotspots", []):
                if hotspot.get("calls", 0) > 1000:
                    recommendations["caching"].append(f"Consider caching results of {hotspot['function']} to reduce {hotspot['calls']} calls")
                if hotspot.get("cumulative_time", 0) > 5.0:
                    recommendations["code_improvements"].append(f"Optimize {hotspot['function']} - high cumulative time: {hotspot['cumulative_time']}s")

        # Load test recommendations
        if "load_test" in analysis_results:
            lt = analysis_results["load_test"]
            if lt.get("requests_per_second", 0) < 10:
                recommendations["scaling"].append("Consider horizontal scaling - low requests per second")
            if lt.get("failures", 0) > 0:
                recommendations["infrastructure"].append("Address load test failures - check resource limits")

        return recommendations