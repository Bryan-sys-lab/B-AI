import re
from typing import Dict, Any, List
import json

class PerformanceAnalyzer:
    def analyze_cprofile_data(self, profile_data: str) -> Dict[str, Any]:
        """Analyze cProfile output to extract key metrics."""
        try:
            lines = profile_data.strip().split('\n')
            hotspots = []
            total_time = 0.0

            # Parse the profile stats
            for line in lines:
                if line.strip() and not line.startswith(' ') and 'function calls' not in line.lower():
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 6:
                        try:
                            ncalls = float(parts[0].split('/')[0])
                            tottime = float(parts[1])
                            percall = float(parts[2])
                            cumtime = float(parts[3])
                            percall_cum = float(parts[4])
                            filename_lineno = parts[5]

                            hotspots.append({
                                "function": filename_lineno,
                                "calls": ncalls,
                                "total_time": tottime,
                                "cumulative_time": cumtime,
                                "per_call": percall,
                                "per_call_cumulative": percall_cum
                            })

                            total_time += cumtime
                        except (ValueError, IndexError):
                            continue

            # Sort by cumulative time
            hotspots.sort(key=lambda x: x["cumulative_time"], reverse=True)

            return {
                "total_time": total_time,
                "hotspots": hotspots[:10],  # Top 10
                "bottlenecks": self.identify_bottlenecks(hotspots),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def analyze_pyspy_data(self, profile_data: str) -> Dict[str, Any]:
        """Analyze Py-Spy output."""
        # Py-Spy outputs JSON or speedscope format
        try:
            if profile_data.strip().startswith('{'):
                data = json.loads(profile_data)
                # Extract key metrics from speedscope format
                profiles = data.get("profiles", [])
                if profiles:
                    profile = profiles[0]
                    samples = profile.get("samples", [])
                    frames = profile.get("frames", [])

                    # Simple analysis: count function occurrences
                    function_counts = {}
                    for sample in samples:
                        for frame_id in sample:
                            if frame_id < len(frames):
                                frame = frames[frame_id]
                                func_name = frame.get("name", "unknown")
                                function_counts[func_name] = function_counts.get(func_name, 0) + 1

                    sorted_functions = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)

                    return {
                        "total_samples": len(samples),
                        "hotspots": [{"function": func, "samples": count} for func, count in sorted_functions[:10]],
                        "success": True
                    }
            else:
                # Plain text output
                return {
                    "raw_data": profile_data,
                    "success": True
                }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def analyze_load_test_data(self, load_data: str) -> Dict[str, Any]:
        """Analyze Locust load test results."""
        try:
            # Parse Locust output (simplified)
            metrics = {
                "requests_per_second": 0,
                "response_times": {},
                "failures": 0,
                "success": True
            }

            lines = load_data.split('\n')
            for line in lines:
                if 'Requests/sec' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        metrics["requests_per_second"] = float(match.group(1))
                elif 'response time' in line.lower():
                    # Extract response time percentiles
                    pass
                elif 'failures' in line.lower():
                    match = re.search(r'(\d+)', line)
                    if match:
                        metrics["failures"] = int(match.group(1))

            return metrics
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def identify_bottlenecks(self, hotspots: List[Dict]) -> List[str]:
        """Identify potential bottlenecks from hotspots."""
        bottlenecks = []
        for hotspot in hotspots[:5]:  # Top 5
            if hotspot["cumulative_time"] > 1.0:  # More than 1 second
                bottlenecks.append(f"High cumulative time in {hotspot['function']}: {hotspot['cumulative_time']}s")
            if hotspot["calls"] > 1000:  # Many calls
                bottlenecks.append(f"High call count in {hotspot['function']}: {hotspot['calls']} calls")

        return bottlenecks