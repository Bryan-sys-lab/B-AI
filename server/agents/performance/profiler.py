import cProfile
import pstats
import io
import requests
import json
from typing import Dict, Any, Optional
import tempfile
import os

class Profiler:
    def __init__(self, sandbox_url: str = "http://localhost:8002"):
        self.sandbox_url = sandbox_url

    def profile_with_cprofile(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Profile code execution using cProfile."""
        if language != "python":
            return {"error": "cProfile only supports Python"}

        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Profile the code
            pr = cProfile.Profile()
            pr.enable()

            # Execute the code (in a safe way, but for now assume it's safe)
            exec(code)

            pr.disable()

            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            profile_output = s.getvalue()

            # Clean up
            os.unlink(temp_file)

            return {
                "tool": "cProfile",
                "profile_data": profile_output,
                "success": True
            }
        except Exception as e:
            return {
                "tool": "cProfile",
                "error": str(e),
                "success": False
            }

    def profile_with_pyspy(self, code: str, language: str = "python", duration: int = 10) -> Dict[str, Any]:
        """Profile code execution using Py-Spy."""
        if language != "python":
            return {"error": "Py-Spy only supports Python"}

        try:
            # Create a script to run the code
            script_content = f"""
import time
{code}
time.sleep({duration})
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                temp_file = f.name

            # Use sandbox to run py-spy
            command = f"py-spy record --format speedscope --output profile.speedscope -- python {temp_file}"
            response = requests.post(
                f"{self.sandbox_url}/execute",
                json={
                    "command": "sh",
                    "args": ["-c", command],
                    "working_dir": "/workspace",
                    "timeout": duration + 10
                }
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("exit_code") == 0:
                    # Read the profile file (assuming it's captured)
                    profile_data = result.get("stdout", "")
                    return {
                        "tool": "Py-Spy",
                        "profile_data": profile_data,
                        "success": True
                    }
                else:
                    return {
                        "tool": "Py-Spy",
                        "error": result.get("stderr", "Unknown error"),
                        "success": False
                    }
            else:
                return {
                    "tool": "Py-Spy",
                    "error": f"Sandbox error: {response.status_code}",
                    "success": False
                }
        except Exception as e:
            return {
                "tool": "Py-Spy",
                "error": str(e),
                "success": False
            }
        finally:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass