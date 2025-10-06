import requests
import tempfile
import os
from typing import Dict, Any, Optional

class LoadTester:
    def __init__(self, sandbox_url: str = "http://localhost:8002"):
        self.sandbox_url = sandbox_url

    def run_load_test(self, target_url: str, users: int = 10, duration: int = 60) -> Dict[str, Any]:
        """Run load testing using Locust."""
        try:
            # Create a Locust test script
            locust_script = f"""
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def load_page(self):
        self.client.get("/")

if __name__ == "__main__":
    import os
    os.system(f"locust -f {{__file__}} --host={target_url} --users={users} --run-time={duration}s --headless --csv=results")
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(locust_script)
                temp_file = f.name

            # Run Locust in sandbox
            command = f"python {temp_file}"
            response = requests.post(
                f"{self.sandbox_url}/execute",
                json={
                    "command": "python",
                    "args": [temp_file],
                    "working_dir": "/workspace",
                    "timeout": duration + 30
                }
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("exit_code") == 0:
                    # Parse results (simplified)
                    stdout = result.get("stdout", "")
                    return {
                        "tool": "Locust",
                        "results": stdout,
                        "success": True,
                        "users": users,
                        "duration": duration,
                        "target_url": target_url
                    }
                else:
                    return {
                        "tool": "Locust",
                        "error": result.get("stderr", "Unknown error"),
                        "success": False
                    }
            else:
                return {
                    "tool": "Locust",
                    "error": f"Sandbox error: {response.status_code}",
                    "success": False
                }
        except Exception as e:
            return {
                "tool": "Locust",
                "error": str(e),
                "success": False
            }
        finally:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass