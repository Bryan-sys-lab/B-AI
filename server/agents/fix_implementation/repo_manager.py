import requests
import os
import subprocess
import json
from typing import Optional, Dict, List
from pathlib import Path

class RepoManager:
    def __init__(self):
        self.tool_api_url = os.getenv("TOOL_API_GATEWAY_URL", "http://localhost:8001")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_api_url = "https://api.github.com"

    def apply_patch(self, repo_url: str, branch: str, patch: str) -> bool:
        # Parse the unified diff to get file path and new content
        file_path, new_content = self._parse_diff(patch)
        if not file_path or not new_content:
            return False

        # Use tool API to write the file
        response = requests.post(f"{self.tool_api_url}/git_write_file", json={
            "repo_url": repo_url,
            "file_path": file_path,
            "content": new_content,
            "branch": branch,
            "commit_message": "Apply fix patch"
        })

        return response.status_code == 200 and response.json().get("success", False)

    def _parse_diff(self, diff: str) -> tuple[Optional[str], Optional[str]]:
        # Simple parser for unified diff
        lines = diff.split('\n')
        if len(lines) < 3:
            return None, None

        # Assume format: --- a/file\n+++ b/file\n@@ ... @@\n+new lines
        # For MVP, extract file path from +++ line
        file_line = None
        for line in lines:
            if line.startswith('+++ b/'):
                file_line = line[6:]  # remove +++ b/
                break

        if not file_line:
            return None, None

        # For simplicity, assume the diff adds lines, extract + lines
        new_lines = [line[1:] for line in lines if line.startswith('+')]
        new_content = '\n'.join(new_lines)

        return file_line, new_content

    def create_github_repo(self, name: str, description: str = "", private: bool = False) -> Optional[Dict]:
        """Create a new GitHub repository"""
        if not self.github_token:
            return {"error": "GitHub token not configured"}

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": False
        }

        response = requests.post(f"{self.github_api_url}/user/repos", headers=headers, json=data)

        if response.status_code == 201:
            return response.json()
        else:
            return {"error": f"Failed to create repo: {response.text}"}

    def initialize_git_repo(self, local_path: str) -> bool:
        """Initialize a git repository in the local directory"""
        try:
            subprocess.run(["git", "init"], cwd=local_path, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "CodeAgent"], cwd=local_path, check=True)
            subprocess.run(["git", "config", "user.email", "codeagent@example.com"], cwd=local_path, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to initialize git repo: {e}")
            return False

    def add_remote_origin(self, local_path: str, repo_url: str) -> bool:
        """Add GitHub repository as remote origin"""
        try:
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=local_path, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to add remote origin: {e}")
            return False

    def add_files_to_git(self, local_path: str, files: List[str]) -> bool:
        """Add files to git staging area"""
        try:
            # Add all files if no specific files provided
            if not files:
                subprocess.run(["git", "add", "."], cwd=local_path, check=True, capture_output=True)
            else:
                for file_path in files:
                    subprocess.run(["git", "add", file_path], cwd=local_path, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to add files to git: {e}")
            return False

    def commit_changes(self, local_path: str, message: str = "Initial commit") -> bool:
        """Commit staged changes"""
        try:
            subprocess.run(["git", "commit", "-m", message], cwd=local_path, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to commit changes: {e}")
            return False

    def push_to_github(self, local_path: str, branch: str = "main") -> bool:
        """Push commits to GitHub"""
        try:
            subprocess.run(["git", "push", "-u", "origin", branch], cwd=local_path, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to push to GitHub: {e}")
            return False

    def deploy_project_to_github(self, project_name: str, local_path: str, description: str = "", private: bool = False) -> Dict:
        """Complete workflow to deploy a project to GitHub"""
        result = {"success": False, "repo_url": None, "error": None}

        # 1. Create GitHub repository
        repo_data = self.create_github_repo(project_name, description, private)
        if "error" in repo_data:
            result["error"] = repo_data["error"]
            return result

        repo_url = repo_data["html_url"]
        result["repo_url"] = repo_url

        # 2. Initialize local git repo
        if not self.initialize_git_repo(local_path):
            result["error"] = "Failed to initialize local git repository"
            return result

        # 3. Add remote origin
        if not self.add_remote_origin(local_path, repo_data["clone_url"]):
            result["error"] = "Failed to add GitHub remote"
            return result

        # 4. Add all files
        if not self.add_files_to_git(local_path, []):
            result["error"] = "Failed to add files to git"
            return result

        # 5. Commit changes
        if not self.commit_changes(local_path, f"Deploy {project_name} project"):
            result["error"] = "Failed to commit changes"
            return result

        # 6. Push to GitHub
        if not self.push_to_github(local_path):
            result["error"] = "Failed to push to GitHub"
            return result

        result["success"] = True
        return result