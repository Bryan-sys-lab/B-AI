from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import tempfile
import subprocess
import httpx
from pathlib import Path
import asyncio
from datetime import datetime
import secrets
import time
import difflib
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Enhanced Workspace Service")

# WebSocket connection manager for terminal streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Failed to send message to connection: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Multi-root workspace management
class WorkspaceConfig(BaseModel):
    id: str
    name: str
    type: str  # 'local', 'git', 'cloud', 'project'
    path: Optional[str] = None
    repo_url: Optional[str] = None
    branch: Optional[str] = None
    provider: Optional[str] = None
    project_id: Optional[str] = None  # For project-linked workspaces

class FileSystemAPI:
    def __init__(self):
        self.workspaces = {}
        self.active_workspace = None

    def add_workspace(self, config: WorkspaceConfig):
        """Add a new workspace"""
        self.workspaces[config.id] = config
        if not self.active_workspace:
            self.active_workspace = config.id

    def get_workspace_files(self, workspace_id: str) -> List[Dict]:
        """Get files for a specific workspace"""
        if workspace_id not in self.workspaces:
            raise HTTPException(status_code=404, detail="Workspace not found")

        config = self.workspaces[workspace_id]

        if config.type == 'local':
            return self._scan_local_directory(config.path)
        elif config.type == 'git':
            return self._scan_git_repository(config.repo_url, config.branch)
        elif config.type == 'cloud':
            return self._scan_cloud_storage(config.provider, config.path)
        elif config.type == 'project':
            return self._scan_project_workspace(config.project_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported workspace type")

    def _scan_local_directory(self, path: str) -> List[Dict]:
        """Scan a local directory"""
        items = []

        try:
            entries = os.listdir(path)
        except PermissionError:
            return items

        # Filter out hidden files and common directories to ignore
        ignore_patterns = ['.git', '__pycache__', 'node_modules', '.pytest_cache', '.ruff_cache', 'logs']
        entries = [e for e in entries if not e.startswith('.') and e not in ignore_patterns]

        for entry in sorted(entries):
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                children = self._scan_local_directory(full_path)
                if children:
                    items.append({
                        "name": entry,
                        "type": "directory",
                        "path": f"/{entry}",
                        "children": children
                    })
            else:
                items.append({
                    "name": entry,
                    "type": "file",
                    "path": f"/{entry}"
                })

        return items

    def _scan_git_repository(self, repo_url: str, branch: str = 'main') -> List[Dict]:
        """Scan a git repository"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                subprocess.run(["git", "clone", "--branch", branch, "--depth", "1", repo_url, temp_dir],
                             check=True, capture_output=True)
                return self._scan_local_directory(temp_dir)
            except subprocess.CalledProcessError:
                return []

    def _scan_cloud_storage(self, provider: str, path: str) -> List[Dict]:
        """Scan cloud storage (placeholder for future implementation)"""
        # Placeholder - would integrate with cloud providers
        return []

    def _scan_project_workspace(self, project_id: str) -> List[Dict]:
        """Scan a project workspace by getting project files from database"""
        try:
            # For now, return a placeholder - project scanning would need async handling
            # This is a simplified implementation
            return [
                {
                    "name": f"Project {project_id}",
                    "type": "directory",
                    "path": f"/project-{project_id}",
                    "children": [
                        {
                            "name": "README.md",
                            "type": "file",
                            "path": f"/project-{project_id}/README.md"
                        },
                        {
                            "name": "src",
                            "type": "directory",
                            "path": f"/project-{project_id}/src",
                            "children": []
                        }
                    ]
                }
            ]

        except Exception as e:
            print(f"Error scanning project workspace: {e}")
            return []

    def read_file(self, workspace_id: str, file_path: str) -> str:
        """Read a file from a workspace"""
        if workspace_id not in self.workspaces:
            raise HTTPException(status_code=404, detail="Workspace not found")

        config = self.workspaces[workspace_id]

        if config.type == 'local':
            full_path = os.path.join(config.path, file_path.lstrip('/'))
            if not os.path.abspath(full_path).startswith(os.path.abspath(config.path)):
                raise HTTPException(status_code=403, detail="Access denied")
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif config.type == 'git':
            with tempfile.TemporaryDirectory() as temp_dir:
                subprocess.run(["git", "clone", "--branch", config.branch, "--depth", "1",
                              config.repo_url, temp_dir], check=True, capture_output=True)
                full_path = os.path.join(temp_dir, file_path.lstrip('/'))
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported workspace type")

    def write_file(self, workspace_id: str, file_path: str, content: str):
        """Write a file to a workspace"""
        if workspace_id not in self.workspaces:
            raise HTTPException(status_code=404, detail="Workspace not found")

        config = self.workspaces[workspace_id]

        if config.type == 'local':
            full_path = os.path.join(config.path, file_path.lstrip('/'))
            if not os.path.abspath(full_path).startswith(os.path.abspath(config.path)):
                raise HTTPException(status_code=403, detail="Access denied")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            raise HTTPException(status_code=400, detail="Workspace type does not support writing")

# Aetherium Agent Runtime
class AIAgentRuntime:
    def __init__(self):
        self.conversation_history = {}
        self.context_embeddings = {}
        self.workspace_context_cache = {}
        self.user_patterns = {}
        self.code_understanding_cache = {}

    async def chat_with_agent(self, workspace_id: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Chat with Aetherium agent with workspace context - now uses system agents"""
        try:
            # Get workspace context
            workspace_context = self._get_workspace_context(workspace_id, context)

            # First, classify the request using the task classifier agent
            classification = await self._classify_request(message, context)

            # Route to appropriate specialized agent based on classification
            if classification.get("type") == "task" and classification.get("suggested_agents"):
                # Use specialized agent
                agent_response = await self._call_specialized_agent(
                    message, classification["suggested_agents"][0], context
                )
                ai_response = agent_response.get("response", "Agent execution completed")
                agent_output = agent_response
            else:
                # Fallback to general Aetherium for simple queries
                prompt = self._build_agent_prompt(message, workspace_context)
                from providers.nim_adapter import NIMAdapter
                adapter = NIMAdapter()
                messages = [{"role": "user", "content": prompt}]
                response = adapter.call_model(messages)
                ai_response = response.text if hasattr(response, 'text') else str(response)
                agent_output = None

            # Store in conversation history
            if workspace_id not in self.conversation_history:
                self.conversation_history[workspace_id] = []

            self.conversation_history[workspace_id].append({
                "timestamp": datetime.now(),
                "user_message": message,
                "ai_response": ai_response,
                "classification": classification,
                "agent_output": agent_output,
                "context": context
            })

            # Check for code suggestions
            code_suggestions = self._extract_code_suggestions(ai_response)

            return {
                "response": ai_response,
                "suggestions": code_suggestions,
                "classification": classification,
                "agent_output": agent_output,
                "workspace_context": workspace_context
            }

        except Exception as e:
            return {"error": str(e)}

    async def _classify_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify user request using the task classifier agent"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "http://localhost:8011/classify",
                    json={
                        "user_input": message,
                        "context": context
                    }
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback classification
                    return {
                        "type": "query",
                        "complexity": "simple",
                        "suggested_agents": [],
                        "confidence": 0.5
                    }
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "type": "query",
                "complexity": "simple",
                "suggested_agents": [],
                "confidence": 0.5
            }

    async def _call_specialized_agent(self, message: str, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specialized system agent with workspace context"""
        agent_urls = {
            "fix_implementation_agent": "http://localhost:8004/execute",
            "debugger_agent": "http://localhost:8005/execute",
            "review_agent": "http://localhost:8006/execute",
            "testing_agent": "http://localhost:8007/execute",
            "security_agent": "http://localhost:8008/execute",
            "performance_agent": "http://localhost:8009/execute",
            "architecture_agent": "http://localhost:8020/execute",
        }

        url = agent_urls.get(agent_name)
        if not url:
            return {"error": f"Agent {agent_name} not available"}

        try:
            # Enhance the message with workspace context
            enhanced_message = self._enhance_message_with_workspace_context(message, context)

            # Prepare agent request with workspace context
            agent_request = {
                "description": enhanced_message,
                "workspace_context": {
                    "active_workspace": context.get("workspace", "default"),
                    "active_file": context.get("activeFile"),
                    "file_content": context.get("fileContent"),
                    "workspace_files": await self._get_workspace_file_summary(context.get("workspace", "default")),
                    "available_tools": self._get_workspace_tools(context.get("workspace", "default"))
                }
            }

            async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout for complex tasks
                response = await client.post(url, json=agent_request)

                if response.status_code == 200:
                    result = response.json()

                    # Process any file operations suggested by the agent
                    await self._process_agent_file_operations(result, context.get("workspace", "default"))

                    return {
                        "response": result.get("result", "Task completed successfully"),
                        "agent": agent_name,
                        "output": result,
                        "workspace_actions": result.get("workspace_actions", [])
                    }
                else:
                    return {"error": f"Agent returned status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _enhance_message_with_workspace_context(self, message: str, context: Dict[str, Any]) -> str:
        """Enhance the user message with relevant workspace context"""
        workspace = context.get("workspace", "default")
        active_file = context.get("activeFile")

        context_parts = [f"Original request: {message}"]

        if active_file:
            context_parts.append(f"Active file: {active_file}")

        if context.get("fileContent"):
            # Add a summary of the active file content (first 500 chars)
            content_preview = context["fileContent"][:500]
            if len(context["fileContent"]) > 500:
                content_preview += "..."
            context_parts.append(f"Active file content preview:\n{content_preview}")

        # Add workspace information
        context_parts.append(f"Current workspace: {workspace}")
        context_parts.append("You have access to workspace files and can suggest modifications.")

        return "\n\n".join(context_parts)

    async def _get_workspace_file_summary(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get a summary of files in the workspace for context"""
        try:
            files = self.get_workspace_files(workspace_id)
            # Return a simplified summary to avoid token limits
            summary = []
            for file_item in files[:20]:  # Limit to first 20 files
                summary.append({
                    "name": file_item.get("name"),
                    "type": file_item.get("type"),
                    "path": file_item.get("path")
                })
            return summary
        except Exception:
            return []

    def _get_workspace_tools(self, workspace_id: str) -> List[str]:
        """Get available tools for workspace operations"""
        return [
            "read_workspace_file(workspace_id, file_path)",
            "write_workspace_file(workspace_id, file_path, content)",
            "list_workspace_files(workspace_id)",
            "run_code_in_workspace(code, language)",
            "search_workspace_files(workspace_id, query)"
        ]

    async def _process_agent_file_operations(self, agent_result: Dict[str, Any], workspace_id: str):
        """Process any file operations suggested by the agent"""
        try:
            structured = agent_result.get("structured", {})
            files = structured.get("files", {})

            # If agent generated files, save them to workspace
            if files:
                for file_path, content in files.items():
                    try:
                        # Save to workspace
                        self.write_file(workspace_id, file_path, content)
                        print(f"Agent saved file to workspace: {file_path}")
                    except Exception as e:
                        print(f"Failed to save agent file {file_path}: {e}")

        except Exception as e:
            print(f"Error processing agent file operations: {e}")

    def _get_workspace_context(self, workspace_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive workspace context with enhanced understanding"""
        # Get cached context or build new one
        cache_key = f"{workspace_id}_{context.get('activeFile', '')}"
        if cache_key in self.workspace_context_cache:
            cached_context = self.workspace_context_cache[cache_key]
            # Check if cache is still valid (within last 5 minutes)
            if (datetime.now() - cached_context.get('timestamp', datetime.min)).seconds < 300:
                return cached_context

        # Build comprehensive context
        workspace_context = {
            "active_file": context.get("activeFile"),
            "recent_files": context.get("recentFiles", []),
            "workspace_type": context.get("workspace_type", "local"),
            "language": self._detect_language(context.get("activeFile", "")),
            "timestamp": datetime.now()
        }

        # Add file-specific context
        if context.get("activeFile"):
            workspace_context.update(self._analyze_active_file(context))

        # Add workspace-wide context
        workspace_context.update(self._analyze_workspace_structure(workspace_id))

        # Add user pattern context
        workspace_context["user_patterns"] = self._get_user_patterns(workspace_id)

        # Add project context
        workspace_context["project_context"] = self._analyze_project_context(workspace_id)

        # Add code relationships
        workspace_context["code_relationships"] = self._analyze_code_relationships(workspace_id, context)

        # Cache the context
        self.workspace_context_cache[cache_key] = workspace_context

        return workspace_context

    def _analyze_active_file(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the currently active file for enhanced context"""
        file_path = context.get("activeFile", "")
        file_content = context.get("fileContent", "")

        analysis = {
            "file_size": len(file_content),
            "line_count": len(file_content.split('\n')),
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity": "simple"
        }

        if file_content:
            # Extract functions, classes, imports
            analysis["functions"] = self._extract_functions(file_content, file_path)
            analysis["classes"] = self._extract_classes(file_content, file_path)
            analysis["imports"] = self._extract_imports(file_content, file_path)
            analysis["complexity"] = self._assess_complexity(file_content)

        return {"file_analysis": analysis}

    def _analyze_workspace_structure(self, workspace_id: str) -> Dict[str, Any]:
        """Analyze the overall workspace structure"""
        try:
            files = fs_api.get_workspace_files(workspace_id)
            structure = {
                "total_files": len(files),
                "file_types": {},
                "directories": [],
                "main_languages": set(),
                "has_tests": False,
                "has_docs": False,
                "has_config": False
            }

            for file_item in files:
                if file_item.get("type") == "directory":
                    structure["directories"].append(file_item.get("name"))
                else:
                    # Count file types
                    ext = file_item.get("name", "").split('.')[-1] if '.' in file_item.get("name", "") else "no_ext"
                    structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1

                    # Detect main languages
                    if ext in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go']:
                        structure["main_languages"].add(ext)

                    # Check for special directories/files
                    name = file_item.get("name", "").lower()
                    if 'test' in name:
                        structure["has_tests"] = True
                    if name in ['readme.md', 'readme.txt', 'docs']:
                        structure["has_docs"] = True
                    if name in ['package.json', 'requirements.txt', 'setup.py', 'pom.xml', 'build.gradle']:
                        structure["has_config"] = True

            structure["main_languages"] = list(structure["main_languages"])
            return {"workspace_structure": structure}

        except Exception as e:
            return {"workspace_structure": {"error": str(e)}}

    def _get_user_patterns(self, workspace_id: str) -> Dict[str, Any]:
        """Get user interaction patterns for better context"""
        if workspace_id not in self.user_patterns:
            self.user_patterns[workspace_id] = {
                "preferred_languages": [],
                "common_tasks": [],
                "coding_style": "unknown",
                "interaction_count": 0
            }

        return self.user_patterns[workspace_id]

    def _analyze_project_context(self, workspace_id: str) -> Dict[str, Any]:
        """Analyze project-specific context (frameworks, libraries, etc.)"""
        context = {
            "frameworks": [],
            "libraries": [],
            "project_type": "unknown",
            "build_tools": [],
            "testing_frameworks": []
        }

        try:
            # Look for common project files
            files = fs_api.get_workspace_files(workspace_id)
            file_names = [f.get("name", "").lower() for f in files if f.get("type") == "file"]

            # Detect frameworks and libraries
            if "package.json" in file_names:
                content = self._read_project_file(workspace_id, "package.json")
                if content:
                    context.update(self._analyze_package_json(content))

            if "requirements.txt" in file_names:
                content = self._read_project_file(workspace_id, "requirements.txt")
                if content:
                    context["libraries"].extend(self._analyze_requirements_txt(content))

            if "setup.py" in file_names:
                context["project_type"] = "python_package"

            if "pom.xml" in file_names:
                context["project_type"] = "java_maven"

            # Detect build tools
            if "webpack.config.js" in file_names or "vite.config.js" in file_names:
                context["build_tools"].append("webpack/vite")
            if "Makefile" in file_names:
                context["build_tools"].append("make")

        except Exception as e:
            context["error"] = str(e)

        return context

    def _analyze_code_relationships(self, workspace_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code relationships and dependencies"""
        relationships = {
            "imports": {},
            "function_calls": {},
            "class_hierarchy": {},
            "file_dependencies": []
        }

        try:
            active_file = context.get("activeFile", "")
            if active_file:
                # Analyze relationships for the active file
                content = context.get("fileContent", "")
                if content:
                    relationships["imports"] = self._extract_import_relationships(content, active_file)
                    relationships["function_calls"] = self._extract_function_calls(content, active_file)

        except Exception as e:
            relationships["error"] = str(e)

        return relationships

    def _extract_functions(self, content: str, file_path: str) -> List[str]:
        """Extract function definitions from code"""
        functions = []
        try:
            if file_path.endswith('.py'):
                import re
                # Python function pattern
                pattern = r'def\s+(\w+)\s*\('
                functions = re.findall(pattern, content)
            elif file_path.endswith(('.js', '.ts')):
                # JavaScript/TypeScript function pattern
                pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|function))'
                matches = re.findall(pattern, content)
                functions = [m[0] or m[1] for m in matches if m[0] or m[1]]
        except:
            pass
        return functions

    def _extract_classes(self, content: str, file_path: str) -> List[str]:
        """Extract class definitions from code"""
        classes = []
        try:
            if file_path.endswith('.py'):
                import re
                pattern = r'class\s+(\w+)'
                classes = re.findall(pattern, content)
            elif file_path.endswith(('.js', '.ts')):
                pattern = r'class\s+(\w+)'
                classes = re.findall(pattern, content)
        except:
            pass
        return classes

    def _extract_imports(self, content: str, file_path: str) -> List[str]:
        """Extract import statements"""
        imports = []
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if file_path.endswith('.py'):
                    if line.startswith('import ') or line.startswith('from '):
                        imports.append(line)
                elif file_path.endswith(('.js', '.ts')):
                    if line.startswith('import ') or line.startswith('require('):
                        imports.append(line)
        except:
            pass
        return imports

    def _assess_complexity(self, content: str) -> str:
        """Assess code complexity"""
        lines = len(content.split('\n'))
        functions = len([l for l in content.split('\n') if 'def ' in l or 'function ' in l])
        classes = len([l for l in content.split('\n') if 'class ' in l])

        if lines > 500 or functions > 20 or classes > 10:
            return "high"
        elif lines > 200 or functions > 10 or classes > 5:
            return "medium"
        else:
            return "low"

    def _read_project_file(self, workspace_id: str, filename: str) -> str:
        """Read a project configuration file"""
        try:
            return fs_api.read_file(workspace_id, filename)
        except:
            return ""

    def _analyze_package_json(self, content: str) -> Dict[str, Any]:
        """Analyze package.json for project context"""
        try:
            import json
            data = json.loads(content)
            return {
                "frameworks": data.get("dependencies", {}),
                "dev_dependencies": data.get("devDependencies", {}),
                "scripts": list(data.get("scripts", {}).keys()),
                "project_type": "nodejs"
            }
        except:
            return {}

    def _analyze_requirements_txt(self, content: str) -> List[str]:
        """Analyze requirements.txt for Python dependencies"""
        libraries = []
        try:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifiers)
                    package = line.split()[0].split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                    libraries.append(package)
        except:
            pass
        return libraries

    def _extract_import_relationships(self, content: str, file_path: str) -> Dict[str, List[str]]:
        """Extract import relationships"""
        relationships = {"internal": [], "external": []}
        try:
            imports = self._extract_imports(content, file_path)
            for imp in imports:
                # Simple heuristic: relative imports are internal
                if imp.startswith('from .') or imp.startswith('import .'):
                    relationships["internal"].append(imp)
                else:
                    relationships["external"].append(imp)
        except:
            pass
        return relationships

    def _extract_function_calls(self, content: str, file_path: str) -> Dict[str, int]:
        """Extract function call patterns"""
        calls = {}
        try:
            # Simple regex to find function calls
            import re
            # Match word characters followed by parentheses
            pattern = r'(\w+)\s*\('
            matches = re.findall(pattern, content)

            # Filter out common keywords and built-ins
            exclude = {'if', 'for', 'while', 'print', 'len', 'str', 'int', 'list', 'dict', 'set', 'def', 'class', 'import', 'from', 'return', 'yield'}
            for match in matches:
                if match not in exclude and len(match) > 2:
                    calls[match] = calls.get(match, 0) + 1
        except:
            pass
        return calls

    def _build_agent_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for the Aetherium agent"""
        prompt_parts = [
            "You are an Aetherium coding assistant integrated into an advanced code workspace.",
            f"User message: {message}",
            f"Active file: {context.get('active_file', 'None')}",
            f"Workspace context: {json.dumps(context)}",
            "",
            "Provide helpful, accurate responses. If suggesting code changes, be specific about file paths and line numbers.",
            "Always consider the existing codebase structure and best practices."
        ]

        return "\n".join(prompt_parts)

    def _detect_language(self, filename: str) -> str:
        """Detect programming language from file extension"""
        if not filename:
            return "unknown"

        ext = filename.split('.')[-1].lower()
        lang_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'cs': 'csharp',
            'php': 'php',
            'rb': 'ruby',
            'go': 'go',
            'rs': 'rust',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala'
        }

        return lang_map.get(ext, 'unknown')

    def _extract_code_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Extract code suggestions from Aetherium response"""
        suggestions = []

        # Simple pattern matching for code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)

        for lang, code in code_blocks:
            suggestions.append({
                "type": "code_block",
                "language": lang or "text",
                "content": code.strip(),
                "description": "Code suggestion from Aetherium assistant"
            })

        return suggestions

# Vector Store Integration for Semantic Search
class SemanticSearch:
    def __init__(self):
        self.vector_store_url = "http://localhost:8019"

    async def search_codebase(self, query: str, workspace_id: str = None) -> List[Dict[str, Any]]:
        """Perform semantic search across the codebase"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_store_url}/search_text",
                    json={"query": query, "k": 10}
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])
                    return results
                else:
                    return []

        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def index_workspace(self, workspace_id: str, files: List[Dict[str, Any]]):
        """Index workspace files for semantic search"""
        # This would index all files in the workspace
        # Implementation depends on the specific vector store setup
        pass

# VCS Integration
class VCSIntegration:
    def __init__(self):
        self.git_available = self._check_git_available()

    def _check_git_available(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def get_diff(self, workspace_path: str) -> str:
        """Get git diff for the workspace"""
        if not self.git_available:
            return "Git not available"

        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return "Error getting diff"

    async def commit_changes(self, workspace_path: str, message: str) -> bool:
        """Commit changes with safety checks"""
        if not self.git_available:
            return False

        try:
            # Add all changes
            subprocess.run(["git", "add", "."], cwd=workspace_path, check=True)

            # Commit with message
            subprocess.run(["git", "commit", "-m", message], cwd=workspace_path, check=True)

            return True
        except subprocess.CalledProcessError:
            return False

    async def get_commit_suggestions(self, diff: str) -> List[str]:
        """Generate commit message suggestions based on diff"""
        # Simple implementation - could use Aetherium for better suggestions
        suggestions = []

        if "add" in diff.lower() or "new" in diff.lower():
            suggestions.append("feat: add new functionality")
        if "fix" in diff.lower() or "bug" in diff.lower():
            suggestions.append("fix: resolve issue")
        if "update" in diff.lower() or "change" in diff.lower():
            suggestions.append("refactor: update implementation")

        if not suggestions:
            suggestions.append("chore: update codebase")

        return suggestions

# Visualization Engine
class VisualizationEngine:
    async def generate_dependency_graph(self, workspace_id: str) -> Dict[str, Any]:
        """Generate dependency graph for the workspace"""
        # Placeholder - would analyze imports/dependencies
        return {
            "nodes": [
                {"id": "main.py", "label": "main.py", "type": "file"},
                {"id": "utils.py", "label": "utils.py", "type": "file"}
            ],
            "edges": [
                {"from": "main.py", "to": "utils.py", "label": "imports"}
            ]
        }

    async def generate_call_hierarchy(self, workspace_id: str, symbol: str) -> Dict[str, Any]:
        """Generate call hierarchy for a symbol"""
        # Placeholder - would analyze function calls
        return {
            "symbol": symbol,
            "callers": [],
            "callees": []
        }

# Terminal Management System
class TerminalSession:
    def __init__(self, session_id: str, cwd: str = None):
        self.session_id = session_id
        self.cwd = cwd or os.getcwd()
        self.process = None
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def start_process(self):
        """Start the terminal process"""
        def run_terminal():
            try:
                self.process = subprocess.Popen(
                    ['bash'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.cwd,
                    text=True,
                    bufsize=1
                )

                # Start output reading threads
                def read_stdout():
                    for line in iter(self.process.stdout.readline, ''):
                        self.output_queue.put(('stdout', line.strip()))

                def read_stderr():
                    for line in iter(self.process.stderr.readline, ''):
                        self.output_queue.put(('stderr', line.strip()))

                import threading
                threading.Thread(target=read_stdout, daemon=True).start()
                threading.Thread(target=read_stderr, daemon=True).start()

                # Wait for process to finish
                self.process.wait()

            except Exception as e:
                self.output_queue.put(('error', str(e)))

        self.executor.submit(run_terminal)

    def send_input(self, input_text: str):
        """Send input to the terminal process"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(input_text + '\n')
                self.process.stdin.flush()
            except Exception as e:
                self.output_queue.put(('error', f'Failed to send input: {e}'))

    def get_output(self):
        """Get available output"""
        outputs = []
        while not self.output_queue.empty():
            try:
                outputs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs

    def kill(self):
        """Kill the terminal process"""
        if self.process:
            try:
                self.process.terminate()
                # Wait a bit, then force kill if still running
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                print(f"Error killing process: {e}")

class TerminalManager:
    def __init__(self):
        self.sessions: Dict[str, TerminalSession] = {}

    def create_session(self, cwd: str = None) -> str:
        """Create a new terminal session"""
        session_id = f"term_{secrets.token_hex(8)}"
        session = TerminalSession(session_id, cwd)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """Get a terminal session"""
        return self.sessions.get(session_id)

    def destroy_session(self, session_id: str):
        """Destroy a terminal session"""
        if session_id in self.sessions:
            self.sessions[session_id].kill()
            del self.sessions[session_id]

# Autonomous Execution System
class AutonomousExecutionManager:
    def __init__(self):
        self.pending_approvals: Dict[str, dict] = {}

    def request_approval(self, task_id: str, command: str, context: dict) -> str:
        """Request approval for autonomous execution"""
        approval_id = f"approval_{secrets.token_hex(8)}"
        self.pending_approvals[approval_id] = {
            "task_id": task_id,
            "command": command,
            "context": context,
            "timestamp": time.time()
        }
        return approval_id

    def approve_execution(self, approval_id: str) -> bool:
        """Approve an execution request"""
        return approval_id in self.pending_approvals

    def reject_execution(self, approval_id: str) -> bool:
        """Reject an execution request"""
        if approval_id in self.pending_approvals:
            del self.pending_approvals[approval_id]
            return True
        return False

    def get_pending_approvals(self) -> List[dict]:
        """Get all pending approvals"""
        return [
            {"approval_id": aid, **data}
            for aid, data in self.pending_approvals.items()
        ]

# Developer Experience Manager
class DeveloperExperienceManager:
    def __init__(self):
        self.active_workflows: Dict[str, dict] = {}
        self.workflow_logs: Dict[str, list] = {}
        self.copilot_suggestions: Dict[str, list] = {}

    def start_workflow(self, workflow_id: str, name: str, steps: list):
        self.active_workflows[workflow_id] = {
            "name": name,
            "steps": steps,
            "current_step": 0,
            "status": "running",
            "start_time": time.time(),
            "paused": False
        }
        self.workflow_logs[workflow_id] = [{
            "timestamp": time.time(),
            "level": "info",
            "message": f"Started workflow: {name}"
        }]

    def pause_workflow(self, workflow_id: str):
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["paused"] = True
            self.add_log(workflow_id, "info", "Workflow paused by user")

    def resume_workflow(self, workflow_id: str):
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["paused"] = False
            self.add_log(workflow_id, "info", "Workflow resumed by user")

    def stop_workflow(self, workflow_id: str):
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "stopped"
            self.add_log(workflow_id, "info", "Workflow stopped by user")

    def add_log(self, workflow_id: str, level: str, message: str):
        if workflow_id not in self.workflow_logs:
            self.workflow_logs[workflow_id] = []
        self.workflow_logs[workflow_id].append({
            "timestamp": time.time(),
            "level": level,
            "message": message
        })

    def get_workflow_status(self, workflow_id: str):
        return self.active_workflows.get(workflow_id)

    def get_workflow_logs(self, workflow_id: str):
        return self.workflow_logs.get(workflow_id, [])

    def add_copilot_suggestion(self, context_id: str, suggestion: dict):
        if context_id not in self.copilot_suggestions:
            self.copilot_suggestions[context_id] = []
        self.copilot_suggestions[context_id].append({
            "timestamp": time.time(),
            **suggestion
        })
        if len(self.copilot_suggestions[context_id]) > 10:
            self.copilot_suggestions[context_id] = self.copilot_suggestions[context_id][-10:]

    def get_copilot_suggestions(self, context_id: str):
        return self.copilot_suggestions.get(context_id, [])

# Initialize services
fs_api = FileSystemAPI()
ai_runtime = AIAgentRuntime()
semantic_search = SemanticSearch()
vcs_integration = VCSIntegration()
visualization_engine = VisualizationEngine()
terminal_manager = TerminalManager()
autonomous_manager = AutonomousExecutionManager()
dev_experience_manager = DeveloperExperienceManager()

# API Endpoints
@app.post("/workspaces")
async def add_workspace(config: WorkspaceConfig):
    """Add a new workspace"""
    fs_api.add_workspace(config)
    return {"success": True, "workspace_id": config.id}

@app.post("/workspaces/project/{project_id}")
async def create_project_workspace(project_id: str):
    """Create a workspace linked to a project"""
    try:
        # Get project details from orchestrator
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:8000/api/projects/{project_id}")
            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="Project not found")

            project = response.json()

            # Create workspace config
            workspace_config = WorkspaceConfig(
                id=f"project-{project_id}",
                name=f"Project: {project['name']}",
                type="project",
                project_id=project_id
            )

            fs_api.add_workspace(workspace_config)
            return {
                "success": True,
                "workspace_id": workspace_config.id,
                "project": project
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects")
async def list_projects():
    """Get list of available projects for workspace creation"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/projects")
            if response.status_code == 200:
                projects = response.json()
                return {"projects": projects}
            else:
                return {"projects": []}
    except Exception as e:
        return {"projects": [], "error": str(e)}

@app.get("/filesystem/browse")
async def browse_filesystem(path: str = "/"):
    """Browse the file system"""
    try:
        import os
        if not os.path.exists(path):
            return {"error": "Path does not exist"}

        if not os.path.isdir(path):
            return {"error": "Path is not a directory"}

        # Security check - prevent access to sensitive directories
        sensitive_paths = ['/root', '/etc', '/var', '/usr', '/bin', '/sbin', '/boot']
        if any(path.startswith(sensitive) for sensitive in sensitive_paths):
            return {"error": "Access denied to system directories"}

        items = []
        try:
            entries = os.listdir(path)
        except PermissionError:
            return {"error": "Permission denied"}

        for entry in sorted(entries):
            if entry.startswith('.'):  # Skip hidden files
                continue

            full_path = os.path.join(path, entry)
            try:
                stat_info = os.stat(full_path)
                item = {
                    "name": entry,
                    "path": full_path,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "size": stat_info.st_size if not os.path.isdir(full_path) else None,
                    "modified": stat_info.st_mtime
                }
                items.append(item)
            except (OSError, PermissionError):
                # Skip items we can't access
                continue

        return {"path": path, "items": items}

    except Exception as e:
        return {"error": str(e)}

@app.get("/projects/{project_id}/context")
async def get_project_context(project_id: str):
    """Get comprehensive project context for Aetherium agents"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            # Get project details
            project_response = await client.get(f"http://localhost:8000/api/projects/{project_id}")
            if project_response.status_code != 200:
                raise HTTPException(status_code=404, detail="Project not found")

            project = project_response.json()

            # Get project tasks
            tasks_response = await client.get(f"http://localhost:8000/api/tasks?project_id={project_id}")
            tasks = tasks_response.json() if tasks_response.status_code == 200 else []

            # Get project repositories
            repos_response = await client.get(f"http://localhost:8000/api/repositories?project_id={project_id}")
            repositories = repos_response.json() if repos_response.status_code == 200 else []

            return {
                "project": project,
                "tasks": tasks,
                "repositories": repositories,
                "context_summary": {
                    "name": project.get("name"),
                    "description": project.get("description"),
                    "status": project.get("status"),
                    "task_count": len(tasks),
                    "repo_count": len(repositories),
                    "active_tasks": len([t for t in tasks if t.get("status") == "in_progress"])
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspaces/{workspace_id}/files")
async def get_workspace_files(workspace_id: str):
    """Get files for a workspace"""
    files = fs_api.get_workspace_files(workspace_id)
    return {"files": files}

@app.get("/workspaces/{workspace_id}/files/{file_path:path}")
async def read_workspace_file(workspace_id: str, file_path: str):
    """Read a file from a workspace"""
    content = fs_api.read_file(workspace_id, file_path)
    return {"content": content}

@app.put("/workspaces/{workspace_id}/files/{file_path:path}")
async def write_workspace_file(workspace_id: str, file_path: str, request: dict):
    """Write a file to a workspace"""
    content = request.get("content", "")
    fs_api.write_file(workspace_id, file_path, content)
    return {"success": True}

@app.post("/ai/chat")
async def ai_chat(request: dict):
    """Chat with Aetherium agent"""
    workspace_id = request.get("workspace_id", "default")
    message = request.get("message", "")
    context = request.get("context", {})

    response = await ai_runtime.chat_with_agent(workspace_id, message, context)
    return response

@app.post("/search/semantic")
async def semantic_search_endpoint(request: dict):
    """Perform semantic search"""
    query = request.get("query", "")
    workspace_id = request.get("workspace_id")

    results = await semantic_search.search_codebase(query, workspace_id)
    return {"results": results}

@app.get("/vcs/diff")
async def get_vcs_diff(workspace_path: str):
    """Get VCS diff"""
    diff = await vcs_integration.get_diff(workspace_path)
    return {"diff": diff}

@app.post("/vcs/commit")
async def commit_changes(request: dict):
    """Commit changes"""
    workspace_path = request.get("workspace_path", "")
    message = request.get("message", "Update codebase")

    success = await vcs_integration.commit_changes(workspace_path, message)
    return {"success": success}

@app.get("/vcs/commit-suggestions")
async def get_commit_suggestions(diff: str):
    """Get commit message suggestions"""
    suggestions = await vcs_integration.get_commit_suggestions(diff)
    return {"suggestions": suggestions}

@app.get("/visualization/dependency-graph/{workspace_id}")
async def get_dependency_graph(workspace_id: str):
    """Get dependency graph"""
    graph = await visualization_engine.generate_dependency_graph(workspace_id)
    return graph

@app.get("/visualization/call-hierarchy/{workspace_id}")
async def get_call_hierarchy(workspace_id: str, symbol: str):
    """Get call hierarchy"""
    hierarchy = await visualization_engine.generate_call_hierarchy(workspace_id, symbol)
    return hierarchy

# Agent Workspace Tools
@app.post("/tools/read_workspace_file")
async def tool_read_workspace_file(request: dict):
    """Tool for agents to read workspace files"""
    workspace_id = request.get("workspace_id")
    file_path = request.get("file_path")

    if not workspace_id or not file_path:
        return {"error": "workspace_id and file_path required"}

    try:
        content = fs_api.read_file(workspace_id, file_path)
        return {"content": content, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/tools/write_workspace_file")
async def tool_write_workspace_file(request: dict):
    """Tool for agents to write workspace files"""
    workspace_id = request.get("workspace_id")
    file_path = request.get("file_path")
    content = request.get("content", "")

    if not workspace_id or not file_path:
        return {"error": "workspace_id and file_path required"}

    try:
        fs_api.write_file(workspace_id, file_path, content)
        return {"success": True, "message": f"File {file_path} written successfully"}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/tools/list_workspace_files")
async def tool_list_workspace_files(request: dict):
    """Tool for agents to list workspace files"""
    workspace_id = request.get("workspace_id", "default")

    try:
        files = fs_api.get_workspace_files(workspace_id)
        return {"files": files, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/tools/run_code_in_workspace")
async def tool_run_code_in_workspace(request: dict):
    """Tool for agents to run code in workspace context"""
    code = request.get("code", "")
    language = request.get("language", "python")

    if not code.strip():
        return {"error": "Code is required"}

    try:
        # Forward to sandbox executor with workspace context
        async with httpx.AsyncClient(timeout=30.0) as client:
            exec_request = {
                "command": f"cd /workspace && python3 -c '{code.replace(chr(39), chr(92) + chr(39))}'" if language == "python" else code,
                "working_dir": "/workspace",
                "timeout": 30
            }
            response = await client.post("http://localhost:8002/execute", json=exec_request)

            if response.status_code == 200:
                result = response.json()
                return {
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "exit_code": result.get("exit_code", 0),
                    "success": True
                }
            else:
                return {"error": "Execution failed", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/tools/search_workspace")
async def tool_search_workspace(request: dict):
    """Tool for agents to search workspace files"""
    workspace_id = request.get("workspace_id", "default")
    query = request.get("query", "")

    try:
        # Use semantic search if available
        results = await semantic_search.search_codebase(query, workspace_id)
        return {"results": results, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/health")
def health():
    return {"status": "ok"}

# Extended Workspace API Endpoints

# File Operations with Safety Controls
@app.post("/api/workspace/files/move")
async def move_file(request: dict):
    """Move/rename a file with safety checks"""
    source_path = request.get("source_path", "").lstrip('/')
    dest_path = request.get("dest_path", "").lstrip('/')
    workspace_id = request.get("workspace_id", "default")

    if not source_path or not dest_path:
        raise HTTPException(status_code=400, detail="source_path and dest_path required")

    try:
        # Check if source exists
        if not os.path.exists(os.path.join(fs_api.workspaces[workspace_id].path, source_path)):
            raise HTTPException(status_code=404, detail="Source file not found")

        # Check if destination already exists
        dest_full_path = os.path.join(fs_api.workspaces[workspace_id].path, dest_path)
        if os.path.exists(dest_full_path):
            # Confirm overwrite
            confirm = request.get("confirm_overwrite", False)
            if not confirm:
                return {"requires_confirmation": True, "message": "Destination file exists"}

        # Perform move
        import shutil
        shutil.move(
            os.path.join(fs_api.workspaces[workspace_id].path, source_path),
            dest_full_path
        )

        return {"success": True, "message": f"File moved from {source_path} to {dest_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/workspace/files/{file_path:path}")
async def delete_file(file_path: str, workspace_id: str = "default", confirm: bool = False):
    """Delete a file with confirmation"""
    try:
        full_path = os.path.join(fs_api.workspaces[workspace_id].path, file_path.lstrip('/'))

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Safety check - don't delete important files
        important_files = ['package.json', 'requirements.txt', 'setup.py', 'README.md']
        if os.path.basename(full_path) in important_files and not confirm:
            return {"requires_confirmation": True, "message": "This appears to be an important file"}

        os.remove(full_path)
        return {"success": True, "message": f"File {file_path} deleted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Semantic Search
@app.post("/api/workspace/search/semantic")
async def semantic_search_workspace(request: dict):
    """Perform semantic search across workspace"""
    query = request.get("query", "")
    workspace_id = request.get("workspace_id", "default")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        results = await semantic_search.search_codebase(query, workspace_id)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Git Integration
@app.get("/api/workspace/git/status")
async def get_git_status(workspace_id: str = "default"):
    """Get git status for workspace"""
    try:
        workspace = fs_api.workspaces.get(workspace_id)
        if not workspace or workspace.type != 'local':
            raise HTTPException(status_code=400, detail="Git operations only supported for local workspaces")

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace.path,
            capture_output=True,
            text=True
        )

        return {
            "status": "clean" if result.returncode == 0 and not result.stdout.strip() else "dirty",
            "changes": result.stdout.strip().split('\n') if result.stdout.strip() else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workspace/git/commit")
async def git_commit(request: dict):
    """Commit changes to git"""
    message = request.get("message", "")
    workspace_id = request.get("workspace_id", "default")

    if not message:
        raise HTTPException(status_code=400, detail="Commit message is required")

    try:
        workspace = fs_api.workspaces.get(workspace_id)
        if not workspace or workspace.type != 'local':
            raise HTTPException(status_code=400, detail="Git operations only supported for local workspaces")

        # Add all changes
        subprocess.run(["git", "add", "."], cwd=workspace.path, check=True)

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=workspace.path,
            capture_output=True,
            text=True
        )

        return {
            "success": result.returncode == 0,
            "message": result.stdout if result.returncode == 0 else result.stderr
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Terminal Integration
@app.post("/api/terminal/sessions")
async def create_terminal_session(request: dict):
    """Create a new terminal session"""
    cwd = request.get("cwd")
    session_id = terminal_manager.create_session(cwd)

    # Start the process
    session = terminal_manager.get_session(session_id)
    session.start_process()

    return {"session_id": session_id}

@app.websocket("/ws/terminal/{session_id}")
async def terminal_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for terminal communication"""
    await manager.connect(websocket)

    session = terminal_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004)
        return

    try:
        while True:
            # Send any available output
            outputs = session.get_output()
            for output_type, content in outputs:
                await websocket.send_json({
                    "type": output_type,
                    "content": content
                })

            # Receive input
            data = await websocket.receive_json()
            if data.get("type") == "input":
                session.send_input(data.get("content", ""))

            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        terminal_manager.destroy_session(session_id)

@app.post("/api/terminal/{session_id}/kill")
async def kill_terminal_session(session_id: str):
    """Kill a terminal session"""
    terminal_manager.destroy_session(session_id)
    return {"success": True}

# Autonomous Execution
@app.post("/api/autonomous/execute")
async def request_autonomous_execution(request: dict):
    """Request autonomous command execution"""
    command = request.get("command", "")
    context = request.get("context", {})
    task_id = request.get("task_id", f"task_{secrets.token_hex(8)}")

    if not command:
        raise HTTPException(status_code=400, detail="Command is required")

    # Request approval
    approval_id = autonomous_manager.request_approval(task_id, command, context)

    # Broadcast approval request
    await manager.broadcast(json.dumps({
        "type": "approval_required",
        "approval_id": approval_id,
        "command": command,
        "context": context
    }))

    return {"approval_id": approval_id, "status": "pending_approval"}

@app.post("/api/autonomous/approve/{approval_id}")
async def approve_execution(approval_id: str):
    """Approve an autonomous execution request"""
    if autonomous_manager.approve_execution(approval_id):
        approval_data = autonomous_manager.pending_approvals[approval_id]

        # Execute the command
        try:
            result = subprocess.run(
                approval_data["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Broadcast result
            await manager.broadcast(json.dumps({
                "type": "execution_result",
                "approval_id": approval_id,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }))

            # Clean up approval
            del autonomous_manager.pending_approvals[approval_id]

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": str(e)}
    else:
        raise HTTPException(status_code=404, detail="Approval request not found")

@app.get("/api/autonomous/pending")
async def get_pending_approvals():
    """Get pending approval requests"""
    return {"approvals": autonomous_manager.get_pending_approvals()}

# Developer Experience Features
@app.post("/api/copilot/suggestions")
async def get_copilot_suggestions(request: dict):
    """Get copilot-style suggestions"""
    context = request.get("context", {})
    context_id = request.get("context_id", "default")

    # Generate suggestions based on context
    suggestions = []

    # File-based suggestions
    if context.get("active_file"):
        file_path = context["active_file"]
        if file_path.endswith('.py'):
            suggestions.extend([
                {"type": "command", "title": "Run Python file", "command": f"python {file_path}"},
                {"type": "command", "title": "Run with pytest", "command": f"python -m pytest {file_path}"},
                {"type": "code", "title": "Add main guard", "code": 'if __name__ == "__main__":\n    main()'}
            ])

    # Add suggestions to manager
    for suggestion in suggestions:
        dev_experience_manager.add_copilot_suggestion(context_id, suggestion)

    return {"suggestions": suggestions}

@app.post("/api/workflows/start")
async def start_workflow(request: dict, background_tasks: BackgroundTasks):
    """Start an automated workflow"""
    workflow_type = request.get("workflow_type")
    workflow_name = request.get("name", f"Workflow: {workflow_type}")

    if not workflow_type:
        raise HTTPException(status_code=400, detail="workflow_type is required")

    # Define workflow steps based on type
    steps = []
    if workflow_type == "python_install_run":
        steps = [
            {"name": "Install dependencies", "command": "pip install -r requirements.txt"},
            {"name": "Run application", "command": "python main.py"}
        ]
    elif workflow_type == "js_install_start":
        steps = [
            {"name": "Install dependencies", "command": "npm install"},
            {"name": "Start application", "command": "npm start"}
        ]

    if not steps:
        raise HTTPException(status_code=400, detail="Unsupported workflow type")

    workflow_id = f"workflow_{secrets.token_hex(8)}"
    dev_experience_manager.start_workflow(workflow_id, workflow_name, steps)

    # Start execution in background
    background_tasks.add_task(execute_workflow_background, workflow_id)

    await manager.broadcast(json.dumps({
        "type": "workflow_started",
        "workflow_id": workflow_id,
        "name": workflow_name
    }))

    return {"workflow_id": workflow_id, "steps": steps}

async def execute_workflow_background(workflow_id: str):
    """Execute workflow steps in background"""
    workflow = dev_experience_manager.get_workflow_status(workflow_id)
    if not workflow:
        return

    for i, step in enumerate(workflow["steps"]):
        # Check if workflow is still running
        current = dev_experience_manager.get_workflow_status(workflow_id)
        if not current or current["status"] != "running":
            break

        dev_experience_manager.active_workflows[workflow_id]["current_step"] = i
        dev_experience_manager.add_log(workflow_id, "info", f"Executing: {step['name']}")

        # Execute step
        try:
            result = subprocess.run(
                step["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            success = result.returncode == 0
            dev_experience_manager.add_log(
                workflow_id,
                "success" if success else "error",
                f"Step {step['name']}: {'completed' if success else 'failed'}"
            )

            await manager.broadcast(json.dumps({
                "type": "workflow_step_result",
                "workflow_id": workflow_id,
                "step": i,
                "success": success,
                "output": result.stdout if success else result.stderr
            }))

            if not success:
                dev_experience_manager.active_workflows[workflow_id]["status"] = "failed"
                break

        except subprocess.TimeoutExpired:
            dev_experience_manager.add_log(workflow_id, "error", f"Step {step['name']} timed out")
            dev_experience_manager.active_workflows[workflow_id]["status"] = "failed"
            break

        await asyncio.sleep(1)

    # Mark as completed
    if dev_experience_manager.active_workflows[workflow_id]["status"] == "running":
        dev_experience_manager.active_workflows[workflow_id]["status"] = "completed"
        dev_experience_manager.add_log(workflow_id, "info", "Workflow completed")

    await manager.broadcast(json.dumps({
        "type": "workflow_completed",
        "workflow_id": workflow_id,
        "status": dev_experience_manager.active_workflows[workflow_id]["status"]
    }))

@app.get("/api/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    status = dev_experience_manager.get_workflow_status(workflow_id)
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {"status": status}

@app.get("/api/workflows/{workflow_id}/logs")
async def get_workflow_logs(workflow_id: str):
    """Get workflow logs"""
    logs = dev_experience_manager.get_workflow_logs(workflow_id)
    return {"logs": logs}

# Aetherium Integration for Safe Edits and Debugging
@app.post("/api/ai/safe-edit")
async def ai_safe_edit(request: dict, background_tasks: BackgroundTasks):
    """Request Aetherium safe code edit"""
    file_path = request.get("file_path", "").lstrip('/')
    edit_description = request.get("description", "")
    workspace_id = request.get("workspace_id", "default")

    if not file_path or not edit_description:
        raise HTTPException(status_code=400, detail="file_path and description required")

    # Read current content
    try:
        current_content = fs_api.read_file(workspace_id, file_path)
    except:
        raise HTTPException(status_code=404, detail="File not found")

    # Create edit task
    task_id = f"edit_{secrets.token_hex(8)}"

    # Generate Aetherium edit (simplified)
    prompt = f"Edit this code safely:\n\n{current_content}\n\nDescription: {edit_description}\n\nProvide the modified code:"

    try:
        from providers.nim_adapter import NIMAdapter
        adapter = NIMAdapter()
        messages = [{"role": "user", "content": prompt}]
        response = adapter.call_model(messages)

        proposed_content = response.text if hasattr(response, 'text') else str(response)

        # Generate diff
        diff = list(difflib.unified_diff(
            current_content.splitlines(keepends=True),
            proposed_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        ))

        await manager.broadcast(json.dumps({
            "type": "ai_edit_result",
            "task_id": task_id,
            "file_path": file_path,
            "diff": ''.join(diff),
            "has_changes": len(diff) > 0
        }))

        return {
            "task_id": task_id,
            "diff": ''.join(diff),
            "has_changes": len(diff) > 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aetherium edit failed: {str(e)}")

@app.post("/api/ai/debug")
async def ai_debug(request: dict):
    """Request Aetherium debugging assistance"""
    error_message = request.get("error_message", "")
    code_context = request.get("code_context", "")

    if not error_message:
        raise HTTPException(status_code=400, detail="error_message required")

    # Generate debug analysis
    prompt = f"Debug this error:\n\nError: {error_message}\n\nCode context:\n{code_context}\n\nProvide analysis and fix suggestions:"

    try:
        from providers.nim_adapter import NIMAdapter
        adapter = NIMAdapter()
        messages = [{"role": "user", "content": prompt}]
        response = adapter.call_model(messages)

        analysis = response.text if hasattr(response, 'text') else str(response)

        return {"analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aetherium debug failed: {str(e)}")