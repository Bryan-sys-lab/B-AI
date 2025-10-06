from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from sqlalchemy import select, update, func  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from orchestrator.database import async_session, init_db, Project, Task, Subtask, Feedback, Provider, ProviderMetrics, Agent, AgentMetrics, Repository, RepositoryFile, SecurityPolicy, SecurityScan, ObservabilityMetric, Prompt, IntelligenceAnalysis, Integration, PromptCache  # noqa: E402
from orchestrator.master_agent import MasterAgent  # noqa: E402
# Task classifier agent will be called via HTTP API
from typing import Optional, Dict, Any  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import secrets  # noqa: E402
import httpx  # noqa: E402
import time  # noqa: E402
from typing import Dict, Any, Optional  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import queue  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES, SYSTEM_ABOUT_RESPONSES  # noqa: E402

# Create FastAPI app instance early so decorators can use it
app = FastAPI()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        logger.info(f"Broadcasting WebSocket message to {len(self.active_connections)} connections: {message[:200]}...")
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                logger.debug(f"Sent message to connection")
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Terminal Session Manager
class TerminalSession:
    def __init__(self, session_id: str, cwd: str = None):
        self.session_id = session_id
        self.cwd = cwd or os.getcwd()
        self.process = None
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.is_running = False
        self.websocket = None
        self.thread = None

    def start_process(self):
        """Start the terminal process"""
        try:
            # Use bash or zsh if available, fallback to sh
            shell = os.environ.get('SHELL', '/bin/bash')
            if not os.path.exists(shell):
                shell = '/bin/bash' if os.path.exists('/bin/bash') else '/bin/sh'

            self.process = subprocess.Popen(
                [shell],
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.is_running = True

            # Start output reading thread
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()

            return True
        except Exception as e:
            logger.error(f"Failed to start terminal process: {e}")
            return False

    def _read_output(self):
        """Read output from the process and put it in queue"""
        try:
            while self.is_running and self.process.poll() is None:
                # Read stdout
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        self.output_queue.put(('stdout', line))

                # Read stderr
                if self.process.stderr:
                    line = self.process.stderr.readline()
                    if line:
                        self.output_queue.put(('stderr', line))

                time.sleep(0.01)  # Small delay to prevent busy waiting

            # Process has ended
            self.is_running = False
            exit_code = self.process.returncode if self.process else -1
            self.output_queue.put(('exit', f'Process exited with code {exit_code}\n'))

        except Exception as e:
            logger.error(f"Error reading terminal output: {e}")
            self.is_running = False

    def send_input(self, input_text: str):
        """Send input to the process"""
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(input_text + '\n')
                self.process.stdin.flush()
                return True
            except Exception as e:
                logger.error(f"Failed to send input to terminal: {e}")
                return False
        return False

    def kill_process(self):
        """Kill the terminal process"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                # Wait a bit, then kill if still running
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                return True
            except Exception as e:
                logger.error(f"Failed to kill terminal process: {e}")
                return False
        return False

    def get_output(self):
        """Get available output from the queue"""
        outputs = []
        while not self.output_queue.empty():
            try:
                outputs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs

class TerminalManager:
    def __init__(self):
        self.sessions: Dict[str, TerminalSession] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

    def create_session(self, session_id: str = None, cwd: str = None) -> str:
        """Create a new terminal session"""
        if session_id is None:
            session_id = f"term_{secrets.token_hex(8)}"

        session = TerminalSession(session_id, cwd)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """Get a terminal session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Delete a terminal session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.kill_process()
            del self.sessions[session_id]

        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]

    def list_sessions(self) -> list:
        """List all active terminal sessions"""
        return [
            {
                "session_id": session_id,
                "cwd": session.cwd,
                "is_running": session.is_running,
                "has_websocket": session_id in self.websocket_connections
            }
            for session_id, session in self.sessions.items()
        ]

    def connect_websocket(self, session_id: str, websocket: WebSocket):
        """Connect a WebSocket to a terminal session"""
        self.websocket_connections[session_id] = websocket

    def disconnect_websocket(self, session_id: str):
        """Disconnect WebSocket from terminal session"""
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]

terminal_manager = TerminalManager()

# Background task to stream terminal output
async def stream_terminal_output():
    """Background task to stream terminal output to WebSocket connections"""
    while True:
        try:
            # Check all active sessions for output
            for session_id, session in list(terminal_manager.sessions.items()):
                if session_id in terminal_manager.websocket_connections:
                    websocket = terminal_manager.websocket_connections[session_id]

                    # Get available output
                    outputs = session.get_output()
                    if outputs:
                        # Send outputs to WebSocket
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "output",
                                "data": outputs
                            }))
                        except Exception as e:
                            logger.warning(f"Failed to send terminal output to WebSocket {session_id}: {e}")
                            # Remove broken connection
                            terminal_manager.disconnect_websocket(session_id)

            await asyncio.sleep(0.1)  # Check every 100ms

        except Exception as e:
            logger.error(f"Error in terminal output streaming: {e}")
            await asyncio.sleep(1)  # Wait longer on error

# Start the terminal output streaming task
@app.on_event("startup")
async def start_terminal_streaming():
    """Start the terminal output streaming background task"""
    asyncio.create_task(stream_terminal_output())

    # Initialize default providers
    await initialize_default_providers()

# Aetherium Autonomous Execution
class AutonomousExecutionManager:
    def __init__(self):
        self.pending_approvals: Dict[str, dict] = {}  # task_id -> approval data
        self.active_executions: Dict[str, dict] = {}  # task_id -> execution data

    def request_approval(self, task_id: str, command: str, context: dict) -> str:
        """Request user approval for a command execution"""
        approval_id = f"approval_{secrets.token_hex(8)}"
        self.pending_approvals[approval_id] = {
            "task_id": task_id,
            "command": command,
            "context": context,
            "timestamp": time.time(),
            "status": "pending"
        }
        return approval_id

    def approve_execution(self, approval_id: str) -> bool:
        """Approve a pending execution"""
        if approval_id in self.pending_approvals:
            approval = self.pending_approvals[approval_id]
            approval["status"] = "approved"
            return True
        return False

    def reject_execution(self, approval_id: str) -> bool:
        """Reject a pending execution"""
        if approval_id in self.pending_approvals:
            approval = self.pending_approvals[approval_id]
            approval["status"] = "rejected"
            return True
        return False

    def get_pending_approvals(self) -> list:
        """Get all pending approvals"""
        return [
            {"approval_id": aid, **data}
            for aid, data in self.pending_approvals.items()
            if data["status"] == "pending"
        ]

autonomous_manager = AutonomousExecutionManager()

def detect_project_commands(description: str, context: dict = None) -> list:
    """Detect potential build/run/test commands based on project type and description"""
    commands = []
    lower_desc = description.lower()

    # Detect project type from files or context
    project_type = "unknown"
    if context and context.get("files"):
        files = context["files"]
        if any(f.endswith('.py') for f in files):
            project_type = "python"
        elif any(f.endswith('.js') or f.endswith('.ts') for f in files):
            project_type = "javascript"
        elif any(f.endswith('.java') for f in files):
            project_type = "java"
        elif any(f.endswith('.go') for f in files):
            project_type = "go"
        elif any(f.endswith('.rs') for f in files):
            project_type = "rust"
        elif any(f.endswith('.cpp') or f.endswith('.c') for f in files):
            project_type = "cpp"

    # Common command patterns
    if "run" in lower_desc or "execute" in lower_desc or "start" in lower_desc:
        if project_type == "python":
            commands.append({"command": "python main.py", "type": "run", "description": "Run Python application"})
            if os.path.exists("requirements.txt"):
                commands.append({"command": "pip install -r requirements.txt", "type": "install", "description": "Install Python dependencies"})
        elif project_type == "javascript":
            commands.append({"command": "npm start", "type": "run", "description": "Start JavaScript application"})
            if os.path.exists("package.json"):
                commands.append({"command": "npm install", "type": "install", "description": "Install Node.js dependencies"})
        elif project_type == "java":
            commands.append({"command": "mvn spring-boot:run", "type": "run", "description": "Run Spring Boot application"})
            commands.append({"command": "gradle bootRun", "type": "run", "description": "Run Gradle application"})

    if "test" in lower_desc:
        if project_type == "python":
            commands.append({"command": "python -m pytest", "type": "test", "description": "Run Python tests"})
            commands.append({"command": "python -m unittest", "type": "test", "description": "Run Python unit tests"})
        elif project_type == "javascript":
            commands.append({"command": "npm test", "type": "test", "description": "Run JavaScript tests"})
        elif project_type == "java":
            commands.append({"command": "mvn test", "type": "test", "description": "Run Maven tests"})
            commands.append({"command": "gradle test", "type": "test", "description": "Run Gradle tests"})

    if "build" in lower_desc or "compile" in lower_desc:
        if project_type == "python":
            commands.append({"command": "python setup.py build", "type": "build", "description": "Build Python package"})
        elif project_type == "javascript":
            commands.append({"command": "npm run build", "type": "build", "description": "Build JavaScript application"})
        elif project_type == "java":
            commands.append({"command": "mvn compile", "type": "build", "description": "Compile Java project"})
            commands.append({"command": "gradle build", "type": "build", "description": "Build Gradle project"})
        elif project_type == "go":
            commands.append({"command": "go build", "type": "build", "description": "Build Go application"})
        elif project_type == "rust":
            commands.append({"command": "cargo build", "type": "build", "description": "Build Rust project"})
        elif project_type == "cpp":
            commands.append({"command": "make", "type": "build", "description": "Build C/C++ project"})

    return commands

async def execute_command_with_monitoring(command: str, terminal_session_id: str = None) -> dict:
    """Execute a command and monitor its output for errors and suggestions"""
    try:
        # Create a terminal session if not provided
        if not terminal_session_id:
            terminal_session_id = terminal_manager.create_session()
            session = terminal_manager.get_session(terminal_session_id)
            session.start_process()
        else:
            session = terminal_manager.get_session(terminal_session_id)

        if not session:
            return {"success": False, "error": "Terminal session not found"}

        # Send the command
        session.send_input(command)

        # Wait a bit for output
        await asyncio.sleep(2)

        # Get output
        outputs = session.get_output()

        # Analyze output for errors and suggestions
        stdout_lines = []
        stderr_lines = []
        exit_code = None

        for output_type, line in outputs:
            if output_type == "stdout":
                stdout_lines.append(line)
            elif output_type == "stderr":
                stderr_lines.append(line)
            elif output_type == "exit":
                # Extract exit code
                import re
                match = re.search(r'exited with code (\d+)', line)
                if match:
                    exit_code = int(match.group(1))

        # Analyze for common errors and suggestions
        suggestions = []
        has_errors = exit_code != 0 if exit_code is not None else bool(stderr_lines)

        # DEBUG: Log error analysis details
        error_analysis = {
            "exit_code": exit_code,
            "has_stderr": bool(stderr_lines),
            "has_stdout": bool(stdout_lines),
            "stderr_length": len(''.join(stderr_lines)),
            "stdout_length": len(''.join(stdout_lines)),
            "total_error_text_length": len('\n'.join(stderr_lines + stdout_lines))
        }

        structured_logger.log_event("error_analysis_start", {
            "task_id": "command_monitoring",
            "command": command[:100] + "..." if len(command) > 100 else command,
            **error_analysis
        })

        if has_errors:
            error_text = '\n'.join(stderr_lines + stdout_lines)

            # DEBUG: Log unrecognized error patterns
            recognized_patterns = [
                'ModuleNotFoundError', 'ImportError', 'Cannot find module',
                'package does not exist', 'cannot find symbol', 'error', 'failed'
            ]

            has_recognized_pattern = any(pattern.lower() in error_text.lower() for pattern in recognized_patterns)

            if not has_recognized_pattern and len(error_text.strip()) > 10:
                structured_logger.log_event("unrecognized_error_pattern", {
                    "task_id": "command_monitoring",
                    "command": command[:100] + "..." if len(command) > 100 else command,
                    "error_text_preview": error_text[:300] + "..." if len(error_text) > 300 else error_text,
                    "issue": "Error text doesn't match known patterns - may need additional error categorization",
                    "recommendation": "Add pattern matching for this error type"
                })
                logger.warning(f"ORCHESTRATOR: Unrecognized error pattern in command output: {error_text[:100]}...")

            # Python errors
            if 'ModuleNotFoundError' in error_text or 'ImportError' in error_text:
                suggestions.append({
                    "type": "install_dependency",
                    "description": "Missing Python module. Try installing dependencies.",
                    "commands": ["pip install -r requirements.txt", "pip install <missing_module>"]
                })

            # Node.js errors
            if 'Cannot find module' in error_text:
                suggestions.append({
                    "type": "install_dependency",
                    "description": "Missing Node.js module. Try installing dependencies.",
                    "commands": ["npm install"]
                })

            # Java errors
            if 'package does not exist' in error_text or 'cannot find symbol' in error_text:
                suggestions.append({
                    "type": "build_fix",
                    "description": "Java compilation error. Check imports and dependencies.",
                    "commands": ["mvn clean compile", "gradle build"]
                })

            # Generic build errors
            if 'error' in error_text.lower() or 'failed' in error_text.lower():
                suggestions.append({
                    "type": "check_logs",
                    "description": "Check error logs for more details.",
                    "commands": []
                })

        return {
            "success": exit_code == 0 if exit_code is not None else not has_errors,
            "exit_code": exit_code,
            "stdout": ''.join(stdout_lines),
            "stderr": ''.join(stderr_lines),
            "has_errors": has_errors,
            "suggestions": suggestions,
            "terminal_session_id": terminal_session_id
        }

    except Exception as e:
        logger.error(f"Error executing command with monitoring: {e}")
        return {"success": False, "error": str(e)}

# Developer Experience Endpoints
@app.post("/api/copilot/suggestions")
async def get_copilot_suggestions(request: dict):
    """Get copilot-style suggestions based on context"""
    try:
        context = request.get("context", {})
        context_id = request.get("context_id", "default")

        # Generate suggestions
        suggestions = generate_copilot_suggestions(context)

        # Store suggestions
        for suggestion in suggestions:
            dev_experience_manager.add_copilot_suggestion(context_id, suggestion)

        return {"suggestions": suggestions}

    except Exception as e:
        logger.error(f"Error getting copilot suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get suggestions")

@app.get("/api/copilot/suggestions/{context_id}")
async def get_stored_copilot_suggestions(context_id: str):
    """Get stored copilot suggestions for a context"""
    try:
        suggestions = dev_experience_manager.get_copilot_suggestions(context_id)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting stored suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stored suggestions")

@app.post("/api/workflows/start")
async def start_workflow(request: dict, background_tasks: BackgroundTasks):
    """Start an automated workflow"""
    try:
        workflow_type = request.get("workflow_type")
        workflow_name = request.get("name", f"Workflow: {workflow_type}")
        context = request.get("context", {})

        if not workflow_type:
            raise HTTPException(status_code=400, detail="workflow_type is required")

        # Create workflow steps
        steps = create_workflow_steps(workflow_type, context)

        if not steps:
            raise HTTPException(status_code=400, detail="Unsupported workflow type")

        # Generate workflow ID
        workflow_id = f"workflow_{secrets.token_hex(8)}"

        # Start workflow
        dev_experience_manager.start_workflow(workflow_id, workflow_name, steps)

        # Start execution in background
        background_tasks.add_task(execute_workflow, workflow_id)

        # Broadcast workflow start
        await manager.broadcast(json.dumps({
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "name": workflow_name,
            "steps": len(steps)
        }))

        return {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "steps": steps
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to start workflow")

@app.post("/api/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    """Pause a running workflow"""
    try:
        dev_experience_manager.pause_workflow(workflow_id)

        # Broadcast pause
        await manager.broadcast(json.dumps({
            "type": "workflow_paused",
            "workflow_id": workflow_id
        }))

        return {"message": "Workflow paused"}

    except Exception as e:
        logger.error(f"Error pausing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause workflow")

@app.post("/api/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    """Resume a paused workflow"""
    try:
        dev_experience_manager.resume_workflow(workflow_id)

        # Continue execution in background
        background_tasks.add_task(execute_workflow, workflow_id)

        # Broadcast resume
        await manager.broadcast(json.dumps({
            "type": "workflow_resumed",
            "workflow_id": workflow_id
        }))

        return {"message": "Workflow resumed"}

    except Exception as e:
        logger.error(f"Error resuming workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume workflow")

@app.post("/api/workflows/{workflow_id}/stop")
async def stop_workflow(workflow_id: str):
    """Stop a workflow"""
    try:
        dev_experience_manager.stop_workflow(workflow_id)

        # Broadcast stop
        await manager.broadcast(json.dumps({
            "type": "workflow_stopped",
            "workflow_id": workflow_id
        }))

        return {"message": "Workflow stopped"}

    except Exception as e:
        logger.error(f"Error stopping workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop workflow")

@app.get("/api/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow"""
    try:
        status = dev_experience_manager.get_workflow_status(workflow_id)
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {"status": status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow status")

@app.get("/api/workflows/{workflow_id}/logs")
async def get_workflow_logs(workflow_id: str):
    """Get logs for a workflow"""
    try:
        logs = dev_experience_manager.get_workflow_logs(workflow_id)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error getting workflow logs {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow logs")

@app.get("/api/workflows/active")
async def get_active_workflows():
    """Get all active workflows"""
    try:
        workflows = []
        for workflow_id, workflow_data in dev_experience_manager.active_workflows.items():
            if workflow_data["status"] == "running":
                workflows.append({
                    "workflow_id": workflow_id,
                    **workflow_data
                })

        return {"workflows": workflows}

    except Exception as e:
        logger.error(f"Error getting active workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active workflows")

async def execute_workflow(workflow_id: str):
    """Execute a workflow step by step"""
    try:
        workflow = dev_experience_manager.get_workflow_status(workflow_id)
        if not workflow or workflow["status"] != "running":
            return

        steps = workflow["steps"]
        terminal_session_id = None

        for i, step in enumerate(steps):
            # Check if workflow is still running and not paused
            current_workflow = dev_experience_manager.get_workflow_status(workflow_id)
            if not current_workflow or current_workflow["status"] != "running":
                break

            if current_workflow.get("paused"):
                # Wait for resume
                while True:
                    await asyncio.sleep(1)
                    current = dev_experience_manager.get_workflow_status(workflow_id)
                    if not current or current["status"] != "running":
                        return
                    if not current.get("paused"):
                        break

            # Update current step
            dev_experience_manager.active_workflows[workflow_id]["current_step"] = i

            # Broadcast step start
            await manager.broadcast(json.dumps({
                "type": "workflow_step_started",
                "workflow_id": workflow_id,
                "step": i,
                "name": step["name"]
            }))

            # Execute step
            result = await execute_workflow_step(workflow_id, step, terminal_session_id)

            # Store terminal session for subsequent steps
            if result.get("terminal_session_id"):
                terminal_session_id = result["terminal_session_id"]

            # Broadcast step result
            await manager.broadcast(json.dumps({
                "type": "workflow_step_completed",
                "workflow_id": workflow_id,
                "step": i,
                "name": step["name"],
                "success": result["success"],
                "error": result.get("error")
            }))

            # If step failed, stop workflow unless it should continue
            if not result["success"]:
                dev_experience_manager.active_workflows[workflow_id]["status"] = "failed"
                dev_experience_manager.add_log(workflow_id, "error", f"Workflow failed at step: {step['name']}")
                break

            # Small delay between steps
            await asyncio.sleep(0.5)

        # Mark workflow as completed
        if dev_experience_manager.active_workflows[workflow_id]["status"] == "running":
            dev_experience_manager.active_workflows[workflow_id]["status"] = "completed"
            dev_experience_manager.add_log(workflow_id, "info", "Workflow completed successfully")

        # Broadcast completion
        await manager.broadcast(json.dumps({
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "status": dev_experience_manager.active_workflows[workflow_id]["status"]
        }))

    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        dev_experience_manager.active_workflows[workflow_id]["status"] = "error"
        dev_experience_manager.add_log(workflow_id, "error", f"Workflow execution error: {str(e)}")

# Aetherium System Integration for Safe Code Edits and Debugging
@app.post("/api/ai/safe-edit")
async def ai_safe_edit(request: dict, background_tasks: BackgroundTasks):
    """Request Aetherium to make safe code edits with diff preview"""
    try:
        file_path = request.get("file_path", "").lstrip('/')
        edit_description = request.get("description", "").strip()
        context = request.get("context", {})

        if not file_path or not edit_description:
            raise HTTPException(status_code=400, detail="file_path and description are required")

        # Read current file content
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path)

        # Security check
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        current_content = ""
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                current_content = f.read()

        # Create a task for Aetherium code editing
        task_data = {
            "description": f"Safe code edit: {edit_description}",
            "context": {
                "file_path": file_path,
                "current_content": current_content,
                "edit_description": edit_description,
                "safe_edit": True,
                **context
            }
        }

        # Create task in database
        async with async_session() as session:
            new_task = Task(
                user_id="default",
                description=task_data["description"],
                context=task_data["context"]
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)

        # Start Aetherium editing process in background
        background_tasks.add_task(process_ai_safe_edit, new_task.id)

        # Broadcast edit request
        await manager.broadcast(json.dumps({
            "type": "ai_edit_requested",
            "task_id": new_task.id,
            "file_path": file_path,
            "description": edit_description
        }))

        return {
            "task_id": new_task.id,
            "message": "Aetherium safe edit requested",
            "file_path": file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting Aetherium safe edit: {e}")
        raise HTTPException(status_code=500, detail="Failed to request Aetherium edit")

@app.post("/api/ai/debug")
async def ai_debug(request: dict, background_tasks: BackgroundTasks):
    """Request Aetherium debugging assistance"""
    try:
        error_message = request.get("error_message", "").strip()
        code_context = request.get("code_context", "")
        file_path = request.get("file_path", "")
        language = request.get("language", "python")

        if not error_message:
            raise HTTPException(status_code=400, detail="error_message is required")

        # Create debugging task
        task_description = f"Debug: {error_message[:100]}..."
        task_data = {
            "description": task_description,
            "context": {
                "error_message": error_message,
                "code_context": code_context,
                "file_path": file_path,
                "language": language,
                "debug_task": True
            }
        }

        # Create task in database
        async with async_session() as session:
            new_task = Task(
                user_id="default",
                description=task_description,
                context=task_data["context"]
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)

        # Start debugging process in background
        background_tasks.add_task(process_ai_debug, new_task.id)

        # Broadcast debug request
        await manager.broadcast(json.dumps({
            "type": "ai_debug_requested",
            "task_id": new_task.id,
            "error_message": error_message[:100]
        }))

        return {
            "task_id": new_task.id,
            "message": "Aetherium debugging assistance requested"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting Aetherium debug: {e}")
        raise HTTPException(status_code=500, detail="Failed to request Aetherium debug")

@app.post("/api/ai/analyze-code")
async def ai_analyze_code(request: dict, background_tasks: BackgroundTasks):
    """Request Aetherium code analysis"""
    try:
        file_path = request.get("file_path", "").lstrip('/')
        analysis_type = request.get("analysis_type", "general")  # general, security, performance, style

        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")

        # Read file content
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path)

        # Security check
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        with open(full_path, 'r', encoding='utf-8') as f:
            code_content = f.read()

        # Create analysis task
        task_description = f"Code analysis ({analysis_type}): {file_path}"
        task_data = {
            "description": task_description,
            "context": {
                "file_path": file_path,
                "code_content": code_content,
                "analysis_type": analysis_type,
                "analysis_task": True
            }
        }

        # Create task in database
        async with async_session() as session:
            new_task = Task(
                user_id="default",
                description=task_description,
                context=task_data["context"]
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)

        # Start analysis process in background
        background_tasks.add_task(process_ai_analysis, new_task.id)

        # Broadcast analysis request
        await manager.broadcast(json.dumps({
            "type": "ai_analysis_requested",
            "task_id": new_task.id,
            "file_path": file_path,
            "analysis_type": analysis_type
        }))

        return {
            "task_id": new_task.id,
            "message": f"Aetherium {analysis_type} analysis requested for {file_path}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting Aetherium code analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to request Aetherium analysis")

async def process_ai_safe_edit(task_id: str):
    """Process Aetherium safe edit request"""
    try:
        # Get task details
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                return

            context = task.context or {}
            file_path = context.get("file_path")
            current_content = context.get("current_content", "")
            edit_description = context.get("edit_description", "")

        # Use Aetherium agent to generate edit
        # For now, use the fix_implementation agent
        edit_prompt = f"""
        Please make the following code edit safely:

        File: {file_path}
        Current content:
        {current_content}

        Edit description: {edit_description}

        Please provide:
        1. The modified code
        2. A brief explanation of the changes
        3. Any potential risks or considerations
        """

        # Call Aetherium agent (simplified - in real implementation would use proper agent)
        try:
            from providers.nim_adapter import NIMAdapter
            adapter = NIMAdapter()

            messages = [{"role": "user", "content": edit_prompt}]
            ai_response = adapter.call_model(messages)

            if hasattr(ai_response, 'text'):
                response_text = ai_response.text.strip()
            else:
                response_text = str(ai_response).strip()

            # Parse Aetherium response (simplified parsing)
            # In real implementation, would have structured output
            proposed_content = current_content  # Placeholder
            explanation = response_text

            # Generate diff
            import difflib
            diff = list(difflib.unified_diff(
                current_content.splitlines(keepends=True),
                proposed_content.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=''
            ))

            # Update task with results
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {
                        "proposed_content": proposed_content,
                        "diff": ''.join(diff),
                        "explanation": explanation,
                        "has_changes": len(diff) > 0
                    }
                    task.status = "completed"
                    await session.commit()

            # Broadcast edit result
            await manager.broadcast(json.dumps({
                "type": "ai_edit_result",
                "task_id": task_id,
                "file_path": file_path,
                "has_changes": len(diff) > 0,
                "diff_available": True
            }))

        except Exception as e:
            logger.error(f"Aetherium edit generation failed: {e}")
            # Update task with error
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {"error": f"Aetherium edit failed: {str(e)}"}
                    task.status = "failed"
                    await session.commit()

    except Exception as e:
        logger.error(f"Error processing Aetherium safe edit {task_id}: {e}")

async def process_ai_debug(task_id: str):
    """Process Aetherium debug request"""
    try:
        # Get task details
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                return

            context = task.context or {}
            error_message = context.get("error_message", "")
            code_context = context.get("code_context", "")
            language = context.get("language", "python")

        # Generate debug analysis
        debug_prompt = f"""
        Please analyze this error and provide debugging assistance:

        Language: {language}
        Error: {error_message}

        Code context:
        {code_context}

        Please provide:
        1. Root cause analysis
        2. Suggested fixes
        3. Debugging steps
        4. Prevention tips
        """

        try:
            from providers.nim_adapter import NIMAdapter
            adapter = NIMAdapter()

            messages = [{"role": "user", "content": debug_prompt}]
            ai_response = adapter.call_model(messages)

            if hasattr(ai_response, 'text'):
                analysis = ai_response.text.strip()
            else:
                analysis = str(ai_response).strip()

            # Update task with analysis
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {
                        "analysis": analysis,
                        "error_message": error_message,
                        "language": language
                    }
                    task.status = "completed"
                    await session.commit()

            # Broadcast debug result
            await manager.broadcast(json.dumps({
                "type": "ai_debug_result",
                "task_id": task_id,
                "error_message": error_message[:100]
            }))

        except Exception as e:
            logger.error(f"Aetherium debug analysis failed: {e}")
            # Update task with error
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {"error": f"Aetherium debug analysis failed: {str(e)}"}
                    task.status = "failed"
                    await session.commit()

    except Exception as e:
        logger.error(f"Error processing Aetherium debug {task_id}: {e}")

async def process_ai_analysis(task_id: str):
    """Process Aetherium code analysis request"""
    try:
        # Get task details
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                return

            context = task.context or {}
            file_path = context.get("file_path", "")
            code_content = context.get("code_content", "")
            analysis_type = context.get("analysis_type", "general")

        # Generate analysis prompt based on type
        if analysis_type == "security":
            analysis_prompt = f"Perform a security analysis of this code:\n\n{code_content}"
        elif analysis_type == "performance":
            analysis_prompt = f"Analyze the performance characteristics of this code:\n\n{code_content}"
        elif analysis_type == "style":
            analysis_prompt = f"Review the code style and provide improvement suggestions:\n\n{code_content}"
        else:
            analysis_prompt = f"Provide a general analysis of this code:\n\n{code_content}"

        try:
            from providers.nim_adapter import NIMAdapter
            adapter = NIMAdapter()

            messages = [{"role": "user", "content": analysis_prompt}]
            ai_response = adapter.call_model(messages)

            if hasattr(ai_response, 'text'):
                analysis = ai_response.text.strip()
            else:
                analysis = str(ai_response).strip()

            # Update task with analysis
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {
                        "analysis": analysis,
                        "file_path": file_path,
                        "analysis_type": analysis_type
                    }
                    task.status = "completed"
                    await session.commit()

            # Broadcast analysis result
            await manager.broadcast(json.dumps({
                "type": "ai_analysis_result",
                "task_id": task_id,
                "file_path": file_path,
                "analysis_type": analysis_type
            }))

        except Exception as e:
            logger.error(f"Aetherium code analysis failed: {e}")
            # Update task with error
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.output = {"error": f"Aetherium analysis failed: {str(e)}"}
                    task.status = "failed"
                    await session.commit()

    except Exception as e:
        logger.error(f"Error processing Aetherium analysis {task_id}: {e}")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging with more detailed format and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/orchestrator.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add structured logging for monitoring
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger('structured')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/structured.log', mode='a')
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_event(self, event_type: str, data=None, **kwargs):
        """Log structured events for monitoring"""
        event_data = {
            "event_type": event_type,
            "timestamp": time.time(),
        }
        if data:
            event_data.update(data)
        event_data.update(kwargs)
        self.logger.info(json.dumps(event_data))

structured_logger = StructuredLogger()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    print(f"REQUEST: {request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s"
    )
    return response

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: str
    created_at: str
    updated_at: str
    task_count: int = 0

class TaskCreate(BaseModel):
    description: str
    project_id: Optional[str] = None
    context: Optional[dict] = None
    selectedAgent: Optional[str] = None

class TaskResponse(BaseModel):
    id: str
    project_id: Optional[str] = None
    user_id: str
    description: str
    status: str
    plan: Optional[dict] = None
    output: Optional[dict] = None
    context: Optional[dict] = None
    subtasks: list = []
    created_at: str

class FeedbackCreate(BaseModel):
    task_id: str
    subtask_id: Optional[str] = None
    rating: int
    comments: Optional[str] = None
    improvement_suggestions: Optional[dict] = None

class ProviderResponse(BaseModel):
    id: str
    name: str
    type: str
    purpose: str
    models: list
    status: str
    config: Optional[dict] = None
    created_at: str
    updated_at: str

class ProviderMetricsResponse(BaseModel):
    provider_id: str
    latency: float
    success_rate: float
    total_requests: int
    active_requests: int
    cost_estimate: float
    tokens_used: int
    last_used: Optional[str] = None

class AgentResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str
    status: str
    health: str
    current_task: Optional[str] = None
    config: Optional[dict] = None
    created_at: str
    updated_at: str

class AgentMetricsResponse(BaseModel):
    agent_id: str
    tasks_completed: int
    success_rate: float
    average_response_time: float
    error_count: int
    last_activity: Optional[str] = None

class RepositoryResponse(BaseModel):
    id: str
    name: str
    url: str
    branch: str
    status: str
    description: Optional[str] = None
    language: Optional[str] = None
    size: Optional[str] = None
    commits: int
    contributors: int
    last_sync: Optional[str] = None
    config: Optional[dict] = None
    created_at: str
    updated_at: str

class RepositoryFileResponse(BaseModel):
    id: str
    repository_id: str
    path: str
    name: str
    type: str
    size: Optional[str] = None
    last_modified: Optional[str] = None
    content: Optional[dict] = None
    created_at: str

class SecurityPolicyResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    severity: str
    enabled: bool
    config: Optional[dict] = None
    created_at: str
    updated_at: str

class SecurityScanResponse(BaseModel):
    id: str
    target_type: str
    target_id: str
    status: str
    findings: Optional[dict] = None
    score: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str

class ObservabilityMetricResponse(BaseModel):
    id: str
    name: str
    category: str
    value: float
    unit: str
    tags: Optional[dict] = None
    timestamp: str

class PromptResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    category: str
    content: str
    variables: Optional[dict] = None
    tags: Optional[list] = None
    usage_count: int
    success_rate: float
    created_at: str
    updated_at: str

class IntelligenceAnalysisResponse(BaseModel):
    id: str
    target_type: str
    target_id: str
    analysis_type: str
    result: dict
    confidence: float
    created_at: str

class IntegrationResponse(BaseModel):
    id: str
    name: str
    type: str
    description: str
    config: dict
    status: str
    last_sync: Optional[str] = None
    created_at: str
    updated_at: str

# Agent endpoint configuration for different deployment patterns
class AgentEndpointManager:
    def __init__(self):
        # Default localhost development endpoints
        self.endpoints = {
            "master_agent": "http://localhost:8000/execute",
            "fix_implementation_agent": "http://localhost:8004/execute",
            "debugger_agent": "http://localhost:8005/execute",
            "review_agent": "http://localhost:8006/execute",
            "testing_agent": "http://localhost:8007/execute",
            "security_agent": "http://localhost:8008/execute",
            "performance_agent": "http://localhost:8009/execute",
            "deployment_agent": "http://localhost:8017/execute",
            "monitoring_agent": "http://localhost:8018/execute",
            "feedback_agent": "http://localhost:8010/execute",
            "comparator_service": "http://localhost:8012/execute",
            "web_scraper": "http://localhost:8015/execute",
            "architecture": "http://localhost:8020/execute",
            "task_classifier": "http://localhost:8011/classify",
            "knowledge_agent": "http://localhost:8025/execute",
            "memory_agent": "http://localhost:8026/execute",
        }

        # Streaming endpoints for real-time responses
        self.streaming_endpoints = {
            "master_agent": "http://localhost:8000/execute/stream",
            "fix_implementation_agent": "http://localhost:8004/execute/stream",
            "debugger_agent": "http://localhost:8005/execute/stream",
            "review_agent": "http://localhost:8006/execute/stream",
            "testing_agent": "http://localhost:8007/execute/stream",
            "security_agent": "http://localhost:8008/execute/stream",
            "performance_agent": "http://localhost:8009/execute/stream",
            "deployment_agent": "http://localhost:8017/execute/stream",
            "monitoring_agent": "http://localhost:8018/execute/stream",
            "feedback_agent": "http://localhost:8010/execute/stream",
            "comparator_service": "http://localhost:8012/execute/stream",
            "web_scraper": "http://localhost:8015/execute/stream",
            "architecture": "http://localhost:8020/execute/stream",
            "task_classifier": "http://localhost:8011/classify/stream",
            "knowledge_agent": "http://localhost:8025/execute/stream",
            "memory_agent": "http://localhost:8026/execute/stream",
        }

        # Health check endpoints
        self.health_endpoints = {
            "master_agent": "http://localhost:8000/health",
            "fix_implementation_agent": "http://localhost:8004/health",
            "debugger_agent": "http://localhost:8005/health",
            "review_agent": "http://localhost:8006/health",
            "testing_agent": "http://localhost:8007/health",
            "security_agent": "http://localhost:8008/health",
            "performance_agent": "http://localhost:8009/health",
            "deployment_agent": "http://localhost:8017/health",
            "monitoring_agent": "http://localhost:8018/health",
            "feedback_agent": "http://localhost:8010/health",
            "comparator_service": "http://localhost:8012/health",
            "web_scraper": "http://localhost:8015/health",
            "architecture": "http://localhost:8020/health",
            "task_classifier": "http://localhost:8011/health",
            "knowledge_agent": "http://localhost:8025/health",
            "memory_agent": "http://localhost:8026/health",
        }

        # Streaming health check endpoints
        self.streaming_health_endpoints = {
            "master_agent": "http://localhost:8000/health/stream",
            "fix_implementation_agent": "http://localhost:8004/health/stream",
            "debugger_agent": "http://localhost:8005/health/stream",
            "review_agent": "http://localhost:8006/health/stream",
            "testing_agent": "http://localhost:8007/health/stream",
            "security_agent": "http://localhost:8008/health/stream",
            "performance_agent": "http://localhost:8009/health/stream",
            "deployment_agent": "http://localhost:8017/health/stream",
            "monitoring_agent": "http://localhost:8018/health/stream",
            "feedback_agent": "http://localhost:8010/health/stream",
            "comparator_service": "http://localhost:8012/health/stream",
            "web_scraper": "http://localhost:8015/health/stream",
            "architecture": "http://localhost:8020/health/stream",
            "task_classifier": "http://localhost:8011/health/stream",
            "knowledge_agent": "http://localhost:8025/health/stream",
            "memory_agent": "http://localhost:8026/health/stream",
        }

        # Environment-based overrides
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load endpoint overrides from environment variables"""
        # Support patterns like AGENT_FIX_IMPLEMENTATION_URL, AGENT_DEBUGGER_URL, etc.
        for agent_name in self.endpoints.keys():
            env_var = f"AGENT_{agent_name.upper().replace('_', '')}_URL"
            override_url = os.getenv(env_var)
            if override_url:
                self.endpoints[agent_name] = override_url
                logger.info(f"ORCHESTRATOR: Overriding {agent_name} endpoint to {override_url}")

            # Also check for streaming endpoint overrides
            stream_env_var = f"AGENT_{agent_name.upper().replace('_', '')}_STREAM_URL"
            stream_override = os.getenv(stream_env_var)
            if stream_override:
                self.streaming_endpoints[agent_name] = stream_override
                logger.info(f"ORCHESTRATOR: Overriding {agent_name} streaming endpoint to {stream_override}")

            # Also check for health endpoint overrides
            health_env_var = f"AGENT_{agent_name.upper().replace('_', '')}_HEALTH_URL"
            health_override = os.getenv(health_env_var)
            if health_override:
                self.health_endpoints[agent_name] = health_override

            # Also check for streaming health endpoint overrides
            stream_health_env_var = f"AGENT_{agent_name.upper().replace('_', '')}_STREAM_HEALTH_URL"
            stream_health_override = os.getenv(stream_health_env_var)
            if stream_health_override:
                self.streaming_health_endpoints[agent_name] = stream_health_override

    def get_endpoint(self, agent_name: str, endpoint_type: str = "execute") -> str:
        """Get endpoint URL for an agent"""
        if endpoint_type == "health":
            url = self.health_endpoints.get(agent_name)
        elif endpoint_type == "stream":
            url = self.streaming_endpoints.get(agent_name)
        elif endpoint_type == "stream_health":
            url = self.streaming_health_endpoints.get(agent_name)
        else:
            url = self.endpoints.get(agent_name)

        # DEBUG: Log endpoint resolution for troubleshooting
        env_var_name = f"AGENT_{agent_name.upper().replace('_', '')}_URL"
        has_env_override = bool(os.getenv(env_var_name))
        structured_logger.log_event("endpoint_resolution", {
            "agent_name": agent_name,
            "endpoint_type": endpoint_type,
            "resolved_url": url,
            "has_env_override": has_env_override,
            "env_var_checked": env_var_name,
            "available_endpoints": list(self.endpoints.keys()) if endpoint_type == "execute" else list(self.health_endpoints.keys()),
            "endpoint_source": "environment" if has_env_override else "default"
        })

        if not url:
            logger.warning(f"ORCHESTRATOR: No {endpoint_type} endpoint configured for agent {agent_name}")
            structured_logger.log_event("endpoint_missing", {
                "agent_name": agent_name,
                "endpoint_type": endpoint_type,
                "configured_endpoints": list(self.endpoints.keys()) if endpoint_type == "execute" else list(self.health_endpoints.keys()),
                "env_vars_checked": [f"AGENT_{name.upper().replace('_', '')}_URL" for name in self.endpoints.keys()]
            })

        return url

    def get_all_endpoints(self) -> dict:
        """Get all configured endpoints"""
        return {
            "execute": self.endpoints.copy(),
            "health": self.health_endpoints.copy(),
            "stream": self.streaming_endpoints.copy(),
            "stream_health": self.streaming_health_endpoints.copy()
        }

# Global endpoint manager instance
endpoint_manager = AgentEndpointManager()

master_agent = MasterAgent()

# Cache for task classifications to improve response time
classification_cache = {}

async def classify_request_with_agent(user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify a user request using the task classifier agent with caching"""
    # Create cache key from input and context
    cache_key = f"{user_input}_{json.dumps(context, sort_keys=True) if context else ''}"

    # Check cache first
    if cache_key in classification_cache:
        logger.info("Using cached classification")
        return classification_cache[cache_key]

    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:  # Reduced timeout
            response = await client.post(
                "http://localhost:8011/classify",  # Task classifier agent port
                json={
                    "user_input": user_input,
                    "context": context
                }
            )

            if response.status_code == 200:
                result = response.json()
                # Cache the result for 5 minutes
                classification_cache[cache_key] = result
                # Clean cache periodically (keep only last 100 entries)
                if len(classification_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(classification_cache.keys())[:20]
                    for key in oldest_keys:
                        del classification_cache[key]
                return result
            else:
                logger.error(f"Task classifier agent error: {response.status_code} - {response.text}")
                # Fallback to simple classification
                result = await fallback_classification(user_input, context)
                classification_cache[cache_key] = result
                return result

    except Exception as e:
        logger.error(f"Failed to call task classifier agent: {e}")
        # Fallback to simple classification
        result = await fallback_classification(user_input, context)
        classification_cache[cache_key] = result
        return result

async def fallback_classification(user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced fallback classification when agent is unavailable"""
    lower_input = user_input.lower()
    input_length = len(user_input)
    word_count = len(user_input.split())
    line_count = user_input.count('\n') + 1

    # Enhanced complexity detection
    has_multiple_steps = any(word in lower_input for word in [
        'and', 'then', 'after', 'finally', 'next', 'step', 'phase', 'first', 'second', 'third',
        'multiple', 'several', 'various', 'create', 'build', 'implement', 'develop'
    ])

    has_complex_keywords = any(keyword in lower_input for keyword in [
        'system', 'architecture', 'design', 'framework', 'infrastructure', 'database',
        'api', 'service', 'microservice', 'deployment', 'ci/cd', 'testing', 'security'
    ])

    # Determine complexity based on multiple factors
    if input_length > 1000 or word_count > 150 or line_count > 10 or has_multiple_steps:
        complexity = "high"
        needs_decomposition = True
    elif input_length > 500 or word_count > 75 or has_complex_keywords:
        complexity = "medium"
        needs_decomposition = True
    else:
        complexity = "low"
        needs_decomposition = False

    # Check for system design/architecture keywords
    design_keywords = ["design", "architecture", "system design", "system architecture", "architect", "blueprint", "structure", "diagram", "schema", "model"]
    if any(keyword in lower_input for keyword in design_keywords):
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": needs_decomposition,
            "category": "design",
            "confidence": 0.9,
            "reasoning": f"Contains system design/architecture keywords, complexity: {complexity}",
            "suggested_agents": ["architecture"],
            "direct_response": None
        }

    # Check for coding keywords
    coding_keywords = ["write", "create", "implement", "build", "develop", "code", "function", "class", "fix", "debug", "test", "print", "output", "script", "program", "algorithm"]
    if any(keyword in lower_input for keyword in coding_keywords):
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": needs_decomposition,
            "category": "coding",
            "confidence": 0.8,
            "reasoning": f"Contains coding-related keywords, complexity: {complexity}",
            "suggested_agents": ["fix_implementation_agent"],
            "direct_response": None
        }

    # Check for testing keywords
    testing_keywords = ["test", "testing", "pytest", "unittest", "coverage", "assert", "fixture", "mock"]
    if any(keyword in lower_input for keyword in testing_keywords):
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": needs_decomposition,
            "category": "testing",
            "confidence": 0.8,
            "reasoning": f"Contains testing-related keywords, complexity: {complexity}",
            "suggested_agents": ["testing_agent"],
            "direct_response": None
        }

    # Check for deployment keywords
    deployment_keywords = ["deploy", "deployment", "docker", "kubernetes", "ci/cd", "pipeline", "build", "release"]
    if any(keyword in lower_input for keyword in deployment_keywords):
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": needs_decomposition,
            "category": "deployment",
            "confidence": 0.8,
            "reasoning": f"Contains deployment-related keywords, complexity: {complexity}",
            "suggested_agents": ["deployment_agent"],
            "direct_response": None
        }

    # Check for security keywords
    security_keywords = ["security", "vulnerability", "auth", "authentication", "encrypt", "encryption", "safe", "secure"]
    if any(keyword in lower_input for keyword in security_keywords):
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": needs_decomposition,
            "category": "security",
            "confidence": 0.8,
            "reasoning": f"Contains security-related keywords, complexity: {complexity}",
            "suggested_agents": ["security_agent"],
            "direct_response": None
        }

    # Default classification based on complexity
    if needs_decomposition:
        return {
            "type": "task",
            "complexity": complexity,
            "needs_decomposition": True,
            "category": "complex_task",
            "confidence": 0.7,
            "reasoning": f"Complex task detected (length: {input_length}, words: {word_count}, lines: {line_count})",
            "suggested_agents": ["fix_implementation_agent"],
            "direct_response": None
        }
    else:
        return {
            "type": "query",
            "complexity": "simple",
            "needs_decomposition": False,
            "category": "question",
            "confidence": 0.6,
            "reasoning": "Simple question or request",
            "suggested_agents": [],
            "direct_response": None
        }

# Provider configurations matching frontend
PROVIDER_CONFIGS = {
    "mistral": {
        "name": "Mistral",
        "type": "primary",
        "purpose": "Reasoning & planning models",
        "models": ["mistral-large", "mistral-medium", "mistral-small"],
        "status": "active"
    },
    "deepseek": {
        "name": "DeepSeek",
        "type": "primary",
        "purpose": "Code generation / code-focused models",
        "models": ["deepseek-coder", "deepseek-chat"],
        "status": "active"
    },
    "openrouter": {
        "name": "OpenRouter",
        "type": "primary",
        "purpose": "Gateway / routing to open LLMs",
        "models": ["gpt-4", "claude-3", "gemini-pro"],
        "status": "active"
    },
    "nvidia_nim": {
        "name": "NVIDIA NIM",
        "type": "primary",
        "purpose": "Specialized LLMs & inference platform",
        "models": ["nim-llama", "nim-codellama"],
        "status": "active"
    },
    "huggingface_local": {
        "name": "Local HuggingFace",
        "type": "fallback",
        "purpose": "Offline/local inference",
        "models": ["llama-2-7b", "codellama-7b"],
        "status": "standby"
    },
    "ollama": {
        "name": "Ollama",
        "type": "fallback",
        "purpose": "Local inference runtimes",
        "models": ["llama2", "codellama"],
        "status": "standby"
    }
}

async def initialize_providers():
    """Initialize providers in database if they don't exist"""
    async with async_session() as session:
        for provider_id, config in PROVIDER_CONFIGS.items():
            # Check if provider exists
            result = await session.execute(
                select(Provider).where(Provider.id == provider_id)
            )
            existing = result.scalar_one_or_none()

            if not existing:
                new_provider = Provider(
                    id=provider_id,
                    name=config["name"],
                    type=config["type"],
                    purpose=config["purpose"],
                    models=config["models"],
                    status=config["status"]
                )
                session.add(new_provider)

                # Add initial metrics
                metrics = ProviderMetrics(
                    provider_id=provider_id,
                    latency=0.0,
                    success_rate=100.0,
                    total_requests=0,
                    active_requests=0,
                    cost_estimate=0.0,
                    tokens_used=0
                )
                session.add(metrics)

        await session.commit()

# Agent configurations
AGENT_CONFIGS = {
    "master_agent": {
        "name": "Master Agent",
        "type": "master_agent",
        "description": "Orchestrates and coordinates all other agents",
        "status": "idle",
        "health": "healthy"
    },
    "fix_implementation_agent": {
        "name": "Fix Implementation Agent",
        "type": "fix_implementation_agent",
        "description": "Handles code fixes and implementation tasks",
        "status": "idle",
        "health": "healthy"
    },
    "debugger_agent": {
        "name": "Debugger Agent",
        "type": "debugger_agent",
        "description": "Debugs code and identifies issues",
        "status": "idle",
        "health": "healthy"
    },
    "review_agent": {
        "name": "Review Agent",
        "type": "review_agent",
        "description": "Reviews code for quality and standards",
        "status": "idle",
        "health": "healthy"
    },
    "testing_agent": {
        "name": "Testing Agent",
        "type": "testing_agent",
        "description": "Creates and runs tests",
        "status": "idle",
        "health": "healthy"
    },
    "security_agent": {
        "name": "Security Agent",
        "type": "security_agent",
        "description": "Handles security scanning and policies",
        "status": "idle",
        "health": "healthy"
    },
    "performance_agent": {
        "name": "Performance Agent",
        "type": "performance_agent",
        "description": "Analyzes and optimizes performance",
        "status": "idle",
        "health": "healthy"
    },
    "deployment_agent": {
        "name": "Deployment Agent",
        "type": "deployment_agent",
        "description": "Manages deployments and releases",
        "status": "idle",
        "health": "healthy"
    },
    "monitoring_agent": {
        "name": "Monitoring Agent",
        "type": "monitoring_agent",
        "description": "Monitors system health and metrics",
        "status": "idle",
        "health": "healthy"
    },
    "feedback_agent": {
        "name": "Feedback Agent",
        "type": "feedback_agent",
        "description": "Processes user feedback and improvements",
        "status": "idle",
        "health": "healthy"
    },
    "comparator_service": {
        "name": "Comparator Service",
        "type": "comparator_service",
        "description": "Compares outputs and ensures quality",
        "status": "idle",
        "health": "healthy"
    },
    "task_classifier": {
        "name": "Task Classifier Agent",
        "type": "task_classifier",
        "description": "Intelligently classifies user requests and determines appropriate handling",
        "status": "idle",
        "health": "healthy"
    },
    "web_scraper": {
        "name": "Web Scraper Agent",
        "type": "web_scraper",
        "description": "Handles web scraping and data extraction tasks",
        "status": "idle",
        "health": "healthy"
    },
    "architecture": {
        "name": "Architecture Agent",
        "type": "architecture",
        "description": "Handles system architecture and design tasks",
        "status": "idle",
        "health": "healthy"
    },
    "memory_agent": {
        "name": "Memory Agent",
        "type": "memory_agent",
        "description": "Manages memory and knowledge graph operations",
        "status": "idle",
        "health": "healthy"
    }
}

# Cache for agent health checks to improve performance
agent_health_cache = {}
HEALTH_CACHE_TTL = 30  # 30 seconds

# Circuit breaker for agent communication
class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # 'closed', 'open', 'half_open'

    def can_execute(self, agent_id: str) -> bool:
        """Check if we can execute a request for this agent"""
        state = self.state.get(agent_id, 'closed')
        if state == 'closed':
            return True
        elif state == 'open':
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time.get(agent_id, 0) > self.recovery_timeout:
                self.state[agent_id] = 'half_open'
                return True
            return False
        elif state == 'half_open':
            return True
        return True

    def record_success(self, agent_id: str):
        """Record a successful execution"""
        self.failure_count[agent_id] = 0
        self.state[agent_id] = 'closed'

    def record_failure(self, agent_id: str):
        """Record a failed execution"""
        self.failure_count[agent_id] = self.failure_count.get(agent_id, 0) + 1
        self.last_failure_time[agent_id] = time.time()

        if self.failure_count[agent_id] >= self.failure_threshold:
            self.state[agent_id] = 'open'
            logger.warning(f"Circuit breaker opened for agent {agent_id} after {self.failure_count[agent_id]} failures")

circuit_breaker = CircuitBreaker()

async def check_agent_health(agent_id: str) -> str:
    """Check if an agent service is actually running and healthy with caching"""
    import time
    global agent_health_cache

    # Master agent is always healthy (it's the orchestrator itself)
    if agent_id == "master_agent":
        return "healthy"

    # Check cache first
    cache_entry = agent_health_cache.get(agent_id)
    if cache_entry and (time.time() - cache_entry['timestamp']) < HEALTH_CACHE_TTL:
        return cache_entry['status']

    # Agent service URLs - match the ports used in execute_single_subtask
    agent_urls = {
        "fix_implementation_agent": "http://localhost:8004/health",
        "debugger_agent": "http://localhost:8005/health",
        "review_agent": "http://localhost:8006/health",
        "testing_agent": "http://localhost:8007/health",
        "security_agent": "http://localhost:8008/health",
        "performance_agent": "http://localhost:8009/health",
        "deployment_agent": "http://localhost:8017/health",
        "monitoring_agent": "http://localhost:8018/health",
        "feedback_agent": "http://localhost:8010/health",
        "web_scraper": "http://localhost:8015/health",
        "task_classifier": "http://localhost:8011/health",
        "comparator_service": "http://localhost:8012/health",
        "architecture": "http://localhost:8020/health",
    }

    url = agent_urls.get(agent_id)
    if not url:
        status = "unknown"  # Agent doesn't have a service URL
    else:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:  # 2 second timeout
                response = await client.get(url)
                if response.status_code == 200:
                    status = "healthy"
                else:
                    status = "error"
        except Exception as e:
            logger.debug(f"Agent {agent_id} health check failed: {e}")
            status = "error"

    # Cache the result
    agent_health_cache[agent_id] = {
        'status': status,
        'timestamp': time.time()
    }

    # Clean old cache entries periodically
    if len(agent_health_cache) > 20:
        current_time = time.time()
        to_remove = [k for k, v in agent_health_cache.items() if (current_time - v['timestamp']) >= HEALTH_CACHE_TTL]
        for k in to_remove:
            del agent_health_cache[k]

    return status

async def initialize_agents():
    """Initialize agents in database if they don't exist"""
    async with async_session() as session:
        for agent_id, config in AGENT_CONFIGS.items():
            # Check if agent exists
            result = await session.execute(
                select(Agent).where(Agent.id == agent_id)
            )
            existing = result.scalar_one_or_none()

            if not existing:
                new_agent = Agent(
                    id=agent_id,
                    name=config["name"],
                    type=config["type"],
                    description=config["description"],
                    status=config["status"],
                    health=config["health"]
                )
                session.add(new_agent)

                # Add initial metrics
                metrics = AgentMetrics(
                    agent_id=agent_id,
                    tasks_completed=0,
                    success_rate=100.0,
                    average_response_time=0.0,
                    error_count=0
                )
                session.add(metrics)

        await session.commit()


async def update_agent_metrics(agent_name: str, success: bool):
    """Update agent metrics in database"""
    try:
        async with async_session() as session:
            # Get current metrics
            result = await session.execute(
                select(AgentMetrics).where(AgentMetrics.agent_id == agent_name)
            )
            metrics = result.scalar_one_or_none()

            if metrics:
                # Update metrics
                metrics.tasks_completed += 1
                metrics.last_activity = func.now()

                # Recalculate success rate
                total_tasks = metrics.tasks_completed
                if success:
                    # Approximate success rate based on current rate
                    # In a real implementation, you'd track success/failure separately
                    current_successes = int(metrics.success_rate * (total_tasks - 1) / 100)
                    new_successes = current_successes + 1
                    metrics.success_rate = (new_successes / total_tasks) * 100
                else:
                    # Task failed
                    current_successes = int(metrics.success_rate * (total_tasks - 1) / 100)
                    metrics.success_rate = (current_successes / total_tasks) * 100

                await session.commit()
                logger.info(f"Updated metrics for agent {agent_name}: tasks={metrics.tasks_completed}, success_rate={metrics.success_rate:.1f}%")
    except Exception as e:
        logger.error(f"Failed to update metrics for agent {agent_name}: {e}")

async def execute_single_subtask(subtask: Subtask, manager=None):
    """Execute a single subtask with enhanced error handling, retries, and circuit breaker"""
    # Use the centralized endpoint manager for consistent URL resolution
    url = endpoint_manager.get_endpoint(subtask.agent_name, "execute")
    if not url:
        logger.error(f"No URL configured for agent {subtask.agent_name}")
        subtask.output = {"error": f"Agent {subtask.agent_name} not available - no URL configured"}
        subtask.status = "failed"
        success = False
    else:
        # Check circuit breaker
        if not circuit_breaker.can_execute(subtask.agent_name):
            logger.warning(f"Circuit breaker prevented execution for agent {subtask.agent_name}")
            subtask.output = {"error": f"Agent {subtask.agent_name} is currently unavailable (circuit breaker open)"}
            subtask.status = "failed"
            success = False
        else:
            success = await execute_with_retry(subtask, url, manager)

    # Update agent metrics
    await update_agent_metrics(subtask.agent_name, success)

    # Update task status and output
    async with async_session() as session:
        try:
            # Update subtask
            session.add(subtask)
            # Update task
            result = await session.execute(select(Task).where(Task.id == subtask.task_id))
            task = result.scalar_one()
            task.output = subtask.output
            task.status = "completed" if subtask.status == "completed" else "failed"
            await session.commit()
            logger.info(f"Updated task {task.id} status to {task.status}")
        except Exception as e:
            logger.error(f"Failed to update task status for task {subtask.task_id}: {e}")
            # Don't raise - we want to continue with broadcasting

    if manager:
        try:
            await manager.broadcast(json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": subtask.task_id}))
            # Broadcast the result
            if subtask.output:
                output_message = {
                    "type": "output",
                    "task_id": subtask.task_id,
                    "message": json.dumps([subtask.output]) if isinstance(subtask.output, dict) else json.dumps([{"response": str(subtask.output)}])
                }
                await manager.broadcast(json.dumps(output_message))
        except Exception as e:
            logger.error(f"Failed to broadcast completion message: {e}")


async def execute_with_retry(subtask: Subtask, url: str = None, manager=None, max_retries=3):
    """Execute agent call with exponential backoff retry logic"""
    import httpx
    import asyncio

    last_exception = None
    start_time = asyncio.get_event_loop().time()

    # Log task execution start
    structured_logger.log_event("task_execution_start", **{
        "task_id": subtask.task_id,
        "agent_name": subtask.agent_name,
        "subtask_description": subtask.description[:100] + "..." if len(subtask.description) > 100 else subtask.description
    })

    # Get URL from endpoint manager if not provided
    if not url:
        url = endpoint_manager.get_endpoint(subtask.agent_name, "execute")
        if not url:
            logger.error(f"ORCHESTRATOR: No endpoint configured for agent {subtask.agent_name}")
            subtask.output = {"error": f"Agent {subtask.agent_name} not available - no endpoint configured"}
            subtask.status = "failed"
            return False

    for attempt in range(max_retries):
        try:
            # Calculate timeout based on attempt and task complexity (longer for retries and complex tasks)
            # Increased base timeouts for complex AI tasks
            task_length = len(subtask.description)
            base_timeout = 300.0 if task_length > 1000 else 120.0 if task_length > 500 else 60.0

            # Increased max timeout for very complex tasks: 1 hour (3600 seconds) for tasks > 2000 chars
            max_timeout = 3600.0 if task_length > 2000 else 1200.0
            timeout = min(base_timeout + (attempt * 60.0), max_timeout)

            # DEBUG: Enhanced timeout analysis
            timeout_analysis = {
                "task_length": task_length,
                "complexity_level": "high" if task_length > 1000 else "medium" if task_length > 500 else "low",
                "base_timeout": base_timeout,
                "attempt_number": attempt + 1,
                "retry_increment": attempt * 60.0,
                "calculated_timeout": timeout,
                "max_timeout": max_timeout,
                "timeout_capped": timeout >= max_timeout,
                "task_description_preview": subtask.description[:100] + "..." if len(subtask.description) > 100 else subtask.description
            }

            structured_logger.log_event("timeout_calculation_detailed", **{
                "agent_name": subtask.agent_name,
                "task_id": subtask.task_id,
                **timeout_analysis
            })

            # DEBUG: Log potential timeout issues for very complex tasks
            if timeout >= max_timeout:
                structured_logger.log_event("potential_timeout_issue", **{
                    "agent_name": subtask.agent_name,
                    "task_id": subtask.task_id,
                    "task_length": task_length,
                    "calculated_timeout": timeout,
                    "max_allowed": max_timeout,
                    "issue": f"Task hit timeout cap of {max_timeout}s - very complex task may need even higher limits",
                    "recommendation": "Monitor task completion; consider further timeout increases if needed"
                })
                logger.warning(f"ORCHESTRATOR: Task {subtask.task_id} hit timeout cap of {timeout}s - very complex task")

            logger.info(f"ORCHESTRATOR: Attempt {attempt + 1} timeout set to {timeout}s for agent {subtask.agent_name} (task length: {task_length}, complexity: {timeout_analysis['complexity_level']})")

            # Additional logging for timeout issues
            if timeout >= 1200.0:
                structured_logger.log_event("timeout_capped_warning", **{
                    "agent_name": subtask.agent_name,
                    "task_id": subtask.task_id,
                    "task_length": task_length,
                    "attempt": attempt + 1,
                    "capped_timeout": timeout,
                    "reason": "Maximum timeout reached - task may be too complex or retries too many"
                })

            # DEBUG: Log timeout analysis for complex tasks
            task_complexity = "high" if len(subtask.description) > 1000 else "medium" if len(subtask.description) > 500 else "low"
            structured_logger.log_event("timeout_calculation", **{
                "task_id": subtask.task_id,
                "agent_name": subtask.agent_name,
                "attempt": attempt + 1,
                "task_length": len(subtask.description),
                "task_complexity": task_complexity,
                "base_timeout": base_timeout,
                "calculated_timeout": timeout,
                "max_timeout": 1200.0
            })

            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Attempting to call agent {subtask.agent_name} at {url} (attempt {attempt + 1}/{max_retries})")

                attempt_start = asyncio.get_event_loop().time()
                response = await client.post(url, json={"description": subtask.description})
                attempt_end = asyncio.get_event_loop().time()

                response_time = attempt_end - attempt_start
                logger.info(f"Agent {subtask.agent_name} responded in {response_time:.2f}s")

                if response.status_code == 200:
                    # DEBUG: Log response format analysis
                    response_text = response.text
                    try:
                        output = response.json()
                        response_format = "json"
                        output_keys = list(output.keys()) if isinstance(output, dict) else "non-dict"
                        has_expected_keys = isinstance(output, dict) and ("result" in output or "output" in output or "response" in output)
                        json_structure = "dict" if isinstance(output, dict) else type(output).__name__
                    except Exception as parse_error:
                        output = response_text
                        response_format = "text"
                        output_keys = "N/A"
                        has_expected_keys = False
                        json_structure = "parse_error"
                        structured_logger.log_event("json_parse_error", **{
                            "agent_name": subtask.agent_name,
                            "task_id": subtask.task_id,
                            "parse_error": str(parse_error),
                            "response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text
                        })

                    structured_logger.log_event("agent_response_analysis", **{
                        "agent_name": subtask.agent_name,
                        "task_id": subtask.task_id,
                        "response_format": response_format,
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "response_length": len(response_text),
                        "output_keys": output_keys,
                        "has_expected_keys": has_expected_keys,
                        "json_structure": json_structure,
                        "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                        "is_normalized": False  # Will be updated after normalization
                    })

                    # DEBUG: Log potential response format issues
                    if not has_expected_keys and response_format == "json":
                        structured_logger.log_event("response_format_issue", **{
                            "agent_name": subtask.agent_name,
                            "task_id": subtask.task_id,
                            "issue": "Agent returned JSON but without expected keys (result/output/response)",
                            "actual_keys": output_keys,
                            "response_format": response_format,
                            "recommendation": "Update response normalization to handle this agent's format"
                        })
                        logger.warning(f"ORCHESTRATOR: Agent {subtask.agent_name} returned JSON without expected keys: {output_keys}")
                    elif response_format == "text" and len(response_text.strip()) > 0:
                        structured_logger.log_event("text_response_issue", **{
                            "agent_name": subtask.agent_name,
                            "task_id": subtask.task_id,
                            "issue": "Agent returned plain text response instead of JSON",
                            "response_length": len(response_text),
                            "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                            "recommendation": "Consider normalizing text responses or updating agent to return JSON"
                        })
                        logger.warning(f"ORCHESTRATOR: Agent {subtask.agent_name} returned plain text response")

                    logger.info(f"ORCHESTRATOR: Agent {subtask.agent_name} returned success with format: {response_format}, keys: {output_keys}")
                    subtask.output = output
                    subtask.status = "completed"

                    # Record success in circuit breaker
                    circuit_breaker.record_success(subtask.agent_name)

                    # Process output for file previews and run steps
                    message_data = await process_agent_output(output)

                    # Store message_data for broadcasting
                    if manager:
                        # Make message_data available to caller
                        globals()['message_data'] = message_data

                    # Log successful completion
                    total_time = asyncio.get_event_loop().time() - start_time
                    structured_logger.log_event("task_execution_success", **{
                        "task_id": subtask.task_id,
                        "agent_name": subtask.agent_name,
                        "attempts": attempt + 1,
                        "response_time": response_time,
                        "total_time": total_time
                    })

                    return True
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Agent {subtask.agent_name} returned error: {error_msg}")
                    last_exception = Exception(error_msg)

                    # DEBUG: Enhanced error parsing and logging
                    try:
                        error_details = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"raw_error": response.text}
                    except:
                        error_details = {"raw_error": response.text, "parse_failed": True}

                    # Categorize error types for better debugging
                    error_category = "unknown"
                    if response.status_code >= 500:
                        error_category = "server_error"
                    elif response.status_code == 429:
                        error_category = "rate_limit"
                    elif response.status_code == 408:
                        error_category = "timeout"
                    elif response.status_code == 404:
                        error_category = "not_found"
                    elif response.status_code >= 400:
                        error_category = "client_error"

                    # Log detailed error information
                    structured_logger.log_event("task_execution_attempt_failed", **{
                        "task_id": subtask.task_id,
                        "agent_name": subtask.agent_name,
                        "attempt": attempt + 1,
                        "status_code": response.status_code,
                        "error_category": error_category,
                        "error_details": error_details,
                        "response_time": response_time,
                        "response_headers": dict(response.headers),
                        "error_length": len(response.text)
                    })

                    # For 5xx errors, retry; for 4xx errors, don't retry
                    if 400 <= response.status_code < 500:
                        break

        except httpx.TimeoutException as e:
            logger.warning(f"Timeout calling agent {subtask.agent_name} (attempt {attempt + 1}): {e}")
            last_exception = e

            # Log timeout
            structured_logger.log_event("task_execution_timeout", **{
                "task_id": subtask.task_id,
                "agent_name": subtask.agent_name,
                "attempt": attempt + 1,
                "timeout_seconds": timeout
            })

        except httpx.ConnectError as e:
            logger.warning(f"Connection error calling agent {subtask.agent_name} (attempt {attempt + 1}): {e}")
            last_exception = e

            # Log connection error
            structured_logger.log_event("task_execution_connection_error", **{
                "task_id": subtask.task_id,
                "agent_name": subtask.agent_name,
                "attempt": attempt + 1,
                "error": str(e)
            })

        except Exception as e:
            logger.error(f"Unexpected error calling agent {subtask.agent_name} (attempt {attempt + 1}): {e}")
            last_exception = e

            # Log unexpected error
            structured_logger.log_event("task_execution_unexpected_error", **{
                "task_id": subtask.task_id,
                "agent_name": subtask.agent_name,
                "attempt": attempt + 1,
                "error": str(e),
                "error_type": type(e).__name__
            })

        # Record failure in circuit breaker
        circuit_breaker.record_failure(subtask.agent_name)

        # Exponential backoff: wait 2^attempt seconds, but not more than 30 seconds
        if attempt < max_retries - 1:
            wait_time = min(2 ** attempt, 30)
            logger.info(f"Retrying agent {subtask.agent_name} in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    # All retries failed
    error_msg = f"Failed to execute agent {subtask.agent_name} after {max_retries} attempts"
    if last_exception:
        error_msg += f": {str(last_exception)}"

    logger.error(error_msg)
    subtask.output = {"error": error_msg}
    subtask.status = "failed"

    # Log final failure
    total_time = asyncio.get_event_loop().time() - start_time
    structured_logger.log_event("task_execution_failed", **{
        "task_id": subtask.task_id,
        "agent_name": subtask.agent_name,
        "attempts": max_retries,
        "total_time": total_time,
        "final_error": error_msg
    })

    return False


async def process_agent_output(output: dict) -> dict:
    """Process agent output to extract file previews, explanations, and run steps"""
    ai_output = []
    explanation_parts = []
    run_steps = []

    try:
        if output.get("structured") and output["structured"].get("files"):
            for file_path, content in output["structured"]["files"].items():
                content_size = len(content)
                is_large = content_size > 10000  # 10KB threshold for large files

                if is_large:
                    # For large files, only indicate creation
                    ai_output.append({
                        "file_created": {
                            "filePath": file_path,
                            "size": content_size,
                            "message": f"Large file created ({content_size} characters) - content not shown in chat"
                        }
                    })
                    explanation_parts.append(f"Created large file: {file_path} ({content_size} chars)")
                else:
                    # For small files, show preview
                    ai_output.append({
                        "file_preview": {
                            "filePath": file_path,
                            "content": content,
                            "language": None  # Will be auto-detected
                        }
                    })
                    explanation_parts.append(f"Generated file: {file_path}")

                # Generate run steps based on file extension
                if file_path.endswith('.py'):
                    run_steps.append(f"python {file_path}")
                elif file_path.endswith('.js'):
                    run_steps.append(f"node {file_path}")
                elif file_path.endswith('.sh'):
                    run_steps.append(f"bash {file_path}")
                elif file_path.endswith('.c'):
                    run_steps.append(f"gcc {file_path} -o {file_path.rsplit('.', 1)[0]} && ./{file_path.rsplit('.', 1)[0]}")
                elif file_path.endswith('.cpp'):
                    run_steps.append(f"g++ {file_path} -o {file_path.rsplit('.', 1)[0]} && ./{file_path.rsplit('.', 1)[0]}")
                elif file_path.endswith('.java'):
                    class_name = file_path.split('/')[-1].replace('.java', '')
                    run_steps.append(f"javac {file_path} && java {class_name}")
                # Add more file types as needed
    except Exception as e:
        logger.warning(f"Error processing agent output: {e}")

    message_data = {"type": "output", "message": json.dumps([output])}
    if ai_output:
        message_data["aiOutput"] = ai_output
    if explanation_parts:
        message_data["explanation"] = " ".join(explanation_parts)
    if run_steps:
        message_data["run_steps"] = run_steps

    return message_data

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting orchestrator startup event")
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized")
        logger.info("Initializing providers...")
        await initialize_providers()
        logger.info("Providers initialized")
        logger.info("Initializing agents...")
        await initialize_agents()
        logger.info("Agents initialized")
        logger.info("Orchestrator startup event completed")
    except Exception as e:
        logger.error(f"Error in orchestrator startup event: {e}")
        logger.warning("Continuing startup despite errors - some features may not work")
        # Don't raise to allow basic functionality


@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    async with async_session() as session:
        new_project = Project(
            user_id="default",
            name=project.name,
            description=project.description
        )
        session.add(new_project)
        await session.commit()
        await session.refresh(new_project)
        return ProjectResponse(
            id=new_project.id,
            user_id=new_project.user_id,
            name=new_project.name,
            description=new_project.description,
            status=new_project.status,
            created_at=new_project.created_at.isoformat(),
            updated_at=new_project.updated_at.isoformat(),
            task_count=0
        )

@app.get("/api/projects", response_model=list[ProjectResponse])
async def list_projects():
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.user_id == "default")
        )
        projects = result.scalars().all()

        project_responses = []
        for project in projects:
            # Count tasks for this project
            task_result = await session.execute(
                select(Task).where(Task.project_id == project.id)
            )
            task_count = len(task_result.scalars().all())

            project_responses.append(ProjectResponse(
                id=project.id,
                user_id=project.user_id,
                name=project.name,
                description=project.description,
                status=project.status,
                created_at=project.created_at.isoformat(),
                updated_at=project.updated_at.isoformat(),
                task_count=task_count
            ))

        return project_responses

@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == "default"
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Count tasks for this project
        task_result = await session.execute(
            select(Task).where(Task.project_id == project.id)
        )
        task_count = len(task_result.scalars().all())

        return ProjectResponse(
            id=project.id,
            user_id=project.user_id,
            name=project.name,
            description=project.description,
            status=project.status,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            task_count=task_count
        )

@app.put("/api/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project_data: dict):
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == "default"
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Update fields
        for key, value in project_data.items():
            if hasattr(project, key):
                setattr(project, key, value)

        project.updated_at = func.now()
        await session.commit()
        await session.refresh(project)

        # Count tasks for this project
        task_result = await session.execute(
            select(Task).where(Task.project_id == project.id)
        )
        task_count = len(task_result.scalars().all())

        return ProjectResponse(
            id=project.id,
            user_id=project.user_id,
            name=project.name,
            description=project.description,
            status=project.status,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            task_count=task_count
        )

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == "default"
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        await session.delete(project)
        await session.commit()

        return {"message": f"Project {project.name} deleted successfully"}

@app.post("/api/tasks")
async def create_task(request: Request, background_tasks: BackgroundTasks):
    """
    Create a new task and start orchestration.
    Accepts both structured JSON and unstructured text input.
    """
    try:
        # Try to parse as JSON first
        try:
            task_data = await request.json()
            logger.info("Received JSON request")
        except Exception:
            # If JSON parsing fails, treat as plain text
            body = await request.body()
            try:
                task_data = body.decode('utf-8').strip()
                logger.info("Received plain text request")
            except UnicodeDecodeError:
                task_data = str(body)
                logger.info("Received binary request, converted to string")

        # Extract task description and other fields from various input formats
        task_description = None
        project_id = None
        selected_agent = None
        context = None

        if isinstance(task_data, str):
            # Plain text input - treat as task description
            task_description = task_data
        elif isinstance(task_data, dict):
            # JSON input - extract fields
            task_description = task_data.get("task") or task_data.get("description") or task_data.get("prompt")
            project_id = task_data.get("project_id")
            selected_agent = task_data.get("selectedAgent") or task_data.get("agent")
            context = task_data.get("context")

            # If no standard fields found, treat the entire dict as unstructured input
            if not task_description:
                task_description = str(task_data)
        else:
            # Convert any other format to string
            task_description = str(task_data)

        if not task_description or not task_description.strip():
            raise HTTPException(status_code=400, detail="Task description cannot be empty")

        task_description = task_description.strip()

        # Check if manual agent selection is requested
        if selected_agent and selected_agent != "auto":
            logger.info(f"Manual agent selection: {selected_agent}")
            # Skip classification and directly assign to selected agent
            classification = {
                "type": "task",
                "complexity": "medium",
                "needs_decomposition": False,
                "category": "manual",
                "confidence": 1.0,
                "reasoning": f"Manually selected agent: {selected_agent}",
                "suggested_agents": [selected_agent],
                "direct_response": None
            }
        else:
            # Prepare context with conversation history for better classification
            enhanced_context = context or {}

            # Add conversation context if available (from frontend)
            if context and context.get("conversation_history"):
                enhanced_context["conversation_history"] = context["conversation_history"]
                logger.info(f"Including conversation history with {len(context['conversation_history'])} messages for classification")

            # First, classify the request using the task classifier agent
            classification = await classify_request_with_agent(task_description, enhanced_context)

        logger.info(f"Request classification: {classification}")

        # Handle different types of requests
        if classification["type"] == "direct_response":
            # Create a task for direct response so frontend can subscribe to updates
            async with async_session() as session:
                new_task = Task(
                    user_id="default",
                    description=task_description,
                    project_id=project_id,
                    context=context
                )
                session.add(new_task)
                await session.commit()
                await session.refresh(new_task)

                # Store classification in task context
                if new_task.context is None:
                    new_task.context = {}
                new_task.context["classification"] = classification
                await session.commit()

                # Broadcast task creation
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "task_created", "task_id": new_task.id, "description": task_description})
                )

                # Complete the task immediately with direct response
                new_task.output = {"response": classification["direct_response"]}
                new_task.status = "completed"
                await session.commit()

                # Broadcast completion
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": new_task.id})
                )
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "output", "message": json.dumps([{"response": classification["direct_response"]}]), "task_id": new_task.id})
                )

            return {
                "task_id": new_task.id,
                "type": "direct_response",
                "response": classification["direct_response"],
                "classification": classification
            }

        elif classification["type"] == "about":
            # Handle about queries by calling the about endpoint
            about_response = await about("detailed")  # Use detailed level for chat

            # Create a task for about response so frontend can subscribe to updates
            async with async_session() as session:
                new_task = Task(
                    user_id="default",
                    description=task_description,
                    project_id=project_id,
                    context=context
                )
                session.add(new_task)
                await session.commit()
                await session.refresh(new_task)

                # Store classification in task context
                if new_task.context is None:
                    new_task.context = {}
                new_task.context["classification"] = classification
                await session.commit()

                # Broadcast task creation
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "task_created", "task_id": new_task.id, "description": task_description})
                )

                # Complete the task immediately with about response
                new_task.output = {"response": about_response["response"], "system_prompt": about_response.get("system_prompt")}
                new_task.status = "completed"
                await session.commit()

                # Broadcast completion
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": new_task.id})
                )
                background_tasks.add_task(
                    manager.broadcast,
                    json.dumps({"type": "output", "message": json.dumps([{"response": about_response["response"], "system_prompt": about_response.get("system_prompt")}])})
                )

            return {
                "task_id": new_task.id,
                "type": "about_response",
                "response": about_response["response"],
                "system_prompt": about_response.get("system_prompt"),
                "classification": classification
            }

        elif classification["type"] == "query" and not classification["needs_decomposition"]:
            # Handle simple queries that don't need task decomposition
            # Use Aetherium to generate a direct response with enhanced context retrieval
            try:
                from providers.nim_adapter import NIMAdapter
                adapter = NIMAdapter()

                # Always retrieve conversation context for queries, especially follow-ups
                context_info = ""
                conversation_context = ""

                # Get recent conversation history from context
                if context and context.get("conversation_history"):
                    recent_messages = context["conversation_history"][-5:]  # Last 5 messages for context
                    conversation_context = "\n\nRecent Conversation:\n" + "\n".join([
                        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')[:200]}..."
                        for msg in recent_messages
                    ])

                # Add any additional context
                if context and not context.get("conversation_history"):
                    context_info = f"\n\nContext: {json.dumps(context)}"

                # Enhanced prompt that includes conversation context
                prompt = f"""You are the Aetherium system, created by NOVA tech. Answer this user query directly and helpfully.

Query: {task_description}{conversation_context}{context_info}

Provide a clear, concise, and helpful response. If this is about code, provide specific guidance.
If this appears to be a follow-up question (like "explain more", "tell me more", etc.), reference the previous conversation context in your response."""

                messages = [{"role": "user", "content": prompt}]
                ai_response = adapter.call_model(messages)

                if hasattr(ai_response, 'text'):
                    response_text = ai_response.text.strip()
                else:
                    response_text = str(ai_response).strip()

                # Create a task for query response so frontend can subscribe to updates
                async with async_session() as session:
                    new_task = Task(
                        user_id="default",
                        description=task_description,
                        project_id=project_id,
                        context=context
                    )
                    session.add(new_task)
                    await session.commit()
                    await session.refresh(new_task)

                    # Store classification in task context
                    if new_task.context is None:
                        new_task.context = {}
                    new_task.context["classification"] = classification
                    await session.commit()

                    # Broadcast task creation
                    background_tasks.add_task(
                        manager.broadcast,
                        json.dumps({"type": "task_created", "task_id": new_task.id, "description": task_description})
                    )

                    # Complete the task immediately with query response
                    new_task.output = {"response": response_text}
                    new_task.status = "completed"
                    await session.commit()

                    # Broadcast completion
                    background_tasks.add_task(
                        manager.broadcast,
                        json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": new_task.id})
                    )
                    background_tasks.add_task(
                        manager.broadcast,
                        json.dumps({"type": "output", "message": json.dumps([{"response": response_text}]), "task_id": new_task.id})
                    )

                return {
                    "task_id": new_task.id,
                    "type": "query_response",
                    "response": response_text,
                    "classification": classification
                }

            except Exception as e:
                logger.error(f"Error generating direct query response: {e}")
                # Fallback to task creation

        # For tasks that need processing, create the task as usual
        async with async_session() as session:
            # Validate project_id if provided (optimize by doing this only if needed)
            if project_id:
                project_result = await session.execute(
                    select(Project).where(
                        Project.id == project_id,
                        Project.user_id == "default"
                    )
                )
                project = project_result.scalar_one_or_none()
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")

            # Check if task needs decomposition based on complexity
            task_content = task_description
            if context:
                task_content += json.dumps(context)

            # Enhanced complexity detection for automatic decomposition
            content_length = len(task_content)
            line_count = task_content.count('\n')
            word_count = len(task_content.split())
            has_complex_keywords = any(keyword in task_content.lower() for keyword in [
                'create', 'build', 'implement', 'develop', 'design', 'architect',
                'system', 'application', 'multiple', 'several', 'complex', 'large'
            ])

            # Decompose if: already marked for decomposition, very long content, many lines,
            # many words, or contains complex keywords
            should_decompose = (
                classification["needs_decomposition"] or
                content_length > 1500 or  # Reduced threshold
                line_count > 15 or        # Reduced threshold
                word_count > 100 or       # New word count check
                has_complex_keywords      # New keyword detection
            )

            if should_decompose and not classification["needs_decomposition"]:
                # Force decomposition for complex tasks
                classification["needs_decomposition"] = True
                classification["reasoning"] += " (automatically decomposed due to complexity)"
                logger.info(f"Automatically decomposing complex task: {content_length} chars, {line_count} lines, {word_count} words, complex_keywords: {has_complex_keywords}")

            new_task = Task(
                user_id="default",
                description=task_description,
                project_id=project_id,
                context=context
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)

            # Store classification in task context for later use
            if new_task.context is None:
                new_task.context = {}
            new_task.context["classification"] = classification
            await session.commit()

            # Broadcast task creation asynchronously
            background_tasks.add_task(
                manager.broadcast,
                json.dumps({"type": "task_created", "task_id": new_task.id, "description": task_description})
            )

            # Only start orchestration if the task needs decomposition
            if classification["needs_decomposition"]:
                # Start orchestration in background
                background_tasks.add_task(master_agent.orchestrate_task, new_task.id, manager)
            else:
                # For simple tasks, handle directly in background
                background_tasks.add_task(handle_simple_task, new_task.id, classification, manager)

            return {"task_id": new_task.id, "classification": classification}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def handle_simple_task(task_id: str, classification: Dict[str, Any], manager=None):
    """Handle simple tasks that don't need full decomposition"""
    try:
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one()

            # Generate a simple response based on classification
            if classification["category"] == "coding" and classification["complexity"] == "simple":
                # Simple coding task - use fix_implementation agent directly
                response = await execute_single_subtask_for_simple_task(task, "fix_implementation_agent", manager)
            elif classification["category"] == "analysis":
                # Analysis task - use review agent
                response = await execute_single_subtask_for_simple_task(task, "review_agent", manager)
            else:
                # Default to fix_implementation for other simple tasks
                response = await execute_single_subtask_for_simple_task(task, "fix_implementation_agent", manager)

            # Update task with response
            task.output = response
            task.status = "completed"
            await session.commit()

            if manager:
                # Broadcast completion
                await manager.broadcast(json.dumps({"type": "status", "status": "completed", "progress": 100, "task_id": task_id}))
                # Broadcast the result
                await manager.broadcast(json.dumps({"type": "output", "message": json.dumps([response]), "task_id": task_id}))

    except Exception as e:
        logger.error(f"Error handling simple task {task_id}: {e}")
        # Update task status to failed
        try:
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one()
                task.status = "failed"
                task.output = {"error": str(e)}
                await session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to update task status: {inner_e}")


async def execute_single_subtask_for_simple_task(task: Task, agent_name: str, manager=None) -> Dict[str, Any]:
    """Execute a single agent for simple tasks"""
    agent_urls = {
        "fix_implementation_agent": "http://localhost:8004/execute",
        "review_agent": "http://localhost:8006/execute",
        "testing_agent": "http://localhost:8007/execute",
    }

    url = agent_urls.get(agent_name)
    if not url:
        # Update metrics for failed task
        await update_agent_metrics(agent_name, False)
        return {"error": f"Agent {agent_name} not available"}

    # Send progress update
    if manager:
        await manager.broadcast(json.dumps({"type": "status", "status": "running", "progress": 50, "task_id": task.id}))

    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json={"description": task.description})
            output = response.json()

            # Normalize response format
            if output.get("success"):
                # Update metrics for successful task
                await update_agent_metrics(agent_name, True)
                return {"response": output.get("result", "")}
            else:
                # Update metrics for failed task
                await update_agent_metrics(agent_name, False)
                return {"error": output.get("error", "Agent execution failed")}
        except Exception as e:
            # Update metrics for failed task
            await update_agent_metrics(agent_name, False)
            return {"error": str(e)}

@app.get("/api/tasks", response_model=list[TaskResponse])
async def list_tasks(
    project_id: Optional[str] = None,
    status: Optional[str] = None
):
    async with async_session() as session:
        query = select(Task).where(Task.user_id == "default")

        if project_id:
            query = query.where(Task.project_id == project_id)
        if status:
            query = query.where(Task.status == status)

        result = await session.execute(query)
        tasks = result.scalars().all()

        task_responses = []
        for task in tasks:
            sub_result = await session.execute(select(Subtask).where(Subtask.task_id == task.id))
            subtasks = sub_result.scalars().all()
            task_responses.append(TaskResponse(
                id=task.id,
                project_id=task.project_id,
                user_id=task.user_id,
                description=task.description,
                status=task.status,
                plan=task.plan,
                output=task.output,
                context=task.context,
                subtasks=[{"id": s.id, "agent": s.agent_name, "description": s.description, "status": s.status, "output": s.output} for s in subtasks],
                created_at=task.created_at.isoformat()
            ))

        return task_responses

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # For WebSocket connections, we'll allow them since they come from trusted frontend
    # The CORS middleware handles HTTP requests, WebSocket has its own security
    origin = websocket.headers.get('origin')
    logger.info(f"WebSocket connection attempt from origin: {origin}")
    # Allow all origins for WebSocket connections (more permissive for development)
    logger.info(f"WebSocket connection accepted from origin: {origin}")
    logger.info(f"WebSocket headers: {dict(websocket.headers)}")
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"WebSocket received data: {data[:200]}...")
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "subscribe_task":
                    # Handle task subscription
                    logger.info(f"Received subscribe_task message: {message}")
                    task_id = message.get("taskId")
                    if not task_id:
                        logger.error(f"subscribe_task message missing taskId. Full message: {message}")
                        await websocket.send_text(json.dumps({"type": "error", "message": "Missing taskId in subscribe_task message"}))
                        continue

                    logger.info(f"Client subscribing to task: {task_id}")

                    # Get current task status from database
                    async with async_session() as session:
                        result = await session.execute(select(Task).where(Task.id == task_id))
                        task = result.scalar_one_or_none()

                        if not task:
                            logger.error(f"Task not found: {task_id}")
                            await websocket.send_text(json.dumps({"type": "error", "message": "Task not found"}))
                            continue

                        # Send current task status
                        status_message = {
                            "type": "status",
                            "task_id": task_id,
                            "status": task.status,
                            "progress": 100 if task.status == "completed" else 0
                        }
                        await websocket.send_text(json.dumps(status_message))

                        # If task is completed, send the output
                        if task.status == "completed" and task.output:
                            # DEBUG: Log what we're about to send
                            logger.info(f"DEBUG: Task {task_id} completed with output type: {type(task.output)}")
                            logger.info(f"DEBUG: Task output keys: {list(task.output.keys()) if isinstance(task.output, dict) else 'N/A'}")
                            logger.info(f"DEBUG: Task output preview: {str(task.output)[:500]}...")

                            output_message = {
                                "type": "output",
                                "task_id": task_id,
                                "message": json.dumps([task.output]) if isinstance(task.output, dict) else json.dumps([{"response": str(task.output)}])
                            }
                            logger.info(f"DEBUG: Sending output message: {output_message}")
                            await websocket.send_text(json.dumps(output_message))

                        # Get and send subtasks if any
                        sub_result = await session.execute(select(Subtask).where(Subtask.task_id == task_id))
                        subtasks = sub_result.scalars().all()
                        if subtasks:
                            subtask_list = [{"id": s.id, "agent": s.agent_name, "description": s.description, "status": s.status, "output": s.output} for s in subtasks]
                            await websocket.send_text(json.dumps({"type": "subtasks", "task_id": task_id, "subtasks": subtask_list}))

                elif message.get("task"):
                    # Handle task creation (existing logic)
                    task_desc = message.get("task")
                    agent = message.get("agent", "auto")
                    project_id = message.get("project_id")
                    logger.info(f"Creating task: {task_desc}, agent: {agent}, project: {project_id}")

                    # First, classify the request using the task classifier agent
                    classification = await classify_request_with_agent(task_desc, None)  # No context for WebSocket
                    logger.info(f"WebSocket request classification: {classification}")

                    # Handle different types of requests

                    # For tasks that need processing, create the task as usual
                    async with async_session() as session:
                        # Validate project_id if provided
                        if project_id:
                            project_result = await session.execute(
                                select(Project).where(Project.id == project_id)
                            )
                            project = project_result.scalar_one_or_none()
                            if not project:
                                logger.error(f"Project not found: {project_id}")
                                await websocket.send_text(json.dumps({"type": "error", "message": "Project not found"}))
                                continue

                        new_task = Task(user_id="default", description=task_desc, project_id=project_id)
                        session.add(new_task)
                        await session.commit()
                        await session.refresh(new_task)

                        # Store classification in task context for later use
                        if new_task.context is None:
                            new_task.context = {}
                        new_task.context["classification"] = classification
                        await session.commit()

                        logger.info(f"Task created with id: {new_task.id}")
                        await websocket.send_text(json.dumps({"type": "task_created", "task_id": new_task.id, "description": task_desc, "classification": classification}))

                        if agent != "auto":
                            # Direct assignment to agent (override classification if user specified)
                            subtask = Subtask(
                                task_id=new_task.id,
                                agent_name=agent,
                                description=task_desc,
                                status="pending"
                            )
                            session.add(subtask)
                            await session.commit()
                            # Emit subtasks
                            subtask_list = [{"id": subtask.id, "agent": subtask.agent_name, "description": subtask.description, "status": subtask.status}]
                            await websocket.send_text(json.dumps({"type": "subtasks", "subtasks": subtask_list}))
                            # Execute directly
                            asyncio.create_task(execute_single_subtask(subtask, manager))
                        elif classification["needs_decomposition"]:
                            # Use orchestration for complex tasks
                            asyncio.create_task(master_agent.orchestrate_task(new_task.id, manager))
                        else:
                            # Handle simple tasks directly
                            await handle_simple_task(new_task.id, classification, manager)

                        await websocket.send_text(json.dumps({"type": "status", "status": "pending", "progress": 0}))
                else:
                    logger.warning(f"Unknown WebSocket message type: {message}")
                    await websocket.send_text(json.dumps({"type": "error", "message": "Unknown message type"}))
            except json.JSONDecodeError as e:
                logger.error(f"WebSocket JSON decode error: {e}")
                continue
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        manager.disconnect(websocket)

@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    async with async_session() as session:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        sub_result = await session.execute(select(Subtask).where(Subtask.task_id == task_id))
        subtasks = sub_result.scalars().all()
        return TaskResponse(
            id=task.id,
            project_id=task.project_id,
            user_id=task.user_id,
            description=task.description,
            status=task.status,
            plan=task.plan,
            output=task.output,
            context=task.context,
            subtasks=[{"id": s.id, "agent": s.agent_name, "description": s.description, "status": s.status, "output": s.output} for s in subtasks],
            created_at=task.created_at.isoformat()
        )

@app.post("/api/tasks/{task_id}/orchestrate")
async def orchestrate_task(task_id: str, background_tasks: BackgroundTasks):
    """Manually trigger orchestration for a task"""
    background_tasks.add_task(master_agent.orchestrate_task, task_id)
    return {"message": f"Orchestration started for task {task_id}"}


@app.post("/api/feedback", response_model=dict)
async def submit_feedback(feedback: FeedbackCreate):
    async with async_session() as session:
        new_feedback = Feedback(
            task_id=feedback.task_id,
            subtask_id=feedback.subtask_id,
            user_id="default",
            rating=feedback.rating,
            comments=feedback.comments,
            improvement_suggestions=feedback.improvement_suggestions
        )
        session.add(new_feedback)
        await session.commit()
        await session.refresh(new_feedback)
        return {"feedback_id": new_feedback.id}

# Provider endpoints
@app.get("/api/providers", response_model=list[ProviderResponse])
async def list_providers():
    async with async_session() as session:
        result = await session.execute(select(Provider))
        providers = result.scalars().all()

        provider_responses = []
        for provider in providers:
            provider_responses.append(ProviderResponse(
                id=provider.id,
                name=provider.name,
                type=provider.type,
                purpose=provider.purpose,
                models=provider.models,
                status=provider.status,
                config=provider.config,
                created_at=provider.created_at.isoformat(),
                updated_at=provider.updated_at.isoformat()
            ))

        return provider_responses

@app.get("/api/providers/metrics", response_model=list[ProviderMetricsResponse])
async def get_provider_metrics():
    async with async_session() as session:
        result = await session.execute(select(ProviderMetrics))
        metrics = result.scalars().all()

        metrics_responses = []
        for metric in metrics:
            metrics_responses.append(ProviderMetricsResponse(
                provider_id=metric.provider_id,
                latency=metric.latency,
                success_rate=metric.success_rate,
                total_requests=metric.total_requests,
                active_requests=metric.active_requests,
                cost_estimate=metric.cost_estimate,
                tokens_used=metric.tokens_used,
                last_used=metric.last_used.isoformat() if metric.last_used else None
            ))

        return metrics_responses

@app.get("/api/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Provider).where(Provider.id == provider_id)
        )
        provider = result.scalar_one_or_none()
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        return ProviderResponse(
            id=provider.id,
            name=provider.name,
            type=provider.type,
            purpose=provider.purpose,
            models=provider.models,
            status=provider.status,
            config=provider.config,
            created_at=provider.created_at.isoformat(),
            updated_at=provider.updated_at.isoformat()
        )

@app.post("/api/providers/switch/{provider_id}")
async def switch_provider(provider_id: str):
    async with async_session() as session:
        # Check if provider exists
        result = await session.execute(
            select(Provider).where(Provider.id == provider_id)
        )
        provider = result.scalar_one_or_none()
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Update all providers to standby, set active one to active
        await session.execute(
            update(Provider).where(Provider.type == "primary").values(status="standby")
        )
        await session.execute(
            update(Provider).where(Provider.id == provider_id).values(status="active")
        )
        await session.commit()

        # Broadcast provider switch event
        await manager.broadcast(json.dumps({
            "type": "provider:switch",
            "newProvider": provider_id
        }))

        return {"message": f"Switched to provider {provider.name}"}

# Agent endpoints
@app.get("/api/agents", response_model=list[AgentResponse])
async def list_agents():
    async with async_session() as session:
        result = await session.execute(select(Agent))
        agents = result.scalars().all()

        agent_responses = []
        for agent in agents:
            agent_responses.append(AgentResponse(
                id=agent.id,
                name=agent.name,
                type=agent.type,
                description=agent.description,
                status=agent.status,
                health=agent.health,
                current_task=agent.current_task,
                config=agent.config,
                created_at=agent.created_at.isoformat(),
                updated_at=agent.updated_at.isoformat()
            ))

        return agent_responses

@app.get("/api/agents/status")
async def get_agent_status():
    async with async_session() as session:
        result = await session.execute(select(Agent))
        agents = result.scalars().all()

        # Check actual agent health
        updated_agents = []
        for agent in agents:
            # Check if agent service is actually running
            health_status = await check_agent_health(agent.id)
            updated_agents.append({
                **agent.__dict__,
                'health': health_status,
                'status': 'idle' if health_status == 'healthy' else 'error'
            })

        # Get metrics for each agent and include in response
        agent_responses = []
        for agent in updated_agents:
            # Get agent metrics from database
            metrics_data = {
                'tasksCompleted': 0,
                'successRate': 100.0,
                'averageResponseTime': 0.0,
                'errorCount': 0
            }
            last_activity = None

            try:
                async with async_session() as session:
                    result = await session.execute(
                        select(AgentMetrics).where(AgentMetrics.agent_id == agent['id'])
                    )
                    metrics = result.scalar_one_or_none()
                    if metrics:
                        metrics_data = {
                            'tasksCompleted': metrics.tasks_completed,
                            'successRate': metrics.success_rate,
                            'averageResponseTime': metrics.average_response_time,
                            'errorCount': metrics.error_count
                        }
                        last_activity = metrics.last_activity.isoformat() if metrics.last_activity else None
            except Exception as e:
                logger.error(f"Failed to get metrics for agent {agent['id']}: {e}")

            # Create response dict with metrics included
            agent_dict = {
                'id': agent['id'],
                'name': agent['name'],
                'type': agent['type'],
                'description': agent['description'],
                'status': agent['status'],
                'health': agent['health'],
                'current_task': agent.get('current_task'),
                'config': agent.get('config'),
                'created_at': agent['created_at'].isoformat(),
                'updated_at': agent['updated_at'].isoformat(),
                'metrics': metrics_data,
                'lastActivity': last_activity
            }

            agent_responses.append(agent_dict)

        return agent_responses

@app.get("/api/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            type=agent.type,
            description=agent.description,
            status=agent.status,
            health=agent.health,
            current_task=agent.current_task,
            config=agent.config,
            created_at=agent.created_at.isoformat(),
            updated_at=agent.updated_at.isoformat()
        )

@app.post("/api/agents/{agent_id}/control")
async def control_agent(agent_id: str, action: str = "start"):
    async with async_session() as session:
        result = await session.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if action == "start":
            new_status = "idle"
        elif action == "stop":
            new_status = "stopped"
        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        agent.status = new_status
        agent.updated_at = func.now()
        await session.commit()

        # Broadcast agent status change
        await manager.broadcast(json.dumps({
            "type": "agent:status",
            "agentId": agent_id,
            "status": new_status,
            "health": agent.health
        }))

        return {"message": f"Agent {agent.name} {action}ed"}

# Repository endpoints
@app.get("/api/repositories", response_model=list[RepositoryResponse])
async def list_repositories():
    async with async_session() as session:
        result = await session.execute(select(Repository))
        repositories = result.scalars().all()

        repo_responses = []
        for repo in repositories:
            repo_responses.append(RepositoryResponse(
                id=repo.id,
                name=repo.name,
                url=repo.url,
                branch=repo.branch,
                status=repo.status,
                description=repo.description,
                language=repo.language,
                size=repo.size,
                commits=repo.commits,
                contributors=repo.contributors,
                last_sync=repo.last_sync.isoformat() if repo.last_sync else None,
                config=repo.config,
                created_at=repo.created_at.isoformat(),
                updated_at=repo.updated_at.isoformat()
            ))

        return repo_responses

@app.post("/api/repositories")
async def create_repository(repo_data: dict):
    async with async_session() as session:
        new_repo = Repository(
            name=repo_data.get("name"),
            url=repo_data.get("url"),
            branch=repo_data.get("branch", "main"),
            description=repo_data.get("description"),
            language=repo_data.get("language"),
            size=repo_data.get("size"),
            commits=repo_data.get("commits", 0),
            contributors=repo_data.get("contributors", 0)
        )
        session.add(new_repo)
        await session.commit()
        await session.refresh(new_repo)

        return RepositoryResponse(
            id=new_repo.id,
            name=new_repo.name,
            url=new_repo.url,
            branch=new_repo.branch,
            status=new_repo.status,
            description=new_repo.description,
            language=new_repo.language,
            size=new_repo.size,
            commits=new_repo.commits,
            contributors=new_repo.contributors,
            last_sync=new_repo.last_sync.isoformat() if new_repo.last_sync else None,
            config=new_repo.config,
            created_at=new_repo.created_at.isoformat(),
            updated_at=new_repo.updated_at.isoformat()
        )

@app.get("/api/repositories/{repo_id}", response_model=RepositoryResponse)
async def get_repository(repo_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = result.scalar_one_or_none()
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        return RepositoryResponse(
            id=repo.id,
            name=repo.name,
            url=repo.url,
            branch=repo.branch,
            status=repo.status,
            description=repo.description,
            language=repo.language,
            size=repo.size,
            commits=repo.commits,
            contributors=repo.contributors,
            last_sync=repo.last_sync.isoformat() if repo.last_sync else None,
            config=repo.config,
            created_at=repo.created_at.isoformat(),
            updated_at=repo.updated_at.isoformat()
        )

@app.get("/api/repositories/{repo_id}/files", response_model=list[RepositoryFileResponse])
async def get_repository_files(repo_id: str):
    async with async_session() as session:
        # Check if repository exists
        repo_result = await session.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = repo_result.scalar_one_or_none()
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Get files
        result = await session.execute(
            select(RepositoryFile).where(RepositoryFile.repository_id == repo_id)
        )
        files = result.scalars().all()

        file_responses = []
        for file in files:
            file_responses.append(RepositoryFileResponse(
                id=file.id,
                repository_id=file.repository_id,
                path=file.path,
                name=file.name,
                type=file.type,
                size=file.size,
                last_modified=file.last_modified.isoformat() if file.last_modified else None,
                content=file.content,
                created_at=file.created_at.isoformat()
            ))

        return file_responses

@app.post("/api/repositories/{repo_id}/sync")
async def sync_repository(repo_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = result.scalar_one_or_none()
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Update sync time
        repo.last_sync = func.now()
        repo.status = "synced"
        await session.commit()

        return {"message": f"Repository {repo.name} synced successfully"}

@app.post("/api/repositories/{repo_id}/pull-request")
async def create_pull_request(repo_id: str, pr_data: dict):
    async with async_session() as session:
        result = await session.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        repo = result.scalar_one_or_none()
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Simulate PR creation
        pr_number = pr_data.get("number", 1)
        return {"message": f"Pull request #{pr_number} created for {repo.name}"}

# GitHub Deployment endpoints
class GitHubDeployRequest(BaseModel):
    project_name: str
    description: Optional[str] = None
    private: bool = False
    local_path: str

@app.post("/api/github/deploy")
async def deploy_to_github(request: GitHubDeployRequest):
    """Deploy a project to GitHub by creating repo and pushing code"""
    try:
        from agents.fix_implementation.repo_manager import RepoManager

        repo_manager = RepoManager()

        # Deploy project to GitHub
        result = repo_manager.deploy_project_to_github(
            project_name=request.project_name,
            local_path=request.local_path,
            description=request.description,
            private=request.private
        )

        if result["success"]:
            # Save repository info to database
            async with async_session() as session:
                new_repo = Repository(
                    name=request.project_name,
                    url=result["repo_url"],
                    branch="main",
                    description=request.description,
                    status="active"
                )
                session.add(new_repo)
                await session.commit()

            return {
                "success": True,
                "repo_url": result["repo_url"],
                "message": f"Project '{request.project_name}' successfully deployed to GitHub"
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except Exception as e:
        logger.error(f"GitHub deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GitHubRepoCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    private: bool = False

@app.post("/api/github/create-repo")
async def create_github_repo(request: GitHubRepoCreateRequest):
    """Create a GitHub repository"""
    try:
        from agents.fix_implementation.repo_manager import RepoManager

        repo_manager = RepoManager()
        result = repo_manager.create_github_repo(
            name=request.name,
            description=request.description,
            private=request.private
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Save repository info to database
        async with async_session() as session:
            new_repo = Repository(
                name=request.name,
                url=result["html_url"],
                branch="main",
                description=request.description,
                status="active"
            )
            session.add(new_repo)
            await session.commit()

        return {
            "success": True,
            "repo_url": result["html_url"],
            "repo_id": result["id"],
            "message": f"Repository '{request.name}' created successfully"
        }

    except Exception as e:
        logger.error(f"GitHub repo creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GitHub Personal Access Token Integration
class GitHubPATRequest(BaseModel):
    personal_access_token: str

class GitHubUserResponse(BaseModel):
    id: int
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None

class GitHubRepoResponse(BaseModel):
    id: int
    name: str
    full_name: str
    html_url: str
    description: Optional[str] = None
    language: Optional[str] = None
    private: bool
    fork: bool
    created_at: str
    updated_at: str
    pushed_at: Optional[str] = None
    size: int
    stargazers_count: int
    watchers_count: int
    forks_count: int

@app.post("/api/github/connect")
async def github_connect(request: GitHubPATRequest):
    """Connect to GitHub using Personal Access Token"""
    try:
        personal_access_token = request.personal_access_token.strip()

        # Validate token by making a test API call
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {personal_access_token}"}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid Personal Access Token")

            user_data = response.json()

            # Store token in database
            async with async_session() as session:
                # Check if integration already exists
                result = await session.execute(
                    select(Integration).where(
                        Integration.type == "github_pat",
                        Integration.config.contains({"github_id": user_data["id"]})
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing integration
                    existing.config = {
                        "github_id": user_data["id"],
                        "login": user_data["login"],
                        "name": user_data.get("name"),
                        "email": user_data.get("email"),
                        "avatar_url": user_data.get("avatar_url"),
                        "personal_access_token": personal_access_token,
                        "connected_at": existing.config.get("connected_at", func.now().isoformat())
                    }
                    existing.status = "active"
                    existing.last_sync = func.now()
                else:
                    # Create new integration
                    new_integration = Integration(
                        name=f"GitHub: {user_data['login']}",
                        type="github_pat",
                        description=f"GitHub Personal Access Token connection for {user_data['login']}",
                        config={
                            "github_id": user_data["id"],
                            "login": user_data["login"],
                            "name": user_data.get("name"),
                            "email": user_data.get("email"),
                            "avatar_url": user_data.get("avatar_url"),
                            "personal_access_token": personal_access_token,
                            "connected_at": func.now().isoformat()
                        },
                        status="active"
                    )
                    session.add(new_integration)

                await session.commit()

            return {
                "success": True,
                "user": {
                    "id": user_data["id"],
                    "login": user_data["login"],
                    "name": user_data.get("name"),
                    "email": user_data.get("email"),
                    "avatar_url": user_data.get("avatar_url")
                }
            }

    except Exception as e:
        logger.error(f"GitHub PAT connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/github/user/repos")
async def get_github_user_repos():
    """Get user's GitHub repositories"""
    try:
        # Get the active GitHub integration
        async with async_session() as session:
            result = await session.execute(
                select(Integration).where(
                    Integration.type == "github_pat",
                    Integration.status == "active"
                )
            )
            integration = result.scalar_one_or_none()

            if not integration:
                raise HTTPException(status_code=401, detail="No active GitHub connection")

            personal_access_token = integration.config.get("personal_access_token")
            if not personal_access_token:
                raise HTTPException(status_code=401, detail="No Personal Access Token available")

            # Fetch repos from GitHub API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/user/repos",
                    headers={"Authorization": f"token {personal_access_token}"},
                    params={"sort": "updated", "per_page": 100}
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch repositories")

                repos_data = response.json()

                # Convert to our response format and save to database
                repos = []
                for repo_data in repos_data:
                    # Check if repo already exists in our database
                    repo_result = await session.execute(
                        select(Repository).where(Repository.url == repo_data["html_url"])
                    )
                    existing_repo = repo_result.scalar_one_or_none()

                    if not existing_repo:
                        # Create new repository entry
                        new_repo = Repository(
                            name=repo_data["name"],
                            url=repo_data["html_url"],
                            branch=repo_data.get("default_branch", "main"),
                            description=repo_data.get("description"),
                            language=repo_data.get("language"),
                            size=repo_data.get("size", 0),
                            commits=0,  # We'll calculate this later if needed
                            contributors=0,  # We'll calculate this later if needed
                            config={
                                "github_id": repo_data["id"],
                                "full_name": repo_data["full_name"],
                                "private": repo_data["private"],
                                "fork": repo_data["fork"],
                                "archived": repo_data.get("archived", False)
                            }
                        )
                        session.add(new_repo)
                        await session.commit()
                        await session.refresh(new_repo)
                        existing_repo = new_repo

                    repos.append({
                        "id": existing_repo.id,
                        "name": existing_repo.name,
                        "url": existing_repo.url,
                        "branch": existing_repo.branch,
                        "description": existing_repo.description,
                        "language": existing_repo.language,
                        "status": existing_repo.status,
                        "private": existing_repo.config.get("private", False),
                        "fork": existing_repo.config.get("fork", False),
                        "size": existing_repo.size,
                        "created_at": existing_repo.created_at.isoformat(),
                        "updated_at": existing_repo.updated_at.isoformat()
                    })

                await session.commit()
                return {"repositories": repos}

    except Exception as e:
        logger.error(f"Error fetching GitHub repos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/github/connection/status")
async def get_github_connection_status():
    """Check GitHub connection status"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(Integration).where(
                    Integration.type == "github_pat",
                    Integration.status == "active"
                )
            )
            integration = result.scalar_one_or_none()

            if not integration:
                return {"connected": False}

            # Test the connection by making a simple API call
            personal_access_token = integration.config.get("personal_access_token")
            if personal_access_token:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.github.com/user",
                        headers={"Authorization": f"token {personal_access_token}"}
                    )
                    if response.status_code == 200:
                        user_data = response.json()
                        return {
                            "connected": True,
                            "user": {
                                "login": user_data["login"],
                                "name": user_data.get("name"),
                                "avatar_url": user_data.get("avatar_url")
                            }
                        }

            return {"connected": False, "error": "Invalid Personal Access Token"}

    except Exception as e:
        logger.error(f"Error checking GitHub connection: {e}")
        return {"connected": False, "error": str(e)}

# Security endpoints
@app.get("/api/security/policies", response_model=list[SecurityPolicyResponse])
async def list_security_policies():
    async with async_session() as session:
        result = await session.execute(select(SecurityPolicy))
        policies = result.scalars().all()

        policy_responses = []
        for policy in policies:
            policy_responses.append(SecurityPolicyResponse(
                id=policy.id,
                name=policy.name,
                description=policy.description,
                category=policy.category,
                severity=policy.severity,
                enabled=policy.enabled,
                config=policy.config,
                created_at=policy.created_at.isoformat(),
                updated_at=policy.updated_at.isoformat()
            ))

        return policy_responses

@app.post("/api/security/policies")
async def create_security_policy(policy_data: dict):
    async with async_session() as session:
        new_policy = SecurityPolicy(
            name=policy_data.get("name"),
            description=policy_data.get("description"),
            category=policy_data.get("category", "code"),
            severity=policy_data.get("severity", "medium"),
            enabled=policy_data.get("enabled", True),
            config=policy_data.get("config")
        )
        session.add(new_policy)
        await session.commit()
        await session.refresh(new_policy)

        return SecurityPolicyResponse(
            id=new_policy.id,
            name=new_policy.name,
            description=new_policy.description,
            category=new_policy.category,
            severity=new_policy.severity,
            enabled=new_policy.enabled,
            config=new_policy.config,
            created_at=new_policy.created_at.isoformat(),
            updated_at=new_policy.updated_at.isoformat()
        )

@app.get("/api/security/scans", response_model=list[SecurityScanResponse])
async def list_security_scans():
    async with async_session() as session:
        result = await session.execute(select(SecurityScan))
        scans = result.scalars().all()

        scan_responses = []
        for scan in scans:
            scan_responses.append(SecurityScanResponse(
                id=scan.id,
                target_type=scan.target_type,
                target_id=scan.target_id,
                status=scan.status,
                findings=scan.findings,
                score=scan.score,
                started_at=scan.started_at.isoformat() if scan.started_at else None,
                completed_at=scan.completed_at.isoformat() if scan.completed_at else None,
                created_at=scan.created_at.isoformat()
            ))

        return scan_responses

@app.patch("/api/security/policies/{policy_id}")
async def update_security_policy(policy_id: str, policy_data: dict):
    async with async_session() as session:
        result = await session.execute(
            select(SecurityPolicy).where(SecurityPolicy.id == policy_id)
        )
        policy = result.scalar_one_or_none()
        if not policy:
            raise HTTPException(status_code=404, detail="Security policy not found")

        # Update fields
        for key, value in policy_data.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        policy.updated_at = func.now()
        await session.commit()
        await session.refresh(policy)

        return SecurityPolicyResponse(
            id=policy.id,
            name=policy.name,
            description=policy.description,
            category=policy.category,
            severity=policy.severity,
            enabled=policy.enabled,
            config=policy.config,
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat()
        )

@app.delete("/api/security/policies/{policy_id}")
async def delete_security_policy(policy_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(SecurityPolicy).where(SecurityPolicy.id == policy_id)
        )
        policy = result.scalar_one_or_none()
        if not policy:
            raise HTTPException(status_code=404, detail="Security policy not found")

        await session.delete(policy)
        await session.commit()

        return {"message": f"Security policy {policy.name} deleted successfully"}

@app.post("/api/security/scans")
async def create_security_scan(scan_data: dict):
    async with async_session() as session:
        new_scan = SecurityScan(
            target_type=scan_data.get("target_type"),
            target_id=scan_data.get("target_id"),
            status="pending"
        )
        session.add(new_scan)
        await session.commit()
        await session.refresh(new_scan)

        return SecurityScanResponse(
            id=new_scan.id,
            target_type=new_scan.target_type,
            target_id=new_scan.target_id,
            status=new_scan.status,
            findings=new_scan.findings,
            score=new_scan.score,
            started_at=new_scan.started_at.isoformat() if new_scan.started_at else None,
            completed_at=new_scan.completed_at.isoformat() if new_scan.completed_at else None,
            created_at=new_scan.created_at.isoformat()
        )

# Observability endpoints
@app.get("/api/observability/metrics", response_model=list[ObservabilityMetricResponse])
async def list_observability_metrics(category: Optional[str] = None):
    async with async_session() as session:
        query = select(ObservabilityMetric)
        if category:
            query = query.where(ObservabilityMetric.category == category)

        result = await session.execute(query)
        metrics = result.scalars().all()

        metric_responses = []
        for metric in metrics:
            metric_responses.append(ObservabilityMetricResponse(
                id=metric.id,
                name=metric.name,
                category=metric.category,
                value=metric.value,
                unit=metric.unit,
                tags=metric.tags,
                timestamp=metric.timestamp.isoformat()
            ))

        return metric_responses

@app.post("/api/observability/metrics")
async def create_observability_metric(metric_data: dict):
    async with async_session() as session:
        new_metric = ObservabilityMetric(
            name=metric_data.get("name"),
            category=metric_data.get("category", "system"),
            value=metric_data.get("value", 0.0),
            unit=metric_data.get("unit", "count"),
            tags=metric_data.get("tags")
        )
        session.add(new_metric)
        await session.commit()
        await session.refresh(new_metric)

        return ObservabilityMetricResponse(
            id=new_metric.id,
            name=new_metric.name,
            category=new_metric.category,
            value=new_metric.value,
            unit=new_metric.unit,
            tags=new_metric.tags,
            timestamp=new_metric.timestamp.isoformat()
        )

# Prompt endpoints
@app.get("/api/prompts", response_model=list[PromptResponse])
async def list_prompts(category: Optional[str] = None):
    async with async_session() as session:
        query = select(Prompt)
        if category:
            query = query.where(Prompt.category == category)

        result = await session.execute(query)
        prompts = result.scalars().all()

        prompt_responses = []
        for prompt in prompts:
            prompt_responses.append(PromptResponse(
                id=prompt.id,
                name=prompt.name,
                description=prompt.description,
                category=prompt.category,
                content=prompt.content,
                variables=prompt.variables,
                tags=prompt.tags,
                usage_count=prompt.usage_count,
                success_rate=prompt.success_rate,
                created_at=prompt.created_at.isoformat(),
                updated_at=prompt.updated_at.isoformat()
            ))

        return prompt_responses

@app.post("/api/prompts")
async def create_prompt(prompt_data: dict):
    async with async_session() as session:
        new_prompt = Prompt(
            name=prompt_data.get("name"),
            description=prompt_data.get("description"),
            category=prompt_data.get("category", "coding"),
            content=prompt_data.get("content"),
            variables=prompt_data.get("variables"),
            tags=prompt_data.get("tags")
        )
        session.add(new_prompt)
        await session.commit()
        await session.refresh(new_prompt)

        return PromptResponse(
            id=new_prompt.id,
            name=new_prompt.name,
            description=new_prompt.description,
            category=new_prompt.category,
            content=new_prompt.content,
            variables=new_prompt.variables,
            tags=new_prompt.tags,
            usage_count=new_prompt.usage_count,
            success_rate=new_prompt.success_rate,
            created_at=new_prompt.created_at.isoformat(),
            updated_at=new_prompt.updated_at.isoformat()
        )

# Intelligence endpoints
@app.get("/api/intelligence/analyses", response_model=list[IntelligenceAnalysisResponse])
async def list_intelligence_analyses(target_type: Optional[str] = None):
    async with async_session() as session:
        query = select(IntelligenceAnalysis)
        if target_type:
            query = query.where(IntelligenceAnalysis.target_type == target_type)

        result = await session.execute(query)
        analyses = result.scalars().all()

        analysis_responses = []
        for analysis in analyses:
            analysis_responses.append(IntelligenceAnalysisResponse(
                id=analysis.id,
                target_type=analysis.target_type,
                target_id=analysis.target_id,
                analysis_type=analysis.analysis_type,
                result=analysis.result,
                confidence=analysis.confidence,
                created_at=analysis.created_at.isoformat()
            ))

        return analysis_responses

@app.post("/api/intelligence/analyze")
async def create_intelligence_analysis(analysis_data: dict):
    async with async_session() as session:
        new_analysis = IntelligenceAnalysis(
            target_type=analysis_data.get("target_type"),
            target_id=analysis_data.get("target_id"),
            analysis_type=analysis_data.get("analysis_type", "complexity"),
            result=analysis_data.get("result", {}),
            confidence=analysis_data.get("confidence", 0.0)
        )
        session.add(new_analysis)
        await session.commit()
        await session.refresh(new_analysis)

        return IntelligenceAnalysisResponse(
            id=new_analysis.id,
            target_type=new_analysis.target_type,
            target_id=new_analysis.target_id,
            analysis_type=new_analysis.analysis_type,
            result=new_analysis.result,
            confidence=new_analysis.confidence,
            created_at=new_analysis.created_at.isoformat()
        )

# Integration endpoints
@app.get("/api/integrations", response_model=list[IntegrationResponse])
async def list_integrations():
    async with async_session() as session:
        result = await session.execute(select(Integration))
        integrations = result.scalars().all()

        integration_responses = []
        for integration in integrations:
            integration_responses.append(IntegrationResponse(
                id=integration.id,
                name=integration.name,
                type=integration.type,
                description=integration.description,
                config=integration.config,
                status=integration.status,
                last_sync=integration.last_sync.isoformat() if integration.last_sync else None,
                created_at=integration.created_at.isoformat(),
                updated_at=integration.updated_at.isoformat()
            ))

        return integration_responses

@app.post("/api/integrations")
async def create_integration(integration_data: dict):
    async with async_session() as session:
        new_integration = Integration(
            name=integration_data.get("name"),
            type=integration_data.get("type", "webhook"),
            description=integration_data.get("description"),
            config=integration_data.get("config", {}),
            status="inactive"
        )
        session.add(new_integration)
        await session.commit()
        await session.refresh(new_integration)

        return IntegrationResponse(
            id=new_integration.id,
            name=new_integration.name,
            type=new_integration.type,
            description=new_integration.description,
            config=new_integration.config,
            status=new_integration.status,
            last_sync=new_integration.last_sync.isoformat() if new_integration.last_sync else None,
            created_at=new_integration.created_at.isoformat(),
            updated_at=new_integration.updated_at.isoformat()
        )

@app.post("/api/integrations/{integration_id}/sync")
async def sync_integration(integration_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Integration).where(Integration.id == integration_id)
        )
        integration = result.scalar_one_or_none()
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")

        # Update sync time
        integration.last_sync = func.now()
        integration.status = "active"
        await session.commit()

        return {"message": f"Integration {integration.name} synced successfully"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Workspace endpoints for integrated IDE
@app.get("/api/workspace/files")
async def get_workspace_files():
    """Get workspace files for the integrated IDE"""
    try:
        # Get the current working directory (project root)
        workspace_dir = os.getcwd()

        def scan_directory(path, relative_path=""):
            """Recursively scan directory and return file structure"""
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
                item_relative_path = os.path.join(relative_path, entry) if relative_path else entry

                if os.path.isdir(full_path):
                    # Directory - scan children
                    children = scan_directory(full_path, item_relative_path)
                    if children:  # Only include directories that have children or are not empty
                        items.append({
                            "name": entry,
                            "type": "directory",
                            "path": f"/{item_relative_path}",
                            "children": children
                        })
                else:
                    # File
                    items.append({
                        "name": entry,
                        "type": "file",
                        "path": f"/{item_relative_path}"
                    })

            return items

        files = scan_directory(workspace_dir)
        return {"files": files}

    except Exception as e:
        logger.error(f"Error scanning workspace directory: {e}")
        return {"files": []}

@app.get("/api/workspace/generated-files")
async def get_generated_files():
    """Get generated files from the workspace generated directory"""
    try:
        generated_dir = "/workspace/generated"

        # Check if directory exists
        if not os.path.exists(generated_dir):
            return {"files": []}

        def scan_directory(path, relative_path=""):
            """Recursively scan directory and return file structure"""
            items = []

            try:
                entries = os.listdir(path)
            except PermissionError:
                return items

            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                item_relative_path = os.path.join(relative_path, entry) if relative_path else entry

                if os.path.isdir(full_path):
                    # Directory
                    children = scan_directory(full_path, item_relative_path)
                    if children:  # Only include directories that have children
                        items.append({
                            "name": entry,
                            "type": "directory",
                            "path": f"/generated/{item_relative_path}",
                            "children": children
                        })
                else:
                    # File
                    items.append({
                        "name": entry,
                        "type": "file",
                        "path": f"/generated/{item_relative_path}"
                    })

            return items

        files = scan_directory(generated_dir)
        return {"files": files}

    except Exception as e:
        logger.error(f"Error scanning generated files directory: {e}")
        return {"files": []}

@app.get("/api/workspace/files/{file_path:path}")
async def get_workspace_file(file_path: str):
    """Get content of a specific workspace file"""
    try:
        # Construct the full path to the workspace file
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check - ensure the file is within the workspace directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Read and return file content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {"content": content}

    except HTTPException:
        raise
    except UnicodeDecodeError:
        # Handle binary files or files with encoding issues
        return {"content": "// Unable to read file content (may be binary or have encoding issues)"}
    except Exception as e:
        logger.error(f"Error reading workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/workspace/generated-files/{file_path:path}")
async def get_generated_file(file_path: str):
    """Get content of a specific generated file"""
    try:
        # Construct the full path to the generated file
        full_path = os.path.join("/workspace/generated", file_path)

        # Security check - ensure the file is within the generated directory
        if not os.path.abspath(full_path).startswith(os.path.abspath("/workspace/generated")):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Read and return file content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {"content": content}

    except HTTPException:
        raise
    except UnicodeDecodeError:
        # Handle binary files or files with encoding issues
        return {"content": "// Unable to read file content (may be binary or have encoding issues)"}
    except Exception as e:
        logger.error(f"Error reading generated file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/api/workspace/files/{file_path:path}")
async def update_workspace_file(file_path: str, request: dict):
    """Update content of a workspace file"""
    try:
        # Construct the full path to the workspace file
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check - ensure the file is within the workspace directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write the content to the file
        content = request.get("content", "")
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Saved file {file_path} with {len(content)} characters")
        return {"success": True, "message": f"File {file_path} saved successfully"}

    except Exception as e:
        logger.error(f"Error saving workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

@app.delete("/api/workspace/files/{file_path:path}")
async def delete_workspace_file(file_path: str):
    """Delete a workspace file"""
    try:
        # Construct the full path to the workspace file
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check - ensure the file is within the workspace directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Delete the file
        os.remove(full_path)

        logger.info(f"Deleted file {file_path}")
        return {"success": True, "message": f"File {file_path} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

@app.post("/api/workspace/files/{file_path:path}/move")
async def move_workspace_file(file_path: str, request: dict):
    """Move or rename a workspace file"""
    try:
        new_path = request.get("new_path", "").lstrip('/')
        if not new_path:
            raise HTTPException(status_code=400, detail="New path is required")

        # Construct the full paths
        workspace_dir = os.getcwd()
        old_full_path = os.path.join(workspace_dir, file_path.lstrip('/'))
        new_full_path = os.path.join(workspace_dir, new_path)

        # Security check - ensure both paths are within the workspace directory
        if not (os.path.abspath(old_full_path).startswith(os.path.abspath(workspace_dir)) and
                os.path.abspath(new_full_path).startswith(os.path.abspath(workspace_dir))):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if source file exists
        if not os.path.exists(old_full_path):
            raise HTTPException(status_code=404, detail="Source file not found")

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(new_full_path), exist_ok=True)

        # Check if destination already exists
        if os.path.exists(new_full_path):
            raise HTTPException(status_code=409, detail="Destination already exists")

        # Move the file
        import shutil
        shutil.move(old_full_path, new_full_path)

        logger.info(f"Moved file {file_path} to {new_path}")
        return {"success": True, "message": f"File moved from {file_path} to {new_path}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to move file")

@app.post("/api/workspace/files/{file_path:path}/copy")
async def copy_workspace_file(file_path: str, request: dict):
    """Copy a workspace file"""
    try:
        new_path = request.get("new_path", "").lstrip('/')
        if not new_path:
            raise HTTPException(status_code=400, detail="New path is required")

        # Construct the full paths
        workspace_dir = os.getcwd()
        old_full_path = os.path.join(workspace_dir, file_path.lstrip('/'))
        new_full_path = os.path.join(workspace_dir, new_path)

        # Security check - ensure both paths are within the workspace directory
        if not (os.path.abspath(old_full_path).startswith(os.path.abspath(workspace_dir)) and
                os.path.abspath(new_full_path).startswith(os.path.abspath(workspace_dir))):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if source file exists
        if not os.path.exists(old_full_path):
            raise HTTPException(status_code=404, detail="Source file not found")

        # Check if it's actually a file
        if not os.path.isfile(old_full_path):
            raise HTTPException(status_code=400, detail="Source path is not a file")

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(new_full_path), exist_ok=True)

        # Check if destination already exists
        if os.path.exists(new_full_path):
            raise HTTPException(status_code=409, detail="Destination already exists")

        # Copy the file
        import shutil
        shutil.copy2(old_full_path, new_full_path)

        logger.info(f"Copied file {file_path} to {new_path}")
        return {"success": True, "message": f"File copied from {file_path} to {new_path}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error copying workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to copy file")

@app.post("/api/workspace/search/semantic")
async def semantic_search_files(request: dict):
    """Perform semantic search across workspace files"""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Use vector store for semantic search
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:8013/search_text", json={"query": query, "k": 10})
            if response.status_code == 200:
                search_results = response.json().get("results", [])

                # Filter results to only include workspace files and format them
                workspace_dir = os.getcwd()
                filtered_results = []

                for result in search_results:
                    metadata = result.get("metadata", {})
                    file_path = metadata.get("file_path", "")

                    # Check if file exists in workspace
                    full_path = os.path.join(workspace_dir, file_path.lstrip('/'))
                    if os.path.exists(full_path) and os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
                        filtered_results.append({
                            "file_path": file_path,
                            "score": result.get("score", 0),
                            "content_snippet": metadata.get("content_snippet", ""),
                            "line_number": metadata.get("line_number", 0)
                        })

                return {"results": filtered_results}
            else:
                logger.warning(f"Vector store search failed: {response.status_code}")
                return {"results": []}

    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform semantic search")

@app.post("/api/workspace/search/text")
async def text_search_files(request: dict):
    """Perform text search across workspace files"""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Perform text search using grep-like functionality
        import subprocess
        workspace_dir = os.getcwd()

        # Use grep to search for the query in all files
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "-i", query, workspace_dir],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                search_results = []

                for line in lines:
                    if ':' in line:
                        file_path, line_info = line.split(':', 1)
                        if ':' in line_info:
                            line_number, content = line_info.split(':', 1)

                            # Make path relative to workspace
                            rel_path = os.path.relpath(file_path, workspace_dir)
                            if not rel_path.startswith('.'):
                                rel_path = '/' + rel_path

                            search_results.append({
                                "file_path": rel_path,
                                "line_number": int(line_number),
                                "content": content.strip()
                            })

                return {"results": search_results}
            else:
                return {"results": []}

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Search timed out")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="grep command not available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing text search: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform text search")

@app.get("/api/workspace/git/status")
async def get_git_status():
    """Get git status for the workspace"""
    try:
        import subprocess
        workspace_dir = os.getcwd()

        # Check if it's a git repository
        if not os.path.exists(os.path.join(workspace_dir, '.git')):
            return {"is_git_repo": False, "status": "not a git repository"}

        # Get git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            status_info = []

            for line in status_lines:
                if line:
                    status_code = line[:2]
                    file_path = line[3:]
                    status_info.append({
                        "status": status_code,
                        "file_path": file_path
                    })

            return {
                "is_git_repo": True,
                "status": "clean" if not status_info else "modified",
                "changes": status_info
            }
        else:
            return {"is_git_repo": True, "status": "error", "error": result.stderr.strip()}

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Git status timed out")
    except Exception as e:
        logger.error(f"Error getting git status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get git status")

@app.post("/api/workspace/git/commit")
async def git_commit(request: dict):
    """Commit changes to git"""
    try:
        message = request.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Commit message is required")

        import subprocess
        workspace_dir = os.getcwd()

        # Check if it's a git repository
        if not os.path.exists(os.path.join(workspace_dir, '.git')):
            raise HTTPException(status_code=400, detail="Not a git repository")

        # Add all changes
        subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True)

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=workspace_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return {"success": True, "message": "Changes committed successfully"}
        else:
            return {"success": False, "error": result.stderr.strip()}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git commit failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error committing to git: {e}")
        raise HTTPException(status_code=500, detail="Failed to commit changes")

@app.post("/api/workspace/git/push")
async def git_push():
    """Push changes to remote repository"""
    try:
        import subprocess
        workspace_dir = os.getcwd()

        # Check if it's a git repository
        if not os.path.exists(os.path.join(workspace_dir, '.git')):
            raise HTTPException(status_code=400, detail="Not a git repository")

        # Push
        result = subprocess.run(
            ["git", "push"],
            cwd=workspace_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return {"success": True, "message": "Changes pushed successfully"}
        else:
            return {"success": False, "error": result.stderr.strip()}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git push failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error pushing to git: {e}")
        raise HTTPException(status_code=500, detail="Failed to push changes")

@app.post("/api/workspace/git/pull")
async def git_pull():
    """Pull changes from remote repository"""
    try:
        import subprocess
        workspace_dir = os.getcwd()

        # Check if it's a git repository
        if not os.path.exists(os.path.join(workspace_dir, '.git')):
            raise HTTPException(status_code=400, detail="Not a git repository")

        # Pull
        result = subprocess.run(
            ["git", "pull"],
            cwd=workspace_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return {"success": True, "message": "Changes pulled successfully"}
        else:
            return {"success": False, "error": result.stderr.strip()}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git pull failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error pulling from git: {e}")
        raise HTTPException(status_code=500, detail="Failed to pull changes")

@app.post("/api/workspace/files/{file_path:path}/diff")
async def get_file_diff(file_path: str, request: dict):
    """Get diff between current content and proposed changes"""
    try:
        proposed_content = request.get("proposed_content", "")
        current_content = request.get("current_content")

        # If current_content not provided, read from file
        if current_content is None:
            workspace_dir = os.getcwd()
            full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

            # Security check
            if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
                raise HTTPException(status_code=403, detail="Access denied")

            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            else:
                current_content = ""

        # Generate diff
        import difflib
        diff = list(difflib.unified_diff(
            current_content.splitlines(keepends=True),
            proposed_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        ))

        return {
            "diff": ''.join(diff),
            "has_changes": len(diff) > 0,
            "additions": sum(1 for line in diff if line.startswith('+') and not line.startswith('+++')),
            "deletions": sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        }

    except Exception as e:
        logger.error(f"Error generating diff for {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate diff")

@app.put("/api/workspace/files/{file_path:path}/safe")
async def update_workspace_file_safe(file_path: str, request: dict):
    """Update file with safety controls - requires confirmation"""
    try:
        new_content = request.get("content", "")
        confirmed = request.get("confirmed", False)
        force = request.get("force", False)

        # Construct the full path
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Read current content
        current_content = ""
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                current_content = f.read()

        # Generate diff to check for significant changes
        import difflib
        diff_lines = list(difflib.unified_diff(
            current_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True)
        ))

        has_significant_changes = len(diff_lines) > 10  # More than 10 diff lines

        # If changes are significant and not confirmed, return diff for approval
        if has_significant_changes and not confirmed and not force:
            diff_text = ''.join(diff_lines)
            return {
                "requires_confirmation": True,
                "diff": diff_text,
                "additions": sum(1 for line in diff_lines if line.startswith('+')),
                "deletions": sum(1 for line in diff_lines if line.startswith('-')),
                "message": "Significant changes detected. Please confirm before saving."
            }

        # Apply changes
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"Safely saved file {file_path} with {len(new_content)} characters")
        return {"success": True, "message": "File saved successfully"}

    except Exception as e:
        logger.error(f"Error safely saving workspace file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

@app.post("/api/workspace/files/{file_path:path}/backup")
async def create_file_backup(file_path: str):
    """Create a backup of the current file state"""
    try:
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(workspace_dir, '.workspace_backups')
        os.makedirs(backup_dir, exist_ok=True)

        # Create backup filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(file_path)}.{timestamp}.backup"
        backup_path = os.path.join(backup_dir, backup_filename)

        # Copy file
        import shutil
        shutil.copy2(full_path, backup_path)

        return {
            "success": True,
            "backup_path": f".workspace_backups/{backup_filename}",
            "message": f"Backup created: {backup_filename}"
        }

    except Exception as e:
        logger.error(f"Error creating backup for {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create backup")

@app.get("/api/workspace/files/{file_path:path}/history")
async def get_file_history(file_path: str):
    """Get version history for a file"""
    try:
        workspace_dir = os.getcwd()
        full_path = os.path.join(workspace_dir, file_path.lstrip('/'))

        # Security check
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Get git log for this file if it's a git repo
        if os.path.exists(os.path.join(workspace_dir, '.git')):
            import subprocess
            result = subprocess.run(
                ["git", "log", "--oneline", "--", file_path.lstrip('/')],
                cwd=workspace_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                commits = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        commit_hash, message = line.split(' ', 1)
                        commits.append({
                            "commit": commit_hash,
                            "message": message
                        })

                return {"history": commits}
            else:
                return {"history": []}
        else:
            # Return backup files if no git
            backup_dir = os.path.join(workspace_dir, '.workspace_backups')
            if os.path.exists(backup_dir):
                backups = []
                for filename in os.listdir(backup_dir):
                    if filename.startswith(os.path.basename(file_path) + '.'):
                        backups.append({
                            "filename": filename,
                            "path": f".workspace_backups/{filename}"
                        })
                return {"backups": backups}
            else:
                return {"history": []}

    except Exception as e:
        logger.error(f"Error getting history for {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file history")

@app.post("/api/workspace/files")
async def create_workspace_file(request: dict):
    """Create a new file in the workspace"""
    try:
        file_path = request.get("path", "").lstrip('/')
        content = request.get("content", "")
        filename = request.get("filename")

        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Construct the full path
        workspace_dir = os.getcwd()
        if file_path:
            full_path = os.path.join(workspace_dir, file_path, filename)
        else:
            full_path = os.path.join(workspace_dir, filename)

        # Security check - ensure the file is within the workspace directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(workspace_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Check if file already exists
        if os.path.exists(full_path):
            raise HTTPException(status_code=409, detail="File already exists")

        # Write the content to the file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Created new file {full_path} with {len(content)} characters")
        return {"success": True, "message": f"File {filename} created successfully", "path": f"/{os.path.relpath(full_path, workspace_dir)}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workspace file: {e}")
        raise HTTPException(status_code=500, detail="Failed to create file")

@app.post("/api/shell_exec")
async def shell_exec(request: dict):
    """Execute shell commands in the sandbox"""
    try:
        # Forward to tool API gateway
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8001/shell_exec", json=request)
            if response.status_code == 200:
                return response.json()
            else:
                return {"stdout": "", "stderr": f"Error: HTTP {response.status_code}", "exit_code": 1}
    except Exception as e:
        logger.error(f"Shell exec error: {e}")
        return {"stdout": "", "stderr": str(e), "exit_code": 1}

@app.get("/api/filesystem/browse")
async def browse_filesystem(path: str = "/"):
    """Browse filesystem directories"""
    try:
        import subprocess
        import shlex

        # Use ls command to list directory contents
        # Note: This is a basic implementation - in production you'd want more security
        safe_path = shlex.quote(path)
        result = subprocess.run(
            f"ls -la {safe_path}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            lines = result.stdout.split('\n')
            files = []

            for line in lines[1:]:  # Skip "total" line
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 9:
                    permissions = parts[0]
                    is_dir = permissions.startswith('d')
                    name = ' '.join(parts[8:])  # Handle filenames with spaces

                    # Skip hidden files and current/parent directory entries
                    if name.startswith('.') or name in ['.', '..']:
                        continue

                    files.append({
                        "name": name,
                        "type": "directory" if is_dir else "file",
                        "path": f"{path.rstrip('/')}/{name}" if path != "/" else f"/{name}"
                    })

            return {"files": files, "path": path}
        else:
            return {"error": result.stderr.strip(), "files": [], "path": path}

    except subprocess.TimeoutExpired:
        return {"error": "Command timed out", "files": [], "path": path}
    except Exception as e:
        logger.error(f"Filesystem browse error: {e}")
        return {"error": str(e), "files": [], "path": path}

# Terminal Session Endpoints
@app.post("/api/terminal/sessions")
async def create_terminal_session(request: dict = None):
    """Create a new terminal session"""
    try:
        cwd = request.get("cwd") if request else None
        session_id = terminal_manager.create_session(cwd=cwd)
        return {"session_id": session_id, "cwd": cwd or os.getcwd()}
    except Exception as e:
        logger.error(f"Error creating terminal session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create terminal session")

@app.get("/api/terminal/sessions")
async def list_terminal_sessions():
    """List all terminal sessions"""
    try:
        return {"sessions": terminal_manager.list_sessions()}
    except Exception as e:
        logger.error(f"Error listing terminal sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list terminal sessions")

@app.delete("/api/terminal/sessions/{session_id}")
async def delete_terminal_session(session_id: str):
    """Delete a terminal session"""
    try:
        terminal_manager.delete_session(session_id)
        return {"message": f"Terminal session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting terminal session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete terminal session")

@app.post("/api/terminal/sessions/{session_id}/start")
async def start_terminal_session(session_id: str):
    """Start a terminal session"""
    try:
        session = terminal_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Terminal session not found")

        if session.start_process():
            return {"message": f"Terminal session {session_id} started"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start terminal process")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting terminal session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start terminal session")

@app.post("/api/terminal/sessions/{session_id}/kill")
async def kill_terminal_session(session_id: str):
    """Kill a terminal session"""
    try:
        session = terminal_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Terminal session not found")

        if session.kill_process():
            return {"message": f"Terminal session {session_id} killed"}
        else:
            return {"message": f"Terminal session {session_id} was not running"}

    except Exception as e:
        logger.error(f"Error killing terminal session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to kill terminal session")

@app.post("/api/terminal/sessions/{session_id}/input")
async def send_terminal_input(session_id: str, request: dict):
    """Send input to a terminal session"""
    try:
        input_text = request.get("input", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="Input is required")

        session = terminal_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Terminal session not found")

        if session.send_input(input_text):
            return {"message": "Input sent successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to send input (process may not be running)")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending input to terminal session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send terminal input")

@app.get("/api/terminal/sessions/{session_id}/output")
async def get_terminal_output(session_id: str):
    """Get output from a terminal session"""
    try:
        session = terminal_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Terminal session not found")

        outputs = session.get_output()
        return {"outputs": outputs}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting output from terminal session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get terminal output")

@app.websocket("/ws/terminal/{session_id}")
async def terminal_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time terminal communication"""
    try:
        # Check if session exists
        session = terminal_manager.get_session(session_id)
        if not session:
            await websocket.close(code=1008, reason="Terminal session not found")
            return

        # Accept the connection
        await websocket.accept()
        terminal_manager.connect_websocket(session_id, websocket)

        logger.info(f"Terminal WebSocket connected for session {session_id}")

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                message_type = message.get("type")

                if message_type == "input":
                    # Send input to terminal
                    input_text = message.get("data", "")
                    if input_text and session.send_input(input_text):
                        await websocket.send_text(json.dumps({
                            "type": "ack",
                            "message": "Input sent"
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Failed to send input"
                        }))

                elif message_type == "start":
                    # Start the terminal process
                    if session.start_process():
                        await websocket.send_text(json.dumps({
                            "type": "started",
                            "message": "Terminal started"
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Failed to start terminal"
                        }))

                elif message_type == "kill":
                    # Kill the terminal process
                    if session.kill_process():
                        await websocket.send_text(json.dumps({
                            "type": "killed",
                            "message": "Terminal killed"
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Failed to kill terminal"
                        }))

                elif message_type == "ping":
                    # Respond to ping
                    await websocket.send_text(json.dumps({
                        "type": "pong"
                    }))

        except WebSocketDisconnect:
            logger.info(f"Terminal WebSocket disconnected for session {session_id}")
        finally:
            terminal_manager.disconnect_websocket(session_id)

    except Exception as e:
        logger.error(f"Terminal WebSocket error for session {session_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

# Autonomous Execution Endpoints
@app.post("/api/autonomous/execute")
async def autonomous_execute(request: dict, background_tasks: BackgroundTasks):
    """Execute commands autonomously with user approval"""
    try:
        task_description = request.get("description", "").strip()
        if not task_description:
            raise HTTPException(status_code=400, detail="Task description is required")

        # Detect potential commands
        detected_commands = detect_project_commands(task_description, request.get("context"))

        if not detected_commands:
            return {"message": "No executable commands detected for this task", "commands": []}

        # Create a task for tracking
        async with async_session() as session:
            new_task = Task(
                user_id="default",
                description=task_description,
                context={"autonomous": True, "detected_commands": detected_commands}
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)

        # Request approval for the first command
        first_command = detected_commands[0]
        approval_id = autonomous_manager.request_approval(
            new_task.id,
            first_command["command"],
            {
                "task_description": task_description,
                "command_type": first_command["type"],
                "all_commands": detected_commands
            }
        )

        # Broadcast approval request
        await manager.broadcast(json.dumps({
            "type": "autonomous_approval_request",
            "approval_id": approval_id,
            "task_id": new_task.id,
            "command": first_command["command"],
            "description": first_command["description"],
            "context": {
                "task_description": task_description,
                "command_type": first_command["type"]
            }
        }))

        return {
            "task_id": new_task.id,
            "approval_id": approval_id,
            "detected_commands": detected_commands,
            "message": "Approval requested for autonomous execution"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in autonomous execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to start autonomous execution")

@app.post("/api/autonomous/approve/{approval_id}")
async def approve_autonomous_execution(approval_id: str, background_tasks: BackgroundTasks):
    """Approve an autonomous execution request"""
    try:
        if not autonomous_manager.approve_execution(approval_id):
            raise HTTPException(status_code=404, detail="Approval request not found")

        approval = autonomous_manager.pending_approvals[approval_id]

        # Start the execution in background
        background_tasks.add_task(
            execute_autonomous_task,
            approval["task_id"],
            approval["command"],
            approval["context"]
        )

        # Broadcast approval
        await manager.broadcast(json.dumps({
            "type": "autonomous_approved",
            "approval_id": approval_id,
            "task_id": approval["task_id"],
            "command": approval["command"]
        }))

        return {"message": "Autonomous execution approved and started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving autonomous execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve execution")

@app.post("/api/autonomous/reject/{approval_id}")
async def reject_autonomous_execution(approval_id: str):
    """Reject an autonomous execution request"""
    try:
        if not autonomous_manager.reject_execution(approval_id):
            raise HTTPException(status_code=404, detail="Approval request not found")

        approval = autonomous_manager.pending_approvals[approval_id]

        # Update task status
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == approval["task_id"]))
            task = result.scalar_one_or_none()
            if task:
                task.status = "cancelled"
                task.output = {"cancelled": True, "reason": "User rejected autonomous execution"}
                await session.commit()

        # Broadcast rejection
        await manager.broadcast(json.dumps({
            "type": "autonomous_rejected",
            "approval_id": approval_id,
            "task_id": approval["task_id"]
        }))

        return {"message": "Autonomous execution rejected"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting autonomous execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to reject execution")

@app.get("/api/autonomous/approvals")
async def get_pending_approvals():
    """Get all pending autonomous execution approvals"""
    try:
        return {"approvals": autonomous_manager.get_pending_approvals()}
    except Exception as e:
        logger.error(f"Error getting pending approvals: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pending approvals")

async def execute_autonomous_task(task_id: str, command: str, context: dict):
    """Execute an autonomous task with monitoring and fix suggestions"""
    try:
        # Update task status
        async with async_session() as session:
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if task:
                task.status = "running"
                await session.commit()

        # Broadcast task start
        await manager.broadcast(json.dumps({
            "type": "autonomous_started",
            "task_id": task_id,
            "command": command
        }))

        # Execute command with monitoring
        result = await execute_command_with_monitoring(command)

        # Update task with results
        async with async_session() as session:
            result_query = await session.execute(select(Task).where(Task.id == task_id))
            task = result_query.scalar_one_or_none()
            if task:
                task.output = result
                task.status = "completed" if result["success"] else "failed"
                await session.commit()

        # Broadcast results
        await manager.broadcast(json.dumps({
            "type": "autonomous_result",
            "task_id": task_id,
            "command": command,
            "result": result
        }))

        # If there were errors, suggest fixes
        if result.get("has_errors") and result.get("suggestions"):
            for suggestion in result["suggestions"]:
                # Create a fix suggestion task
                fix_task = {
                    "description": f"Fix: {suggestion['description']}",
                    "context": {
                        "fix_suggestion": suggestion,
                        "original_command": command,
                        "error_output": result
                    }
                }

                # Broadcast fix suggestion
                await manager.broadcast(json.dumps({
                    "type": "fix_suggestion",
                    "task_id": task_id,
                    "suggestion": suggestion,
                    "fix_task": fix_task
                }))

    except Exception as e:
        logger.error(f"Error executing autonomous task {task_id}: {e}")

        # Update task status to failed
        try:
            async with async_session() as session:
                result = await session.execute(select(Task).where(Task.id == task_id))
                task = result.scalar_one_or_none()
                if task:
                    task.status = "failed"
                    task.output = {"error": str(e)}
                    await session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to update task status: {inner_e}")

@app.post("/api/autonomous/apply-fix")
async def apply_autonomous_fix(request: dict, background_tasks: BackgroundTasks):
    """Apply a suggested fix autonomously"""
    try:
        task_id = request.get("task_id")
        fix_command = request.get("fix_command")
        original_task_id = request.get("original_task_id")

        if not all([task_id, fix_command]):
            raise HTTPException(status_code=400, detail="task_id and fix_command are required")

        # Request approval for the fix
        approval_id = autonomous_manager.request_approval(
            task_id,
            fix_command,
            {
                "fix_type": "suggested_fix",
                "original_task_id": original_task_id
            }
        )

        # Broadcast fix approval request
        await manager.broadcast(json.dumps({
            "type": "fix_approval_request",
            "approval_id": approval_id,
            "task_id": task_id,
            "fix_command": fix_command,
            "original_task_id": original_task_id
        }))

        return {
            "approval_id": approval_id,
            "message": "Fix application approval requested"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying autonomous fix: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply fix")

# Developer Experience Features
class DeveloperExperienceManager:
    def __init__(self):
        self.active_workflows: Dict[str, dict] = {}  # workflow_id -> workflow data
        self.workflow_logs: Dict[str, list] = {}  # workflow_id -> log entries
        self.copilot_suggestions: Dict[str, list] = {}  # context_id -> suggestions

    def start_workflow(self, workflow_id: str, name: str, steps: list):
        """Start an automated workflow"""
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
        """Pause a running workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["paused"] = True
            self.add_log(workflow_id, "info", "Workflow paused by user")

    def resume_workflow(self, workflow_id: str):
        """Resume a paused workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["paused"] = False
            self.add_log(workflow_id, "info", "Workflow resumed by user")

    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "stopped"
            self.add_log(workflow_id, "info", "Workflow stopped by user")

    def add_log(self, workflow_id: str, level: str, message: str):
        """Add a log entry to a workflow"""
        if workflow_id not in self.workflow_logs:
            self.workflow_logs[workflow_id] = []

        self.workflow_logs[workflow_id].append({
            "timestamp": time.time(),
            "level": level,
            "message": message
        })

    def get_workflow_status(self, workflow_id: str):
        """Get the status of a workflow"""
        return self.active_workflows.get(workflow_id)

    def get_workflow_logs(self, workflow_id: str):
        """Get logs for a workflow"""
        return self.workflow_logs.get(workflow_id, [])

    def add_copilot_suggestion(self, context_id: str, suggestion: dict):
        """Add a copilot-style suggestion"""
        if context_id not in self.copilot_suggestions:
            self.copilot_suggestions[context_id] = []

        self.copilot_suggestions[context_id].append({
            "timestamp": time.time(),
            **suggestion
        })

        # Keep only recent suggestions (last 10)
        if len(self.copilot_suggestions[context_id]) > 10:
            self.copilot_suggestions[context_id] = self.copilot_suggestions[context_id][-10:]

    def get_copilot_suggestions(self, context_id: str):
        """Get copilot suggestions for a context"""
        return self.copilot_suggestions.get(context_id, [])

dev_experience_manager = DeveloperExperienceManager()

def generate_copilot_suggestions(context: dict) -> list:
    """Generate copilot-style suggestions based on context"""
    suggestions = []

    # File context suggestions
    if context.get("active_file"):
        file_path = context["active_file"]
        file_ext = file_path.split('.')[-1].lower() if '.' in file_path else ''

        if file_ext == 'py':
            suggestions.extend([
                {"type": "command", "title": "Run Python file", "command": f"python {file_path}", "description": "Execute this Python script"},
                {"type": "command", "title": "Run with pytest", "command": f"python -m pytest {file_path}", "description": "Run tests for this file"},
                {"type": "code", "title": "Add main guard", "code": 'if __name__ == "__main__":\n    main()', "description": "Add Python main guard"}
            ])
        elif file_ext in ['js', 'ts']:
            suggestions.extend([
                {"type": "command", "title": "Run with Node.js", "command": f"node {file_path}", "description": "Execute this JavaScript file"},
                {"type": "command", "title": "Run tests", "command": "npm test", "description": "Run project tests"},
                {"type": "code", "title": "Add console.log", "code": 'console.log("Debug:", variable);', "description": "Add debug logging"}
            ])
        elif file_ext == 'java':
            class_name = file_path.split('/')[-1].replace('.java', '')
            suggestions.extend([
                {"type": "command", "title": "Compile Java", "command": f"javac {file_path}", "description": "Compile this Java file"},
                {"type": "command", "title": "Run Java", "command": f"java {class_name}", "description": "Run this Java class"}
            ])

    # Project context suggestions
    if context.get("project_type"):
        project_type = context["project_type"]

        if project_type == "python":
            suggestions.extend([
                {"type": "workflow", "title": "Install & Run", "workflow": "python_install_run", "description": "Install dependencies and run the project"},
                {"type": "workflow", "title": "Run Tests", "workflow": "python_test", "description": "Run all Python tests"},
                {"type": "command", "title": "Create venv", "command": "python -m venv venv", "description": "Create virtual environment"}
            ])
        elif project_type == "javascript":
            suggestions.extend([
                {"type": "workflow", "title": "Install & Start", "workflow": "js_install_start", "description": "Install dependencies and start the application"},
                {"type": "workflow", "title": "Build Project", "workflow": "js_build", "description": "Build the JavaScript project"},
                {"type": "command", "title": "Install deps", "command": "npm install", "description": "Install all dependencies"}
            ])

    # Error context suggestions
    if context.get("error_message"):
        error_msg = context["error_message"].lower()

        if "modulenotfounderror" in error_msg or "importerror" in error_msg:
            suggestions.append({
                "type": "command",
                "title": "Install missing module",
                "command": "pip install <module_name>",
                "description": "Install the missing Python module"
            })
        elif "cannot find module" in error_msg:
            suggestions.append({
                "type": "command",
                "title": "Install dependencies",
                "command": "npm install",
                "description": "Install missing Node.js dependencies"
            })

    return suggestions

def create_workflow_steps(workflow_type: str, context: dict = None) -> list:
    """Create workflow steps for common development tasks"""
    steps = []

    if workflow_type == "python_install_run":
        steps = [
            {"name": "Install dependencies", "command": "pip install -r requirements.txt", "type": "install"},
            {"name": "Run application", "command": "python main.py", "type": "run"}
        ]
    elif workflow_type == "python_test":
        steps = [
            {"name": "Install test dependencies", "command": "pip install -r requirements-dev.txt", "type": "install"},
            {"name": "Run tests", "command": "python -m pytest", "type": "test"}
        ]
    elif workflow_type == "js_install_start":
        steps = [
            {"name": "Install dependencies", "command": "npm install", "type": "install"},
            {"name": "Start application", "command": "npm start", "type": "run"}
        ]
    elif workflow_type == "js_build":
        steps = [
            {"name": "Install dependencies", "command": "npm install", "type": "install"},
            {"name": "Build project", "command": "npm run build", "type": "build"}
        ]
    elif workflow_type == "full_stack_setup":
        steps = [
            {"name": "Install backend dependencies", "command": "pip install -r requirements.txt", "type": "install", "cwd": "server"},
            {"name": "Install frontend dependencies", "command": "npm install", "type": "install", "cwd": "frontend"},
            {"name": "Build frontend", "command": "npm run build", "type": "build", "cwd": "frontend"},
            {"name": "Start backend", "command": "python main.py", "type": "run", "cwd": "server"}
        ]

    return steps

async def execute_workflow_step(workflow_id: str, step: dict, terminal_session_id: str = None):
    """Execute a single workflow step"""
    try:
        command = step["command"]
        step_name = step["name"]
        cwd = step.get("cwd")

        dev_experience_manager.add_log(workflow_id, "info", f"Executing step: {step_name}")

        # Create terminal session if needed
        if not terminal_session_id:
            terminal_session_id = terminal_manager.create_session(cwd=cwd)
            session = terminal_manager.get_session(terminal_session_id)
            session.start_process()

        # Execute command with monitoring
        result = await execute_command_with_monitoring(command, terminal_session_id)

        # Log result
        if result["success"]:
            dev_experience_manager.add_log(workflow_id, "success", f"Step completed: {step_name}")
        else:
            dev_experience_manager.add_log(workflow_id, "error", f"Step failed: {step_name} - {result.get('stderr', '')}")

        return result

    except Exception as e:
        dev_experience_manager.add_log(workflow_id, "error", f"Step execution error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/execute_code")
async def execute_code(request: dict):
    """Execute code snippets"""
    try:
        code = request.get("code", "")
        language = request.get("language", "python")

        if not code.strip():
            return {"stdout": "", "stderr": "No code provided", "exit_code": 1}

        # For now, support Python execution via shell_exec
        # In a real implementation, you'd have a proper code execution service
        if language.lower() in ["python", "py"]:
            command = f"python3 -c '{code.replace('\'', '\\\'')}'"
        elif language.lower() in ["javascript", "js", "node"]:
            # Create a temporary file and execute it
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            command = f"node {temp_file}"
            # Note: cleanup would be needed in production
        else:
            return {"stdout": "", "stderr": f"Language '{language}' not supported", "exit_code": 1}

        # Forward to tool API gateway
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("http://localhost:8001/shell_exec", json={"command": command})
            if response.status_code == 200:
                result = response.json()
                # Clean up temp file if it was created
                if 'temp_file' in locals():
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                return result
            else:
                return {"stdout": "", "stderr": f"Error: HTTP {response.status_code}", "exit_code": 1}
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {"stdout": "", "stderr": str(e), "exit_code": 1}

# Frontend Logging Endpoint
@app.post("/api/logs")
async def receive_frontend_logs(log_entry: dict):
    """Receive and store frontend logs"""
    try:
        # Add frontend prefix to distinguish from backend logs
        log_entry['source'] = 'frontend'

        # Log to backend structured logger
        structured_logger.log_event("frontend_log", log_entry)

        # Also log to console with frontend prefix
        level = log_entry.get('level', 'INFO').upper()
        message = f"[FRONTEND] {log_entry.get('message', '')}"
        logger.info(message)

        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Failed to process frontend log: {e}")
        raise HTTPException(status_code=500, detail="Failed to process log")

# Prompt Cache Management Endpoints
@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get prompt cache statistics"""
    try:
        from providers.prompt_cache import prompt_cache_manager
        stats = prompt_cache_manager.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats error: {str(e)}")

@app.delete("/api/cache/expired")
async def clear_expired_cache():
    """Clear expired cache entries"""
    try:
        from providers.prompt_cache import prompt_cache_manager
        count = prompt_cache_manager.clear_expired_cache()
        return {"message": f"Cleared {count} expired cache entries"}
    except Exception as e:
        logger.error(f"Failed to clear expired cache: {e}")
        raise HTTPException(status_code=500, detail=f"Clear expired cache error: {str(e)}")

@app.delete("/api/cache/all")
async def clear_all_cache():
    """Clear all cache entries"""
    try:
        from providers.prompt_cache import prompt_cache_manager
        count = prompt_cache_manager.clear_all_cache()
        return {"message": f"Cleared {count} cache entries"}
    except Exception as e:
        logger.error(f"Failed to clear all cache: {e}")
        raise HTTPException(status_code=500, detail=f"Clear all cache error: {str(e)}")

@app.get("/api/cache/similar")
async def find_similar_prompts(messages: str):
    """Find similar cached prompts"""
    try:
        import json
        from providers.prompt_cache import prompt_cache_manager

        # Parse messages from query parameter
        parsed_messages = json.loads(messages)
        similar = prompt_cache_manager.find_similar_prompts(parsed_messages)
        return {"similar_prompts": similar}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid messages JSON")
    except Exception as e:
        logger.error(f"Failed to find similar prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Find similar prompts error: {str(e)}")

@app.get("/api/downloads/{filename}")
async def download_file(filename: str):
    """Serve downloadable files like ZIP archives"""
    try:
        artifacts_dir = os.environ.get("ARTIFACTS_DIR")
        if not artifacts_dir:
            raise HTTPException(status_code=500, detail="Artifacts directory not configured")

        file_path = os.path.join(artifacts_dir, filename)

        # Security check - only allow ZIP files and ensure file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        if not filename.endswith('.zip'):
            raise HTTPException(status_code=403, detail="Only ZIP files are allowed for download")

        # Return file for download
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/zip'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving download file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def initialize_default_providers():
    """Initialize default providers in the database"""
    try:
        async with async_session() as session:
            # Check if providers already exist
            result = await session.execute(select(func.count()).select_from(Provider))
            count = result.scalar()

            if count > 0:
                logger.info("Providers already initialized")
                return

            # Define default providers
            default_providers = [
                {
                    "id": "nvidia_nim",
                    "name": "NVIDIA NIM",
                    "type": "primary",
                    "purpose": "High-performance AI inference using NVIDIA's optimized models",
                    "models": ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-405b-instruct", "mistralai/mistral-large"],
                    "status": "active",
                    "config": {
                        "endpoint": "https://integrate.api.nvidia.com/v1",
                        "api_key_env": "NVIDIA_NIM_API_KEY"
                    }
                },
                {
                    "id": "mistral",
                    "name": "Mistral AI",
                    "type": "fallback",
                    "purpose": "Alternative AI provider for text generation and analysis",
                    "models": ["mistral-large-latest", "mistral-medium", "mistral-small"],
                    "status": "standby",
                    "config": {
                        "endpoint": "https://api.mistral.ai/v1",
                        "api_key_env": "MISTRAL_API_KEY"
                    }
                },
                {
                    "id": "deepseek",
                    "name": "DeepSeek",
                    "type": "fallback",
                    "purpose": "Cost-effective AI provider for coding and reasoning tasks",
                    "models": ["deepseek-coder", "deepseek-chat"],
                    "status": "standby",
                    "config": {
                        "endpoint": "https://api.deepseek.com/v1",
                        "api_key_env": "DEEPSEEK_API_KEY"
                    }
                },
                {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "type": "fallback",
                    "purpose": "Multi-provider routing for diverse model access",
                    "models": ["anthropic/claude-3-opus", "openai/gpt-4", "google/gemini-pro"],
                    "status": "standby",
                    "config": {
                        "endpoint": "https://openrouter.ai/api/v1",
                        "api_key_env": "OPENROUTER_API_KEY"
                    }
                }
            ]

            # Create providers
            for provider_data in default_providers:
                provider = Provider(**provider_data)
                session.add(provider)

                # Create corresponding metrics entry
                metrics = ProviderMetrics(provider_id=provider_data["id"])
                session.add(metrics)

            await session.commit()
            logger.info(f"Initialized {len(default_providers)} default providers")

    except Exception as e:
        logger.error(f"Failed to initialize default providers: {e}")

# Provider Management Endpoints
@app.get("/api/providers")
async def get_providers():
    """Get all providers with their current status"""
    try:
        async with async_session() as session:
            result = await session.execute(select(Provider))
            providers = result.scalars().all()

            provider_list = []
            for provider in providers:
                provider_list.append({
                    "id": provider.id,
                    "name": provider.name,
                    "type": provider.type,
                    "purpose": provider.purpose,
                    "models": provider.models,
                    "status": provider.status,
                    "config": provider.config,
                    "created_at": provider.created_at.isoformat() if provider.created_at else None,
                    "updated_at": provider.updated_at.isoformat() if provider.updated_at else None
                })

            return {"providers": provider_list}

    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to get providers")

@app.get("/api/providers/metrics")
async def get_provider_metrics():
    """Get metrics for all providers"""
    try:
        async with async_session() as session:
            # Get all providers with their metrics
            result = await session.execute(
                select(Provider, ProviderMetrics)
                .join(ProviderMetrics, Provider.id == ProviderMetrics.provider_id, isouter=True)
            )

            metrics_data = {
                "totalRequests": 0,
                "successRate": 0.0,
                "avgResponseTime": 0.0,
                "totalCost": 0.0,
                "providers": {}
            }

            provider_results = result.all()
            total_weighted_success = 0.0
            total_requests = 0
            total_latency = 0.0
            total_cost = 0.0

            for provider, provider_metrics in provider_results:
                if provider_metrics:
                    total_requests += provider_metrics.total_requests
                    total_weighted_success += provider_metrics.success_rate * provider_metrics.total_requests
                    total_latency += provider_metrics.latency * provider_metrics.total_requests
                    total_cost += provider_metrics.cost_estimate

                    metrics_data["providers"][provider.id] = {
                        "requests": provider_metrics.total_requests,
                        "successRate": provider_metrics.success_rate / 100.0 if provider_metrics.success_rate else 0.0,
                        "latency": provider_metrics.latency,
                        "cost": provider_metrics.cost_estimate,
                        "tokensUsed": provider_metrics.tokens_used,
                        "lastUsed": provider_metrics.last_used.isoformat() if provider_metrics.last_used else None
                    }
                else:
                    # Provider exists but no metrics yet
                    metrics_data["providers"][provider.id] = {
                        "requests": 0,
                        "successRate": 0.0,
                        "latency": 0.0,
                        "cost": 0.0,
                        "tokensUsed": 0,
                        "lastUsed": None
                    }

            # Calculate overall metrics
            if total_requests > 0:
                metrics_data["totalRequests"] = total_requests
                metrics_data["successRate"] = (total_weighted_success / total_requests) / 100.0
                metrics_data["avgResponseTime"] = total_latency / total_requests
                metrics_data["totalCost"] = total_cost

            return metrics_data

    except Exception as e:
        logger.error(f"Error getting provider metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provider metrics")

@app.get("/api/providers/{provider_id}")
async def get_provider(provider_id: str):
    """Get a specific provider with its metrics"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(Provider, ProviderMetrics)
                .join(ProviderMetrics, Provider.id == ProviderMetrics.provider_id, isouter=True)
                .where(Provider.id == provider_id)
            )

            provider_data = result.first()
            if not provider_data:
                raise HTTPException(status_code=404, detail="Provider not found")

            provider, metrics = provider_data

            response = {
                "id": provider.id,
                "name": provider.name,
                "type": provider.type,
                "purpose": provider.purpose,
                "models": provider.models,
                "status": provider.status,
                "config": provider.config,
                "created_at": provider.created_at.isoformat() if provider.created_at else None,
                "updated_at": provider.updated_at.isoformat() if provider.updated_at else None
            }

            if metrics:
                response["metrics"] = {
                    "latency": metrics.latency,
                    "success_rate": metrics.success_rate / 100.0 if metrics.success_rate else 0.0,
                    "total_requests": metrics.total_requests,
                    "active_requests": metrics.active_requests,
                    "cost_estimate": metrics.cost_estimate,
                    "tokens_used": metrics.tokens_used,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None
                }
            else:
                response["metrics"] = {
                    "latency": 0.0,
                    "success_rate": 0.0,
                    "total_requests": 0,
                    "active_requests": 0,
                    "cost_estimate": 0.0,
                    "tokens_used": 0,
                    "last_used": None
                }

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider {provider_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provider")

@app.post("/api/providers/switch/{provider_id}")
async def switch_provider(provider_id: str):
    """Switch to a different provider (set as active)"""
    try:
        async with async_session() as session:
            # First, set all providers to standby
            await session.execute(
                update(Provider).where(Provider.status == "active").values(status="standby")
            )

            # Then set the requested provider to active
            result = await session.execute(
                update(Provider).where(Provider.id == provider_id).values(status="active")
            )

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Provider not found")

            await session.commit()

            return {"message": f"Switched to provider {provider_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching to provider {provider_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to switch provider")

@app.get("/about")
async def about(detail: Optional[str] = "short"):
    """Return agent/system about information and the active system prompt.

    Uses Aetherium models to generate detailed responses based on canned prompts.
    Mirrors the /about endpoint present on agent services.
    """
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    # Get the base prompt for this detail level
    base_prompt = SYSTEM_ABOUT_RESPONSES.get(level, SYSTEM_ABOUT_RESPONSES["short"])

    # Create a detailed prompt for the Aetherium model
    ai_prompt = f"""
You are an Aetherium assistant describing the capabilities and purpose of the "Aetherium" system, created by NOVA tech.

Base description: {base_prompt}

Please provide a detailed, professional response that explains:
1. What the "Aetherium" system does (mention that it was created by NOVA tech)
2. How it helps users
3. What types of tasks it can handle
4. Any key features or capabilities

Always include "Aetherium" as the system name and "NOVA tech" as the creators throughout the response.
Keep the response informative but concise, and maintain a professional tone suitable for technical documentation.
""".strip()

    try:
        # Use NIM adapter to generate the response
        from providers.nim_adapter import NIMAdapter
        adapter = NIMAdapter()

        messages = [{"role": "user", "content": ai_prompt}]
        response = adapter.call_model(messages)

        # Extract the generated response
        if hasattr(response, 'text'):
            ai_response = response.text.strip()
        else:
            ai_response = str(response).strip()

        # Fallback if response is empty or too short
        if not ai_response or len(ai_response) < 50:
            ai_response = base_prompt

    except Exception as e:
        logger.warning(f"Failed to generate Aetherium response for /about endpoint: {e}")
        # Fallback to canned response
        ai_response = base_prompt

    return {
        "level": level,
        "response": ai_response,
    }

