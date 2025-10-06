import json
import os
import sys
import zipfile
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., prompt_builder) and top-level packages resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    # Intentional import-time sys.path modification to ensure repo root is
    # available in containerized or subdirectory run contexts. Lint rule E402
    # (module level import not at top) is intentionally suppressed here.
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional  # noqa: E402
import logging  # noqa: E402
import asyncio  # noqa: E402
import requests  # noqa: E402
from .prompt_builder import PromptBuilder  # noqa: E402
from .patch_generator import PatchGenerator  # noqa: E402
from .repo_manager import RepoManager  # noqa: E402
from .tester import Tester  # noqa: E402
from .safety import SafetyChecker  # noqa: E402
from .tool_orchestrator import ToolOrchestrator  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES, get_agent_prompt, generate_about_response  # noqa: E402
from common.utils import is_running_in_container  # noqa: E402
from common.endpoints import add_health_endpoint  # noqa: E402

app = FastAPI(title="Fix Implementation Agent")

add_health_endpoint(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_zip_file(files_dict, zip_path):
    """Create a ZIP file from a dictionary of file paths and contents"""
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path, content in files_dict.items():
                # Create proper directory structure in ZIP
                zip_file.writestr(file_path, content)
        return True
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {e}")
        return False

def execute_tool(tool_call):
    """Execute a tool call by calling the tool API gateway"""
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    tool_api_url = "http://localhost:8001"  # TOOL_API_GATEWAY_URL

    try:
        if function_name == "git_read_file":
            response = requests.post(f"{tool_api_url}/git_read_file", json=arguments)
        elif function_name == "git_write_file":
            response = requests.post(f"{tool_api_url}/git_write_file", json=arguments)
        elif function_name == "list_files":
            response = requests.post(f"{tool_api_url}/list_files", json=arguments)
        elif function_name == "run_tests":
            response = requests.post(f"{tool_api_url}/run_tests", json=arguments)
        elif function_name == "shell_exec":
            response = requests.post(f"{tool_api_url}/shell_exec", json=arguments)
        else:
            return {"error": f"Unknown tool: {function_name}"}

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Tool API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

# Tool definitions for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "git_read_file",
            "description": "Read a file from a git repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Git repository URL"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"},
                    "file_path": {"type": "string", "description": "Path to the file in the repository"}
                },
                "required": ["repo_url", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_write_file",
            "description": "Write a file to a git repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Git repository URL"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"},
                    "file_path": {"type": "string", "description": "Path to the file in the repository"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                    "commit_message": {"type": "string", "description": "Commit message"}
                },
                "required": ["repo_url", "file_path", "content", "commit_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a git repository directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Git repository URL"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"},
                    "path": {"type": "string", "description": "Path to directory in the repository", "default": "."}
                },
                "required": ["repo_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run tests in a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "Git repository URL"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"},
                    "test_command": {"type": "string", "description": "Test command to run"}
                },
                "required": ["repo_url", "test_command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory", "default": "/workspace"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
                },
                "required": ["command"]
            }
        }
    }
]

class FixRequest(BaseModel):
    repo_url: str
    failing_tests: List[str]
    context: Optional[str] = None
    branch: str = "main"

class ExecuteRequest(BaseModel):
    description: str
    conversation_history: Optional[List[dict]] = None

class CandidatePatch(BaseModel):
    diff: str
    description: str
    confidence: float

class FixResponse(BaseModel):
    candidate_patches: List[CandidatePatch]
    error: Optional[str] = None

@app.post("/fix", response_model=FixResponse)
async def fix_implementation(request: FixRequest):
    try:
        logger.info(f"Starting fix implementation for {request.repo_url}")

        # Build prompt
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_prompt(request.failing_tests, request.context)

        # Generate patches
        patch_generator = PatchGenerator()
        patches = patch_generator.generate_patches(prompt)

        candidate_patches = []

        for patch in patches:
            # Apply patch via repo manager
            repo_manager = RepoManager()
            applied = repo_manager.apply_patch(request.repo_url, request.branch, patch)

            if applied:
                # Test in sandbox
                tester = Tester()
                test_result = tester.run_tests(request.repo_url, request.branch, request.failing_tests)

                if test_result.success:
                    # Safety check
                    safety_checker = SafetyChecker()
                    safe = safety_checker.check_patch(patch)

                    if safe:
                        candidate_patches.append(CandidatePatch(
                            diff=patch,
                            description="Generated patch that fixes failing tests",
                            confidence=0.8  # placeholder
                        ))

        return FixResponse(candidate_patches=candidate_patches)

    except Exception as e:
        logger.error(f"Error in fix implementation: {str(e)}")
        return FixResponse(candidate_patches=[], error=str(e))

@app.post("/execute")
async def execute_task(request: ExecuteRequest):
    try:
        logger.info(f"Executing task: {request.description}")
        desc = (request.description or "").strip()
        logger.info(f"Stripped desc: '{desc}', lower: '{desc.lower()}'")
        logger.info(f"Conversation history provided: {request.conversation_history is not None}")
        if request.conversation_history:
            logger.info(f"Conversation history length: {len(request.conversation_history)}")

        # Simple canned response still allowed for trivial 'hello'
        if desc.lower() == "hello":
            logger.info("Agent special case triggered for 'hello'")
            return {"result": "Hello, world!", "success": True}

        # Determine if this is a project creation task
        is_creation_task = any(keyword in desc.lower() for keyword in ["create", "make", "generate", "build", "write", "implement"])
        logger.info(f"Is creation task: {is_creation_task}")

        # Choose appropriate system prompt based on task type
        if is_creation_task:
            system_prompt = get_agent_prompt("fix_implementation")
        else:
            system_prompt = SYSTEM_PROMPT
        logger.info(f"System prompt length: {len(system_prompt)}")

        # Use NVIDIA NIM exclusively for model calls
        try:
            adapter = NIMAdapter(role="builders")
            logger.info("NIM adapter initialized successfully")
        except Exception as e:
            msg = f"NIM adapter initialization failed: {e}. Ensure NVIDIA_NIM_API_KEY is set in the environment."
            logger.error(msg)
            return {"error": msg}

        # Initialize tool orchestrator for natural language tool parsing
        tool_orchestrator = ToolOrchestrator()

        # Include conversation history if available
        messages = [{"role": "system", "content": system_prompt}]
        if request.conversation_history:
            # Add conversation history as context
            history_text = "Previous conversation:\n"
            for msg in request.conversation_history[-10:]:  # Last 10 messages to avoid token limits
                role = "User" if msg.get("type") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += f"\nCurrent request: {desc}"
            messages.append({"role": "user", "content": history_text})
            logger.info(f"Included conversation history in prompt ({len(request.conversation_history)} messages)")
        else:
            messages.append({"role": "user", "content": desc})

        # Add tool instructions to system prompt for natural language tool usage
        tool_instructions = """
You have access to tools that you can use by describing them in your response. Available tools:

1. run_tests(repo_url="https://github.com/user/repo", test_command="pytest") - Run tests in a repository
2. git_read_file(repo_url="https://github.com/user/repo", file_path="path/to/file") - Read a file from git
3. git_write_file(repo_url="https://github.com/user/repo", file_path="path/to/file", content="file content", commit_message="message") - Write a file to git
4. list_files(repo_url="https://github.com/user/repo", path="directory") - List files in a git repository
5. shell_exec(command="shell command") - Execute a shell command

To use a tool, simply mention it in your response like: "I need to run: run_tests(repo_url='...', test_command='...')"
"""
        messages[0]["content"] += tool_instructions

        logger.info(f"Calling NIM with {len(messages)} messages")
        # Log the actual prompt for debugging
        logger.info(f"System prompt: {messages[0]['content'][:500]}...")
        logger.info(f"User prompt: {messages[1]['content'][:500]}...")
        try:
            response = adapter.call_model(messages, temperature=0.7 if is_creation_task else 0.2)
            logger.info(f"NIM response received: tokens={response.tokens}, latency={response.latency_ms}ms")
            logger.info(f"Response text length: {len(response.text)}")
            # Check if cache was used
            if hasattr(response, 'cache_hit') and response.cache_hit:
                logger.warning("CACHE HIT: Response may be from cache, could be stale")

            # Check if the response indicates an error
            if response.structured_response and isinstance(response.structured_response, list) and len(response.structured_response) > 0:
                first_item = response.structured_response[0]
                if isinstance(first_item, dict) and "error" in first_item:
                    error_msg = first_item.get("error", "")
                    logger.error(f"NIM model returned error in structured response: {error_msg}")
                    return {"error": f"NIM model error: {error_msg}"}

            # Also check if the text itself indicates an error
            if response.text.strip() in ['[{"error": ""}]', '{"error": ""}']:
                logger.error(f"NIM model returned error in text: {response.text}")
                return {"error": f"NIM model error: {response.text}"}

            # Check if structured response is exactly the error array
            logger.info(f"Checking structured response: {response.structured_response}")
            if response.structured_response == [{"error": ""}]:
                logger.error("NIM model returned error array in structured response")
                return {"error": "NIM model error: empty error response"}

            # For creation tasks, generate code and then run tests to validate
            if is_creation_task:
                logger.info("Processing creation task - generating code and running validation tests")
            else:
                # Parse and execute natural language tool requests for non-creation tasks
                tool_results = tool_orchestrator.execute_tool_requests(response.text)
                if tool_results:
                    logger.info(f"Executed {len(tool_results)} tool requests")

                    # Add tool results to messages and call again for follow-up
                    messages.append({"role": "assistant", "content": response.text})
                    tool_results_text = tool_orchestrator.format_tool_results(tool_results)
                    messages.append({"role": "user", "content": f"Tool results:{tool_results_text}\n\nPlease continue with the task based on these results."})

                    # Call model again with tool results
                    response = adapter.call_model(messages, temperature=0.7 if is_creation_task else 0.2)
                    logger.info(f"Second NIM response received after tool execution")

        except Exception as e:
            logger.error("NIM provider call failed: %s", str(e))
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_msg = str(e) if str(e) else "Unknown error (empty exception message)"
            logger.error(f"Final error message that will be returned: '{error_msg}'")
            return {"error": error_msg}

        # For creation tasks, attempt to extract and structure code
        result = response.text
        structured = response.structured_response or {}
        logger.info(f"Raw result: '{result[:200]}...'")
        if is_creation_task:
            logger.info("Processing as creation task")
            # Extract code from markdown format
            import re
            files = {}

            # First, try to extract files with explicit headers like "## filename.ext"
            file_pattern = r'##\s+([^\n]+)\s*\n```(?:\w+)?\n(.*?)\n```'
            file_matches = re.findall(file_pattern, result, re.DOTALL | re.IGNORECASE)

            if file_matches:
                logger.info(f"Found {len(file_matches)} files with explicit headers")
                for filename, content in file_matches:
                    filename = filename.strip()
                    files[filename] = content.strip()
                    logger.info(f"Extracted file: {filename}")

            # If no explicit headers found, try to extract from code blocks and infer filenames
            if not files:
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', result, re.DOTALL)
                logger.info(f"Found {len(code_blocks)} code blocks without headers")

                if code_blocks:
                    for i, block in enumerate(code_blocks):
                        content = block.strip()
                        if not content:
                            continue

                        # Use AI-powered file type detection
                        filename = await detect_file_type_with_ai(content, i)

                        files[filename] = content
                        logger.info(f"Inferred file: {filename}")

            # If still no files, treat whole response as a single file
            if not files:
                logger.info("No code blocks found, treating response as single file")
                files["main.py"] = result.strip()

            # Generate todo list for complex tasks
            todo_list = []
            if len(files) > 3 or len(desc) > 500:  # Complex task detection
                logger.info("Generating todo list for complex task")
                todo_list = [
                    {"id": "analyze_requirements", "description": "Analyze project requirements and scope", "status": "completed", "step": 1},
                    {"id": "design_architecture", "description": "Design system architecture and components", "status": "completed", "step": 2},
                    {"id": "generate_code", "description": f"Generate {len(files)} code files", "status": "completed", "step": 3},
                    {"id": "extract_files", "description": "Extract and structure generated files", "status": "completed", "step": 4},
                    {"id": "save_artifacts", "description": "Save files to artifacts directory", "status": "in_progress", "step": 5},
                    {"id": "create_workspace", "description": "Create workspace copies of files", "status": "pending", "step": 6},
                    {"id": "generate_zip", "description": "Generate downloadable ZIP package", "status": "pending", "step": 7},
                    {"id": "finalize_output", "description": "Finalize and return results", "status": "pending", "step": 8}
                ]

            structured = {"description": result, "files": files, "todo_list": todo_list}

            # For JavaScript files, run basic validation tests using sandbox executor
            test_results = []
            if is_creation_task and files:
                logger.info("Running validation tests for generated code")
                for filename, content in files.items():
                    if filename.endswith('.js'):
                        try:
                            # Create a temporary test file
                            test_code = f"""
{content}

// Basic validation tests
try {{
    // Test 1: Function exists
    if (typeof validateUserInput === 'function') {{
        console.log('✓ Function validateUserInput exists');
    }} else {{
        console.log('✗ Function validateUserInput not found');
    }}

    // Test 2: Basic functionality
    try {{
        const result1 = validateUserInput('test input');
        if (result1 === 'test input') {{
            console.log('✓ Basic validation works');
        }} else {{
            console.log('✗ Basic validation failed');
        }}
    }} catch (e) {{
        console.log('✗ Basic validation threw error:', e.message);
    }}

    // Test 3: Empty string validation
    try {{
        validateUserInput('');
        console.log('✗ Empty string should throw error');
    }} catch (e) {{
        if (e.message.includes('Invalid input')) {{
            console.log('✓ Empty string validation works');
        }} else {{
            console.log('✗ Empty string validation failed:', e.message);
        }}
    }}

    // Test 4: Non-string validation
    try {{
        validateUserInput(123);
        console.log('✗ Non-string should throw error');
    }} catch (e) {{
        if (e.message.includes('string')) {{
            console.log('✓ Non-string validation works');
        }} else {{
            console.log('✗ Non-string validation failed:', e.message);
        }}
    }}

    console.log('Validation tests completed');
}} catch (e) {{
    console.log('Test execution failed:', e.message);
}}
"""
                            # Execute the test using sandbox executor
                            test_result = execute_tool({{
                                "function": {{
                                    "name": "shell_exec",
                                    "arguments": json.dumps({{
                                        "command": f"node -e '{test_code.replace(chr(39), chr(92) + chr(39))}'",
                                        "working_dir": "/tmp",
                                        "timeout": 10
                                    }})
                                }}
                            }})

                            if test_result and "stdout" in test_result:
                                test_results.append({{
                                    "file": filename,
                                    "output": test_result["stdout"],
                                    "success": "✓" in test_result["stdout"] and "✗" not in test_result["stdout"]
                                }})
                                logger.info(f"Test results for {filename}: {test_result['stdout'][:200]}...")
                            else:
                                test_results.append({{
                                    "file": filename,
                                    "output": "Test execution failed",
                                    "success": False
                                }})

                        except Exception as e:
                            logger.error(f"Failed to run tests for {filename}: {e}")
                            test_results.append({{
                                "file": filename,
                                "output": f"Test execution error: {str(e)}",
                                "success": False
                            }})

                # Add test results to structured response
                if test_results:
                    structured["test_results"] = test_results
                    # Update the description to include test results
                    test_summary = "\\n\\n## Test Results\\n"
                    for test in test_results:
                        status = "✅ PASSED" if test["success"] else "❌ FAILED"
                        test_summary += f"**{test['file']}**: {status}\\n"
                        test_summary += f"```\n{test['output']}\n```\\n\\n"
                    structured["description"] += test_summary

        logger.info(f"Final structured response: {structured}")

        # Save generated files to artifacts directory and workspace automatically for creation tasks
        artifacts_saved = []
        workspace_saved = []
        zip_url = None
        if is_creation_task and structured.get("files"):
            # Update todo list - mark save_artifacts as in_progress
            if "todo_list" in structured:
                for todo in structured["todo_list"]:
                    if todo["id"] == "save_artifacts":
                        todo["status"] = "in_progress"
                        break

            # Use ARTIFACTS_DIR if set, otherwise use default directory
            artifacts_dir = os.environ.get("ARTIFACTS_DIR") or "/tmp/codeagent_artifacts"

            # Create artifacts directory if it doesn't exist
            os.makedirs(artifacts_dir, exist_ok=True)

            # Also create workspace generated directory (use temp dir if /workspace not available)
            workspace_generated_dir = "/workspace/generated"
            try:
                os.makedirs(workspace_generated_dir, exist_ok=True)
            except PermissionError:
                # Fallback to temp directory if /workspace is not writable
                workspace_generated_dir = os.path.join(artifacts_dir, "workspace_generated")
                os.makedirs(workspace_generated_dir, exist_ok=True)
                logger.warning(f"Using fallback workspace directory: {workspace_generated_dir}")

            logger.info(f"Saving {len(structured['files'])} files to artifacts directory: {artifacts_dir} and workspace: {workspace_generated_dir}")
            for file_path, content in structured["files"].items():
                try:
                    # Save to artifacts directory
                    artifacts_full_path = os.path.join(artifacts_dir, file_path)
                    os.makedirs(os.path.dirname(artifacts_full_path), exist_ok=True)
                    with open(artifacts_full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    artifacts_saved.append(file_path)

                    # Save to workspace generated directory
                    workspace_full_path = os.path.join(workspace_generated_dir, file_path)
                    os.makedirs(os.path.dirname(workspace_full_path), exist_ok=True)
                    with open(workspace_full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    workspace_saved.append(file_path)

                    logger.info(f"Saved file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save file {file_path}: {e}")

            # Update todo list - mark save_artifacts and create_workspace as completed
            if "todo_list" in structured:
                for todo in structured["todo_list"]:
                    if todo["id"] in ["save_artifacts", "create_workspace"]:
                        todo["status"] = "completed"
                    elif todo["id"] == "generate_zip":
                        todo["status"] = "in_progress"

            # Create ZIP file for multi-file projects
            if len(structured["files"]) > 1:
                try:
                    # Generate a unique ZIP filename based on task/timestamp
                    import uuid
                    zip_filename = f"project_{uuid.uuid4().hex[:8]}.zip"
                    zip_path = os.path.join(artifacts_dir, zip_filename)

                    if create_zip_file(structured["files"], zip_path):
                        zip_url = f"/api/downloads/{zip_filename}"
                        logger.info(f"Created ZIP file: {zip_filename}")
                    else:
                        logger.error("Failed to create ZIP file")
                except Exception as e:
                    logger.error(f"Failed to create ZIP file: {e}")

            # Update todo list - mark generate_zip and finalize_output as completed
            if "todo_list" in structured:
                for todo in structured["todo_list"]:
                    if todo["id"] in ["generate_zip", "finalize_output"]:
                        todo["status"] = "completed"

        # Return the provider's textual response and structured data if any
        logger.info(f"About to return response_data with structured: {response.structured_response or structured}")
        response_data = {
            "result": result,
            "success": True,
            "tokens": response.tokens,
            "latency_ms": response.latency_ms,
            "structured": response.structured_response or structured,
            "is_creation_task": is_creation_task,
            "artifacts_saved": artifacts_saved,
        }
        # Log response size for debugging truncation issues
        response_json = json.dumps(response_data)
        response_size = len(response_json)
        logger.info(f"Response data size: {response_size} characters")
        logger.info(f"Structured data size: {len(json.dumps(response_data.get('structured', {})))} characters")

        # Check for potentially problematic large responses
        MAX_SAFE_SIZE = 10000  # 10KB limit for safety - lower to trigger compression earlier
        if response_size > MAX_SAFE_SIZE:
            logger.warning(f"Response size {response_size} exceeds safe limit {MAX_SAFE_SIZE}, may be truncated")
            # Try to compress or truncate the structured data if it's too large
            if 'structured' in response_data and response_data['structured']:
                structured_json = json.dumps(response_data['structured'])
                if len(structured_json) > MAX_SAFE_SIZE * 0.8:  # 80% of limit
                    logger.warning("Compressing structured response data")
                    # Keep only essential fields, truncate large content
                    compressed_structured = {}
                    for key, value in response_data['structured'].items():
                        if key == 'files' and isinstance(value, dict):
                            # Truncate large file contents
                            compressed_files = {}
                            for filename, content in value.items():
                                if isinstance(content, str) and len(content) > 10000:  # 10KB per file
                                    compressed_files[filename] = content[:10000] + "\n...[truncated]"
                                else:
                                    compressed_files[filename] = content
                            compressed_structured[key] = compressed_files
                        elif isinstance(value, str) and len(value) > 50000:  # 50KB for other strings
                            compressed_structured[key] = value[:50000] + "\n...[truncated]"
                        else:
                            compressed_structured[key] = value
                    response_data['structured'] = compressed_structured
                    response_data['response_truncated'] = True
                    logger.info("Response data compressed to prevent truncation")

        logger.info(f"Returning response_data: {response_data}")

        # Include ZIP download URL if available
        if zip_url:
            response_data["zip_download_url"] = zip_url

        # Final logging before FastAPI serialization
        final_response_json = json.dumps(response_data)
        final_response_size = len(final_response_json)
        logger.info(f"WORKFLOW: Final response size before FastAPI serialization: {final_response_size} characters")
        logger.info(f"WORKFLOW: Final response keys: {list(response_data.keys())}")
        logger.info(f"WORKFLOW: Final response structured size: {len(json.dumps(response_data.get('structured', {})))} characters")
        logger.info(f"WORKFLOW: Final response preview: {final_response_json[:1000]}...")

        # Check if response might be truncated by common limits
        if final_response_size > 1000000:  # 1MB
            logger.warning(f"WORKFLOW: Response size {final_response_size} exceeds 1MB - may be truncated by HTTP layer")
        if final_response_size > 5000000:  # 5MB
            logger.error(f"WORKFLOW: Response size {final_response_size} exceeds 5MB - very likely to be truncated")

        # DEBUG: Test with small response to isolate HTTP issues
        if "test_small_response" in request.description.lower():
            logger.info("DEBUG: Returning small test response")
            return {"result": "small test response", "success": True, "test": True}

        logger.info("About to execute return statement")
        result = response_data
        logger.info(f"Return value prepared: {type(result)}")
        return result
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_response = {"error": str(e)}
        logger.error(f"Returning error response: {error_response}")
        return error_response

@app.get("/execute/stream")
async def execute_task_stream(
    description: str,
    conversation_history: Optional[str] = None
):
    async def generate():
        try:
            # Parse conversation history if provided
            conv_history = None
            if conversation_history:
                try:
                    conv_history = json.loads(conversation_history)
                except:
                    conv_history = None

            request = ExecuteRequest(
                description=description,
                conversation_history=conv_history
            )

            desc = (request.description or "").strip()

            # Stream initial progress
            yield f"data: {json.dumps({'type': 'progress', 'message': '🤔 Analyzing your request...', 'step': 1, 'total': 8})}\n\n"
            await asyncio.sleep(0.3)

            # Simple canned response still allowed for trivial 'hello'
            if desc.lower() == "hello":
                yield f"data: {json.dumps({'type': 'progress', 'message': '👋 Hello there!', 'step': 8, 'total': 8})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'result': {'result': 'Hello, world!', 'success': True}})}\n\n"
                return

            # Determine if this is a project creation task
            is_creation_task = any(keyword in desc.lower() for keyword in ["create", "make", "generate", "build", "write", "implement"])

            task_type = "Code Generation" if is_creation_task else "General Task"
            yield f"data: {json.dumps({'type': 'progress', 'message': f'📋 Task type: {task_type}', 'step': 2, 'total': 8})}\n\n"
            await asyncio.sleep(0.3)

            # Choose appropriate system prompt based on task type
            if is_creation_task:
                system_prompt = (
                    "You are a skilled software engineer. Generate complete, working code based on the user's request. "
                    "Provide the code with proper structure, comments, and best practices. If creating a project, "
                    "include all necessary files and explain the structure. "
                    "\n\nCRITICAL: For multiple files, you MUST use this exact format:\n"
                    "## filename.ext\n"
                    "```language\n"
                    "file content here\n"
                    "```\n\n"
                    "Example:\n"
                    "## app.py\n"
                    "```python\n"
                    "print('Hello World')\n"
                    "```\n\n"
                    "## requirements.txt\n"
                    "```\n"
                    "flask==2.3.0\n"
                    "```\n\n"
                    "Do NOT use generic headers like '## Main File' or '## Code'. Use actual filenames with extensions. "
                    "For single files, you can use a regular code block without the ## header."
                )
            else:
                system_prompt = SYSTEM_PROMPT

            yield f"data: {json.dumps({'type': 'progress', 'message': '🧠 Initializing Aetherium model...', 'step': 3, 'total': 8})}\n\n"

            # Use NVIDIA NIM exclusively for model calls
            try:
                adapter = NIMAdapter(role="builders")
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Aetherium initialization failed: {str(e)}'})}\n\n"
                return

            # Initialize tool orchestrator for natural language tool parsing
            tool_orchestrator = ToolOrchestrator()

            # Include conversation history if available
            messages = [{"role": "system", "content": system_prompt}]
            if request.conversation_history:
                # Add conversation history as context
                history_text = "Previous conversation:\n"
                for msg in request.conversation_history[-10:]:  # Last 10 messages to avoid token limits
                    role = "User" if msg.get("type") == "user" else "Assistant"
                    content = msg.get("content", "")
                    history_text += f"{role}: {content}\n"
                history_text += f"\nCurrent request: {desc}"
                messages.append({"role": "user", "content": history_text})
            else:
                messages.append({"role": "user", "content": desc})

            # Add tool instructions to system prompt for natural language tool usage
            tool_instructions = """
You have access to tools that you can use by describing them in your response. Available tools:

1. run_tests(repo_url="https://github.com/user/repo", test_command="pytest") - Run tests in a repository
2. git_read_file(repo_url="https://github.com/user/repo", file_path="path/to/file") - Read a file from git
3. git_write_file(repo_url="https://github.com/user/repo", file_path="path/to/file", content="file content", commit_message="message") - Write a file to git
4. list_files(repo_url="https://github.com/user/repo", path="directory") - List files in a git repository
5. shell_exec(command="shell command") - Execute a shell command

To use a tool, simply mention it in your response like: "I need to run: run_tests(repo_url='...', test_command='...')"
"""
            messages[0]["content"] += tool_instructions

            yield f"data: {json.dumps({'type': 'progress', 'message': '🚀 Calling Aetherium model...', 'step': 4, 'total': 8})}\n\n"

            try:
                response = adapter.call_model(messages, temperature=0.7 if is_creation_task else 0.2)

                yield f"data: {json.dumps({'type': 'progress', 'message': f'📊 Aetherium responded ({response.tokens} tokens)', 'step': 5, 'total': 8})}\n\n"

                # Skip tool execution for creation tasks - just generate code directly
                if is_creation_task:
                    yield f"data: {json.dumps({'type': 'progress', 'message': '🔧 Processing code generation...', 'step': 6, 'total': 8})}\n\n"
                else:
                    # Parse and execute natural language tool requests for non-creation tasks
                    tool_results = tool_orchestrator.execute_tool_requests(response.text)
                    if tool_results:
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'🛠️ Executing {len(tool_results)} tools...', 'step': 6, 'total': 8})}\n\n"

                        # Add tool results to messages and call again for follow-up
                        messages.append({"role": "assistant", "content": response.text})
                        tool_results_text = tool_orchestrator.format_tool_results(tool_results)
                        messages.append({"role": "user", "content": f"Tool results:{tool_results_text}\n\nPlease continue with the task based on these results."})

                        # Call model again with tool results
                        response = adapter.call_model(messages, temperature=0.7 if is_creation_task else 0.2)
                        yield f"data: {json.dumps({'type': 'progress', 'message': '🔄 Processing tool results...', 'step': 7, 'total': 8})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Aetherium call failed: {e}'})}" + "\n\n"
                return

            # For creation tasks, attempt to extract and structure code
            result = response.text
            structured = response.structured_response or {}

            if is_creation_task:
                yield f"data: {json.dumps({'type': 'progress', 'message': '📁 Extracting generated files...', 'step': 7, 'total': 8})}\n\n"

                # Extract code from markdown format
                import re
                files = {}

                # First, try to extract files with explicit headers like "## filename.ext"
                file_pattern = r'##\s+([^\n]+)\s*\n```(?:\w+)?\n(.*?)\n```'
                file_matches = re.findall(file_pattern, result, re.DOTALL | re.IGNORECASE)

                if file_matches:
                    for filename, content in file_matches:
                        filename = filename.strip()
                        files[filename] = content.strip()
                        yield f"data: {json.dumps({'type': 'file', 'filename': filename, 'content': content.strip()})}\n\n"

                # If no explicit headers found, try to extract from code blocks and infer filenames
                if not files:
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', result, re.DOTALL)

                    if code_blocks:
                        for i, block in enumerate(code_blocks):
                            content = block.strip()
                            if not content:
                                continue

                            # Try to infer filename from content
                            first_line = content.split('\n')[0].strip()
                            if first_line.startswith('#!/') or 'import ' in first_line or 'from ' in first_line:
                                # Looks like Python
                                filename = f"script_{i+1}.py" if i > 0 else "main.py"
                            elif 'function' in first_line or 'const ' in first_line or 'let ' in first_line:
                                # Looks like JavaScript
                                filename = f"script_{i+1}.js" if i > 0 else "app.js"
                            elif '<!' in content[:50] or '<html' in content[:50].lower():
                                # Looks like HTML
                                filename = f"page_{i+1}.html" if i > 0 else "index.html"
                            elif '{' in content[:10] and ':' in content[:50]:
                                # Looks like JSON
                                filename = f"config_{i+1}.json" if i > 0 else "config.json"
                            else:
                                # Default to Python
                                filename = f"file_{i+1}.py" if i > 0 else "main.py"

                            files[filename] = content
                            yield f"data: {json.dumps({'type': 'file', 'filename': filename, 'content': content})}\n\n"

                # If still no files, treat whole response as a single file
                if not files:
                    files["main.py"] = result.strip()
                    yield f"data: {json.dumps({'type': 'file', 'filename': 'main.py', 'content': result.strip()})}\n\n"

                structured = {"description": result, "files": files}

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Task completed successfully!', 'step': 8, 'total': 8})}\n\n"

            # Send final result
            response_data = {
                "result": result,
                "success": True,
                "tokens": response.tokens,
                "latency_ms": response.latency_ms,
                "structured": response.structured_response or structured,
                "is_creation_task": is_creation_task,
            }

            yield f"data: {json.dumps({'type': 'complete', 'result': response_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

async def detect_file_type_with_ai(content: str, index: int) -> str:
    """Use AI to detect the appropriate file type and extension for code content"""
    try:
        # Create a prompt for the AI to analyze the code
        detection_prompt = f"""
        Analyze this code snippet and determine the most appropriate filename with extension.

        Code content:
        ```
        {content[:1000]}  # Limit to first 1000 chars to avoid token limits
        ```

        Consider:
        1. Programming language (Python, JavaScript, TypeScript, React, etc.)
        2. File type (component, script, config, test, etc.)
        3. Framework/library usage (React, Vue, Angular, etc.)
        4. Content purpose (utility, component, test, config, etc.)

        Respond with ONLY a JSON object in this exact format:
        {{
            "filename": "appropriate_filename.ext",
            "language": "detected_language",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}

        Examples:
        - React component: "Component.tsx"
        - Python script: "script.py"
        - JavaScript utility: "utils.js"
        - TypeScript types: "types.ts"
        - Test file: "Component.test.tsx"
        - Config file: "config.json"
        """

        # Use NIM adapter to detect file type
        adapter = NIMAdapter(role="default")
        messages = [{"role": "user", "content": detection_prompt}]

        response = adapter.call_model(messages, temperature=0.1)  # Low temperature for consistent results

        # Parse the response
        try:
            result = json.loads(response.text.strip())
            detected_filename = result.get("filename", f"file_{index+1}.txt")

            # Ensure we have a valid filename
            if not detected_filename or '.' not in detected_filename:
                detected_filename = f"file_{index+1}.txt"

            logger.info(f"AI detected file type: {detected_filename} (confidence: {result.get('confidence', 0.0)})")
            return detected_filename

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI file detection response: {e}")
            # Fallback to basic detection
            return fallback_file_detection(content, index)

    except Exception as e:
        logger.error(f"AI file type detection failed: {e}")
        # Fallback to basic detection
        return fallback_file_detection(content, index)

def fallback_file_detection(content: str, index: int) -> str:
    """Fallback file detection using basic heuristics"""
    first_lines = content.split('\n')[:5]
    content_sample = '\n'.join(first_lines).lower()

    # Basic detection logic (simplified version of the original)
    if 'import react' in content_sample or 'from \'react\'' in content_sample:
        return f"component_{index+1}.tsx" if index > 0 else "Component.tsx"
    elif any('import ' in line or 'from ' in line for line in first_lines):
        return f"script_{index+1}.py" if index > 0 else "main.py"
    elif any(keyword in content_sample for keyword in ['function', 'const ', 'let ', 'export']):
        return f"script_{index+1}.js" if index > 0 else "app.js"
    elif content.strip().startswith('{') and ':' in content[:100]:
        return f"config_{index+1}.json" if index > 0 else "config.json"
    else:
        return f"file_{index+1}.txt" if index > 0 else "output.txt"


@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return an Aetherium-generated "about" response at three levels: short, medium, detailed.

    Uses Aetherium models to generate detailed responses based on agent-specific prompts.
    Also return the current `SYSTEM_PROMPT` so operators can inspect how the
    agent is being primed.
    """
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = generate_about_response("fix_implementation", level)
    return {
        "level": level,
        "response": resp,
    }