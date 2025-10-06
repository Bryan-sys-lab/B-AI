import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., test_runner) and top-level packages resolve correctly.
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
import json  # noqa: E402
from .test_runner import TestRunner  # noqa: E402
from .failure_analyzer import FailureAnalyzer  # noqa: E402
from .report_generator import ReportGenerator, DebugReport  # noqa: E402
from .patch_suggester import PatchSuggester, PatchRequest  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES, get_agent_prompt  # noqa: E402

app = FastAPI(title="Debugger Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    test_command: str = "pytest"

class DebugResponse(BaseModel):
    debug_report: dict
    patch_requests: List[dict]
    error: Optional[str] = None

@app.post("/debug", response_model=DebugResponse)
async def debug(request: DebugRequest):
    try:
        logger.info(f"Starting debug for {request.repo_url}")

        # Run tests
        test_runner = TestRunner()
        test_result = test_runner.run_tests(request.repo_url, request.branch, request.test_command)

        if test_result.success:
            return DebugResponse(
                debug_report={"message": "All tests passed, no debugging needed"},
                patch_requests=[]
            )

        # Analyze failures
        failure_analyzer = FailureAnalyzer()
        analysis = failure_analyzer.analyze_failures(test_result.output, test_result.errors)

        # Generate report
        report_generator = ReportGenerator()
        debug_report = report_generator.generate_report(analysis)

        # Suggest patches
        patch_suggester = PatchSuggester()
        # For code_context, we might need to fetch code, but for now use empty
        code_context = ""  # TODO: fetch relevant code
        patch_requests = patch_suggester.suggest_patches(analysis, code_context)

        return DebugResponse(
            debug_report={
                "root_cause": debug_report.root_cause,
                "analysis": debug_report.analysis,
                "recommendations": debug_report.recommendations
            },
            patch_requests=[
                {
                    "diff": p.diff,
                    "description": p.description,
                    "confidence": p.confidence
                } for p in patch_requests
            ]
        )

    except Exception as e:
        logger.error(f"Error in debug: {str(e)}")
        return DebugResponse(
            debug_report={},
            patch_requests=[],
            error=str(e)
        )

@app.post("/execute")
async def execute_task(request: dict):
    try:
        logger.info(f"Executing task: {request}")
        desc = (request.get("description", "") or "").strip()

        # Simple canned response still allowed for trivial 'hello'
        if desc.lower() == "hello":
            logger.info("Agent special case triggered for 'hello'")
            return {"result": "Hello, world!", "success": True}

        # Determine if this is a debugging task
        is_debug_task = any(keyword in desc.lower() for keyword in ["debug", "fix", "error", "bug", "issue", "problem"])

        # Choose appropriate system prompt based on task type
        if is_debug_task:
            system_prompt = get_agent_prompt("debugger")
        else:
            system_prompt = SYSTEM_PROMPT

        # Use NVIDIA NIM for debugging tasks
        try:
            adapter = NIMAdapter(role="debugger")
            logger.info("NIM adapter initialized successfully")
        except Exception as e:
            msg = f"NIM adapter initialization failed: {e}. Ensure NVIDIA_NIM_API_KEY is set in the environment."
            logger.error(msg)
            return {"error": msg}

        # Include conversation history if available
        messages = [{"role": "system", "content": system_prompt}]
        if request.get("conversation_history"):
            # Add conversation history as context
            history_text = "Previous conversation:\n"
            for msg in request["conversation_history"][-10:]:  # Last 10 messages
                role = "User" if msg.get("type") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += f"\nCurrent request: {desc}"
            messages.append({"role": "user", "content": history_text})
        else:
            messages.append({"role": "user", "content": desc})

        logger.info(f"Calling NIM with {len(messages)} messages")
        response = adapter.call_model(messages, temperature=0.3 if is_debug_task else 0.2)
        logger.info(f"NIM response received: tokens={response.tokens}, latency={response.latency_ms}ms")

        result = response.text

        return {
            "result": result,
            "success": True,
            "tokens": response.tokens,
            "latency_ms": response.latency_ms,
            "is_debug_task": is_debug_task,
        }
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        return {"error": str(e)}

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

            request = {"description": description, "conversation_history": conv_history}

            desc = (request.get("description", "") or "").strip()

            # Stream initial progress
            yield f"data: {json.dumps({'type': 'progress', 'message': '🔍 Analyzing debugging request...', 'step': 1, 'total': 6})}\n\n"
            await asyncio.sleep(0.3)

            # Simple canned response still allowed for trivial 'hello'
            if desc.lower() == "hello":
                yield f"data: {json.dumps({'type': 'progress', 'message': '👋 Hello there!', 'step': 6, 'total': 6})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'result': {'result': 'Hello, world!', 'success': True}})}\n\n"
                return

            # Determine if this is a debugging task
            is_debug_task = any(keyword in desc.lower() for keyword in ["debug", "fix", "error", "bug", "issue", "problem"])

            task_type = "Debugging" if is_debug_task else "General Task"
            yield f"data: {json.dumps({'type': 'progress', 'message': f'📋 Task type: {task_type}', 'step': 2, 'total': 6})}\n\n"
            await asyncio.sleep(0.3)

            # Choose appropriate system prompt based on task type
            if is_debug_task:
                system_prompt = get_agent_prompt("debugger")
            else:
                system_prompt = SYSTEM_PROMPT

            yield f"data: {json.dumps({'type': 'progress', 'message': '🧠 Initializing Aetherium model...', 'step': 3, 'total': 6})}\n\n"

            # Use NVIDIA NIM for debugging tasks
            try:
                adapter = NIMAdapter(role="debugger")
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Aetherium initialization failed: {str(e)}'})}\n\n"
                return

            # Include conversation history if available
            messages = [{"role": "system", "content": system_prompt}]
            if request.get("conversation_history"):
                # Add conversation history as context
                history_text = "Previous conversation:\n"
                for msg in request["conversation_history"][-10:]:  # Last 10 messages
                    role = "User" if msg.get("type") == "user" else "Assistant"
                    content = msg.get("content", "")
                    history_text += f"{role}: {content}\n"
                history_text += f"\nCurrent request: {desc}"
                messages.append({"role": "user", "content": history_text})
            else:
                messages.append({"role": "user", "content": desc})

            yield f"data: {json.dumps({'type': 'progress', 'message': '🚀 Calling Aetherium model...', 'step': 4, 'total': 6})}\n\n"

            try:
                response = adapter.call_model(messages, temperature=0.3 if is_debug_task else 0.2)

                yield f"data: {json.dumps({'type': 'progress', 'message': f'📊 Aetherium responded ({response.tokens} tokens)', 'step': 5, 'total': 6})}\n\n"

                result = response.text

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Aetherium call failed: {e}'})}" + "\n\n"
                return

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Task completed successfully!', 'step': 6, 'total': 6})}\n\n"

            # Send final result
            response_data = {
                "result": result,
                "success": True,
                "tokens": response.tokens,
                "latency_ms": response.latency_ms,
                "is_debug_task": is_debug_task,
            }

            yield f"data: {json.dumps({'type': 'complete', 'result': response_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/about")
def about(detail: Optional[str] = "short"):
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {
        "level": level,
        "response": resp,
        "response": resp,
    }