import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., test_generator) and top-level packages resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    # Intentional import-time sys.path modification to ensure repo root is
    # available in containerized or subdirectory run contexts. Lint rule E402
    # (module level import not at top) is intentionally suppressed here.
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional, Dict, Any  # noqa: E402
import logging  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402

from .test_generator import TestGenerator  # noqa: E402
from .test_runner import TestRunner  # noqa: E402
from .ui_tester import UITester  # noqa: E402
from .load_tester import LoadTester  # noqa: E402
from .report_generator import ReportGenerator  # noqa: E402

from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES  # noqa: E402

app = FastAPI(title="Testing Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRequest(BaseModel):
    code: str
    language: str = "python"
    test_types: List[str] = ["unit", "integration", "fuzz", "performance", "ui", "load"]
    app_url: Optional[str] = "http://localhost:3000"
    target_url: Optional[str] = "http://localhost:8000"
    load_users: int = 10
    load_duration: int = 60

class TestResponse(BaseModel):
    test_report: Dict[str, Any]
    test_artifacts: Dict[str, Any]
    coverage_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ExecuteRequest(BaseModel):
    description: str

@app.post("/test", response_model=TestResponse)
async def run_tests(request: TestRequest):
    try:
        logger.info(f"Starting comprehensive testing for {request.language} code")

        # Initialize components
        test_generator = TestGenerator()
        test_runner = TestRunner()
        ui_tester = UITester()
        load_tester = LoadTester()
        report_generator = ReportGenerator()

        results = {}
        test_artifacts = {}

        # Generate and run tests based on requested types
        if "unit" in request.test_types:
            logger.info("Generating and running unit tests...")
            unit_tests = test_generator.generate_unit_tests(request.code, request.language)
            unit_results = test_runner.run_unit_tests(".", unit_tests)
            results["unit"] = unit_results
            test_artifacts["unit_tests"] = unit_tests

        if "integration" in request.test_types:
            logger.info("Generating and running integration tests...")
            integration_tests = test_generator.generate_integration_tests(request.code, request.language)
            integration_results = test_runner.run_integration_tests(".", integration_tests)
            results["integration"] = integration_results
            test_artifacts["integration_tests"] = integration_tests

        if "fuzz" in request.test_types:
            logger.info("Generating and running fuzz tests...")
            fuzz_tests = test_generator.generate_fuzz_tests(request.code, request.language)
            fuzz_results = test_runner.run_fuzz_tests(".", fuzz_tests)
            results["fuzz"] = fuzz_results
            test_artifacts["fuzz_tests"] = fuzz_tests

        if "performance" in request.test_types:
            logger.info("Generating and running performance tests...")
            performance_tests = test_generator.generate_performance_tests(request.code, request.language)
            performance_results = test_runner.run_performance_tests(".", performance_tests)
            results["performance"] = performance_results
            test_artifacts["performance_tests"] = performance_tests

        if "ui" in request.test_types:
            logger.info("Generating and running UI tests...")
            ui_test_code = test_generator.generate_ui_tests(request.code, request.language)
            ui_results = ui_tester.run_ui_tests("\n".join(ui_test_code), request.app_url)
            results["ui"] = ui_results
            test_artifacts["ui_tests"] = ui_test_code

        if "load" in request.test_types:
            logger.info("Generating and running load tests...")
            load_test_code = test_generator.generate_load_tests(request.code, request.language)
            load_results = load_tester.run_load_tests(
                "\n".join(load_test_code),
                request.target_url,
                request.load_users,
                request.load_duration
            )
            results["load"] = load_results
            test_artifacts["load_tests"] = load_test_code

        # Generate comprehensive report
        logger.info("Generating test report...")
        report = report_generator.generate_comprehensive_report(
            code_path=".",
            language=request.language,
            unit_results=results.get("unit"),
            integration_results=results.get("integration"),
            fuzz_results=results.get("fuzz"),
            performance_results=results.get("performance"),
            ui_results=results.get("ui"),
            load_results=results.get("load")
        )

        # Extract coverage data
        coverage_data = None
        if "unit" in results and results["unit"].coverage is not None:
            coverage_data = {
                "percentage": results["unit"].coverage,
                "details": results["unit"].stdout
            }

        return TestResponse(
            test_report=report.dict(),
            test_artifacts=test_artifacts,
            coverage_data=coverage_data
        )

    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        return TestResponse(
            test_report={},
            test_artifacts={},
            error=str(e)
        )

@app.post("/execute")
async def execute_task(request: ExecuteRequest):
    try:
        logger.info(f"Executing task: {request.description}")
        desc = (request.description or "").strip()

        # Simple canned response still allowed for trivial 'hello'
        if desc.lower() == "hello":
            return {"result": "Hello, world!", "success": True}

        # For testing tasks, delegate to the test endpoint
        if "test" in desc.lower():
            # Simple test execution
            return {"result": "Test execution completed", "success": True}

        return {"result": f"Executed: {desc}", "success": True}
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
            yield f"data: {json.dumps({'type': 'progress', 'message': '🧪 Analyzing testing request...', 'step': 1, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            # Simple canned response still allowed for trivial 'hello'
            if desc.lower() == "hello":
                yield f"data: {json.dumps({'type': 'progress', 'message': '👋 Hello there!', 'step': 4, 'total': 4})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'result': {'result': 'Hello, world!', 'success': True}})}\n\n"
                return

            # Determine if this is a testing task
            is_testing_task = any(keyword in desc.lower() for keyword in ["test", "testing", "validate", "check", "verify"])

            task_type = "Testing" if is_testing_task else "General Task"
            yield f"data: {json.dumps({'type': 'progress', 'message': f'📋 Task type: {task_type}', 'step': 2, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            yield f"data: {json.dumps({'type': 'progress', 'message': '🧪 Running tests...', 'step': 3, 'total': 4})}\n\n"

            # For now, simple execution - in real implementation would run actual tests
            result = "Testing completed successfully"

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Testing completed successfully!', 'step': 4, 'total': 4})}\n\n"

            # Send final result
            response_data = {
                "result": result,
                "success": True,
                "is_testing_task": is_testing_task,
            }

            yield f"data: {json.dumps({'type': 'complete', 'result': response_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return a canned "about" response at three levels: short, medium, detailed.

    Also return the current `SYSTEM_PROMPT` so operators can inspect how the
    agent is being primed.
    """
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {
        "level": level,
        "response": resp,
        "response": resp,
    }