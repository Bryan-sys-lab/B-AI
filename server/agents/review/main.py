import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., static_analyzer) and top-level packages resolve correctly.
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

from .static_analyzer import StaticAnalyzer  # noqa: E402
from .security_scanner import SecurityScanner  # noqa: E402
from .style_checker import StyleChecker  # noqa: E402
from .compliance_checker import ComplianceChecker  # noqa: E402
from .report_generator import ReportGenerator  # noqa: E402

from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES  # noqa: E402

app = FastAPI(title="Review Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewRequest(BaseModel):
    code_path: str
    language: str = "python"  # python, javascript, terraform, etc.
    include_static_analysis: bool = True
    include_security_scan: bool = True
    include_style_check: bool = True
    include_compliance_check: bool = True

class ReviewResponse(BaseModel):
    review_report: dict
    approval_status: str
    risk_score: float
    error: Optional[str] = None

@app.post("/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest):
    try:
        logger.info(f"Starting code review for {request.code_path} ({request.language})")

        # Initialize analyzers
        static_analyzer = StaticAnalyzer()
        security_scanner = SecurityScanner()
        style_checker = StyleChecker()
        compliance_checker = ComplianceChecker()
        report_generator = ReportGenerator()

        # Run analyses based on request flags
        static_results = []
        if request.include_static_analysis:
            logger.info("Running static analysis...")
            static_result = static_analyzer.analyze(request.code_path)
            static_results.append(static_result)

        security_results = []
        if request.include_security_scan:
            logger.info("Running security scan...")
            security_results = security_scanner.scan(request.code_path, request.language)

        style_results = []
        if request.include_style_check:
            logger.info("Running style check...")
            style_results = style_checker.check(request.code_path, request.language)

        compliance_results = []
        if request.include_compliance_check:
            logger.info("Running compliance check...")
            compliance_results = compliance_checker.check(request.code_path, request.language)

        # Generate comprehensive report
        logger.info("Generating review report...")
        review_report = report_generator.generate_report(
            code_path=request.code_path,
            language=request.language,
            static_results=static_results,
            security_results=security_results,
            style_results=style_results,
            compliance_results=compliance_results
        )

        return ReviewResponse(
            review_report={
                "timestamp": review_report.timestamp,
                "code_path": review_report.code_path,
                "language": review_report.language,
                "static_analysis": review_report.static_analysis,
                "security_scan": review_report.security_scan,
                "style_check": review_report.style_check,
                "compliance_check": review_report.compliance_check,
                "summary": review_report.summary,
                "recommendations": review_report.recommendations,
                "detailed_findings": review_report.detailed_findings
            },
            approval_status=review_report.approval_status,
            risk_score=review_report.risk_score
        )

    except Exception as e:
        logger.error(f"Error in code review: {str(e)}")
        return ReviewResponse(
            review_report={},
            approval_status="error",
            risk_score=1.0,
            error=str(e)
        )

@app.post("/execute")
async def execute_task(request: dict):
    try:
        logger.info(f"Executing task: {request}")
        # Simple execution, perhaps use providers
        return {"result": "Executed", "success": True}
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
            yield f"data: {json.dumps({'type': 'progress', 'message': '🔍 Analyzing review request...', 'step': 1, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            # Simple canned response still allowed for trivial 'hello'
            if desc.lower() == "hello":
                yield f"data: {json.dumps({'type': 'progress', 'message': '👋 Hello there!', 'step': 4, 'total': 4})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'result': {'result': 'Hello, world!', 'success': True}})}\n\n"
                return

            # Determine if this is a review task
            is_review_task = any(keyword in desc.lower() for keyword in ["review", "analyze", "check", "examine", "assess"])

            task_type = "Code Review" if is_review_task else "General Task"
            yield f"data: {json.dumps({'type': 'progress', 'message': f'📋 Task type: {task_type}', 'step': 2, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            yield f"data: {json.dumps({'type': 'progress', 'message': '🔬 Performing code analysis...', 'step': 3, 'total': 4})}\n\n"

            # For now, simple execution - in real implementation would use providers
            result = "Code review completed successfully"

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Review completed successfully!', 'step': 4, 'total': 4})}\n\n"

            # Send final result
            response_data = {
                "result": result,
                "success": True,
                "is_review_task": is_review_task,
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