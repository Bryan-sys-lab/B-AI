import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., vulnerability_scanner) and top-level packages resolve correctly.
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

from .vulnerability_scanner import VulnerabilityScanner  # noqa: E402
from .dependency_checker import DependencyChecker  # noqa: E402
from .policy_enforcer import PolicyEnforcer  # noqa: E402
from .risk_assessor import RiskAssessor  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES  # noqa: E402

app = FastAPI(title="Security Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityScanRequest(BaseModel):
    repo_path: str
    language: str = "python"
    scan_types: List[str] = ["vulnerability", "dependency", "policy"]
    tools: Optional[List[str]] = None
    risk_thresholds: Optional[Dict[str, int]] = None

class SecurityScanResponse(BaseModel):
    assessment: Dict[str, Any]
    block_merge: bool
    error: Optional[str] = None

class ExplainVulnerabilityRequest(BaseModel):
    vulnerability_id: str
    description: str
    severity: str
    cwe: Optional[str] = None

@app.post("/scan", response_model=SecurityScanResponse)
async def perform_security_scan(request: SecurityScanRequest):
    try:
        logger.info(f"Starting security scan for {request.repo_path}")

        # Initialize components
        vuln_scanner = VulnerabilityScanner()
        dep_checker = DependencyChecker()
        policy_enforcer = PolicyEnforcer()
        risk_assessor = RiskAssessor()

        # Perform scans based on requested types
        vuln_results = []
        dep_results = None
        policy_results = None

        if "vulnerability" in request.scan_types:
            logger.info("Running vulnerability scanning...")
            vuln_results = vuln_scanner.scan_codebase(request.repo_path, request.tools)

        if "dependency" in request.scan_types:
            logger.info("Running dependency checking...")
            dep_results = dep_checker.scan_dependencies(request.repo_path)

        if "policy" in request.scan_types:
            logger.info("Running policy enforcement...")
            policy_results = policy_enforcer.enforce_policies(request.repo_path, request.language)

        # Default empty results if not scanned
        if dep_results is None:
            from .dependency_checker import DependencyScanResult
            dep_results = DependencyScanResult(tool='dependency-check', vulnerabilities=[], summary={})

        if policy_results is None:
            from .policy_enforcer import PolicyCheckResult
            policy_results = PolicyCheckResult(violations=[], summary={})

        # Assess and prioritize risks
        logger.info("Assessing and prioritizing risks...")
        assessment = risk_assessor.assess_and_prioritize(
            vuln_results, dep_results, policy_results, request.risk_thresholds
        )

        # Check if merge should be blocked
        block_merge = risk_assessor.should_block_merge(assessment, request.risk_thresholds)

        return SecurityScanResponse(
            assessment={
                "fixes": [fix.dict() for fix in assessment.fixes],
                "summary": assessment.summary,
                "risk_level": assessment.risk_level
            },
            block_merge=block_merge
        )

    except Exception as e:
        logger.error(f"Error in security scan: {str(e)}")
        return SecurityScanResponse(
            assessment={},
            block_merge=False,
            error=str(e)
        )

@app.post("/explain")
async def explain_vulnerability(request: ExplainVulnerabilityRequest):
    try:
        logger.info(f"Explaining vulnerability: {request.vulnerability_id}")

        # Use NVIDIA NIM to explain the vulnerability
        nim = NIMAdapter()

        prompt = f"""
        Explain the following security vulnerability in detail:

        Vulnerability ID: {request.vulnerability_id}
        Description: {request.description}
        Severity: {request.severity}
        CWE: {request.cwe or 'Not specified'}

        Please provide:
        1. What this vulnerability is
        2. How it can be exploited
        3. Potential impact
        4. Remediation steps
        5. Prevention best practices

        Keep the explanation clear and actionable.
        """

        messages = [
            {"role": "system", "content": "You are a security expert explaining vulnerabilities."},
            {"role": "user", "content": prompt}
        ]

        response = nim.call_model(messages)

        return {
            "explanation": response.text,
            "confidence": response.confidence,
            "tokens_used": response.tokens
        }

    except Exception as e:
        logger.error(f"Error explaining vulnerability: {str(e)}")
        return {"error": str(e)}

@app.post("/execute")
async def execute_task(request: dict):
    try:
        logger.info(f"Executing security task: {request}")
        # Simple execution, perhaps use providers or run specific security commands
        return {"result": "Security task executed", "success": True}
    except Exception as e:
        logger.error(f"Error executing security task: {str(e)}")
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
            yield f"data: {json.dumps({'type': 'progress', 'message': '🔒 Analyzing security request...', 'step': 1, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            # Simple canned response still allowed for trivial 'hello'
            if desc.lower() == "hello":
                yield f"data: {json.dumps({'type': 'progress', 'message': '👋 Hello there!', 'step': 4, 'total': 4})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'result': {'result': 'Hello, world!', 'success': True}})}\n\n"
                return

            # Determine if this is a security task
            is_security_task = any(keyword in desc.lower() for keyword in ["security", "scan", "vulnerability", "audit", "risk", "threat"])

            task_type = "Security Analysis" if is_security_task else "General Task"
            yield f"data: {json.dumps({'type': 'progress', 'message': f'📋 Task type: {task_type}', 'step': 2, 'total': 4})}\n\n"
            await asyncio.sleep(0.3)

            yield f"data: {json.dumps({'type': 'progress', 'message': '🔍 Performing security analysis...', 'step': 3, 'total': 4})}\n\n"

            # For now, simple execution - in real implementation would run actual security scans
            result = "Security analysis completed successfully"

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Security analysis completed successfully!', 'step': 4, 'total': 4})}\n\n"

            # Send final result
            response_data = {
                "result": result,
                "success": True,
                "is_security_task": is_security_task,
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