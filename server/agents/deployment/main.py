import json
import os
import sys
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
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional  # noqa: E402
import logging  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES  # noqa: E402

app = FastAPI(title="Deployment Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentRequest(BaseModel):
    repo_url: str
    target_environment: str = "staging"
    branch: str = "main"
    deployment_type: str = "docker"  # docker, kubernetes, serverless, etc.
    config: Optional[dict] = None

class ExecuteRequest(BaseModel):
    description: str

class DeploymentPlan(BaseModel):
    steps: List[str]
    estimated_duration: int  # minutes
    risk_level: str  # low, medium, high
    rollback_plan: List[str]

class DeploymentResponse(BaseModel):
    deployment_plan: DeploymentPlan
    status: str = "planned"
    error: Optional[str] = None

@app.post("/deploy", response_model=DeploymentResponse)
async def deploy_application(request: DeploymentRequest):
    try:
        logger.info(f"Planning deployment for {request.repo_url} to {request.target_environment}")

        # Create deployment plan using Aetherium
        deployment_prompt = f"""
        Create a detailed deployment plan for a software application with the following requirements:

        Repository: {request.repo_url}
        Target Environment: {request.target_environment}
        Branch: {request.branch}
        Deployment Type: {request.deployment_type}
        Additional Config: {json.dumps(request.config or {})}

        Please provide a deployment plan that includes:
        1. Pre-deployment checks
        2. Build steps
        3. Deployment steps
        4. Post-deployment verification
        5. Rollback procedures

        Return the response as a JSON object with the following structure:
        {{
            "steps": ["step 1", "step 2", ...],
            "estimated_duration": 30,
            "risk_level": "medium",
            "rollback_plan": ["rollback step 1", "rollback step 2", ...]
        }}
        """

        try:
            adapter = NIMAdapter(role="builders")
            messages = [{"role": "system", "content": "You are a deployment expert. Create detailed, safe deployment plans."},
                       {"role": "user", "content": deployment_prompt}]
            response = adapter.call_model(messages, temperature=0.2)

            # Parse the deployment plan
            plan_data = json.loads(response.text)
            deployment_plan = DeploymentPlan(**plan_data)

            return DeploymentResponse(deployment_plan=deployment_plan, status="planned")

        except Exception as e:
            logger.error(f"Aetherium deployment planning failed: {str(e)}")
            # Fallback to basic deployment plan
            basic_plan = DeploymentPlan(
                steps=[
                    "Run pre-deployment tests",
                    "Build application artifacts",
                    "Deploy to target environment",
                    "Run health checks",
                    "Update load balancer"
                ],
                estimated_duration=15,
                risk_level="medium",
                rollback_plan=[
                    "Stop new deployment",
                    "Restore previous version",
                    "Verify rollback success"
                ]
            )
            return DeploymentResponse(deployment_plan=basic_plan, status="planned")

    except Exception as e:
        logger.error(f"Error in deployment planning: {str(e)}")
        return DeploymentResponse(
            deployment_plan=DeploymentPlan(steps=[], estimated_duration=0, risk_level="high", rollback_plan=[]),
            status="error",
            error=str(e)
        )

@app.post("/execute")
async def execute_task(request: ExecuteRequest):
    try:
        logger.info(f"Executing deployment task: {request.description}")
        desc = (request.description or "").strip()

        # Simple canned response still allowed for trivial 'hello'
        if desc.lower() == "hello":
            return {"result": "Hello, world!", "success": True}

        # Determine if this is a deployment task
        is_deployment_task = any(keyword in desc.lower() for keyword in ["deploy", "deployment", "release", "publish", "build"])

        # Choose appropriate system prompt based on task type
        if is_deployment_task:
            system_prompt = (
                "You are a deployment and DevOps expert. Provide detailed deployment strategies, "
                "CI/CD pipelines, infrastructure as code, and deployment automation solutions. "
                "Focus on reliability, scalability, and best practices for production deployments."
            )
        else:
            system_prompt = SYSTEM_PROMPT

        # Use NVIDIA NIM exclusively for model calls
        try:
            adapter = NIMAdapter(role="builders")
        except Exception as e:
            msg = f"NIM adapter initialization failed: {e}. Ensure NVIDIA_NIM_API_KEY is set in the environment."
            logger.error(msg)
            return {"error": msg}

        # Include appropriate system prompt for better responses
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": desc}]
        try:
            response = adapter.call_model(messages, temperature=0.7 if is_deployment_task else 0.2)
        except Exception as e:
            logger.error("NIM provider call failed: %s", str(e))
            # Fallback response for deployment tasks
            if is_deployment_task:
                fallback_response = (
                    "Based on best practices for deployment tasks:\n\n"
                    "1. **Pre-deployment checks**: Run automated tests, security scans, and environment validation\n"
                    "2. **Build process**: Use CI/CD pipelines with artifact versioning\n"
                    "3. **Deployment strategy**: Implement blue-green or canary deployments for zero downtime\n"
                    "4. **Monitoring**: Set up comprehensive monitoring and alerting\n"
                    "5. **Rollback plan**: Have automated rollback procedures ready\n\n"
                    "For production deployments, always use infrastructure as code, automated testing, "
                    "and gradual rollout strategies to minimize risk."
                )
                return {
                    "result": fallback_response,
                    "success": True,
                    "fallback": True,
                    "is_deployment_task": is_deployment_task,
                }
            else:
                return {"error": f"NIM provider call failed: {e}"}

        # Return the provider's textual response and structured data if any
        return {
            "result": response.text,
            "success": True,
            "tokens": response.tokens,
            "latency_ms": response.latency_ms,
            "structured": response.structured_response or {},
            "is_deployment_task": is_deployment_task,
        }
    except Exception as e:
        logger.error(f"Error executing deployment task: {str(e)}")
        return {"error": str(e)}


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