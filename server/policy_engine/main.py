import os
import json
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
from typing import Dict, Any, List
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(title="Policy Engine Service", version="1.0.0")

OPA_URL = "http://localhost:8181"

class PolicyEvaluationRequest(BaseModel):
    policy_name: str
    input_data: Dict[str, Any]

class PolicyEvaluationResponse(BaseModel):
    allow: bool
    reason: str = ""

class Policy(BaseModel):
    name: str
    rego_content: str

@app.on_event("startup")
async def startup_event():
    logger.info("Starting policy engine startup event")
    # Load policies into OPA
    policies_dir = "policies"
    if os.path.exists(policies_dir):
        logger.info("Loading policies from directory", policies_dir=policies_dir)
        for file in os.listdir(policies_dir):
            if file.endswith(".rego"):
                policy_name = file[:-5]  # remove .rego
                with open(os.path.join(policies_dir, file), "r") as f:
                    rego_content = f.read()
                try:
                    response = requests.put(f"{OPA_URL}/v1/policies/{policy_name}", data=rego_content)
                    response.raise_for_status()
                    logger.info("Loaded policy", policy_name=policy_name)
                except requests.RequestException as e:
                    logger.error("Failed to load policy", policy_name=policy_name, error=str(e))
        logger.info("All policies loaded")
    else:
        logger.warning("Policies directory not found", policies_dir=policies_dir)
    logger.info("Policy engine startup event completed")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/about")
async def about():
    return {"service": "policy_engine", "version": "1.0.0", "description": "Policy evaluation and management service using OPA"}

@app.post("/evaluate", response_model=PolicyEvaluationResponse)
async def evaluate_policy(request: PolicyEvaluationRequest, background_tasks: BackgroundTasks):
    try:
        response = requests.post(f"{OPA_URL}/v1/data/policy/{request.policy_name}", json={"input": request.input_data})
        response.raise_for_status()
        result = response.json().get("result", {})
        allow = result.get("allow", True)
        reason = "Policy allows" if allow else "Policy denies"
        # Audit log
        background_tasks.add_task(log_decision, request.policy_name, request.input_data, allow, reason)
        return PolicyEvaluationResponse(allow=allow, reason=reason)
    except requests.RequestException as e:
        logger.error("Policy evaluation failed", policy_name=request.policy_name, error=str(e))
        raise HTTPException(status_code=500, detail="Policy evaluation failed")

@app.get("/policies", response_model=List[str])
async def list_policies():
    try:
        response = requests.get(f"{OPA_URL}/v1/policies")
        response.raise_for_status()
        policies = [p["id"] for p in response.json().get("result", [])]
        return policies
    except requests.RequestException as e:
        logger.error("Failed to list policies", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list policies")

@app.post("/policies")
async def add_policy(policy: Policy):
    try:
        response = requests.put(f"{OPA_URL}/v1/policies/{policy.name}", data=policy.rego_content)
        response.raise_for_status()
        logger.info("Added policy", policy_name=policy.name)
        return {"message": "Policy added successfully"}
    except requests.RequestException as e:
        logger.error("Failed to add policy", policy_name=policy.name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add policy")

@app.put("/policies/{policy_name}")
async def update_policy(policy_name: str, policy: Policy):
    if policy.name != policy_name:
        raise HTTPException(status_code=400, detail="Policy name mismatch")
    try:
        response = requests.put(f"{OPA_URL}/v1/policies/{policy_name}", data=policy.rego_content)
        response.raise_for_status()
        logger.info("Updated policy", policy_name=policy_name)
        return {"message": "Policy updated successfully"}
    except requests.RequestException as e:
        logger.error("Failed to update policy", policy_name=policy_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update policy")

def log_decision(policy_name: str, input_data: Dict[str, Any], allow: bool, reason: str):
    logger.info("Policy decision", policy_name=policy_name, input=input_data, allow=allow, reason=reason)