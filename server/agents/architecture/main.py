import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible
load_dotenv()

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional, Dict, Any  # noqa: E402
import logging  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import get_agent_prompt  # noqa: E402
from common.utils import is_running_in_container  # noqa: E402
from common.endpoints import add_health_endpoint  # noqa: E402

app = FastAPI(title="Architecture Agent")

add_health_endpoint(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchitectureRequest(BaseModel):
    description: str
    requirements: Optional[List[str]] = None
    technology_stack: Optional[List[str]] = None
    scale_requirements: Optional[str] = None
    conversation_history: Optional[List[dict]] = None
    include_diagrams: bool = True
    include_requirements_doc: bool = True
    include_tech_specs: bool = True
    include_deployment_plan: bool = True

class TemplateRequest(BaseModel):
    project_type: str
    features: List[str]
    technology_stack: List[str]

class ArchitectureResponse(BaseModel):
    design: Dict[str, Any]
    templates: Dict[str, str]
    recommendations: List[str]
    error: Optional[str] = None

class TemplateResponse(BaseModel):
    files: Dict[str, str]
    structure: Dict[str, Any]
    error: Optional[str] = None

class ExecuteRequest(BaseModel):
    description: str

class ExecuteResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None

@app.post("/execute", response_model=ExecuteResponse)
async def execute_task(request: ExecuteRequest):
    """Execute an architecture task - matches the interface expected by orchestrator"""
    try:
        logger.info(f"Executing architecture task: {request.description}")

        # Create an architecture request from the description
        arch_request = ArchitectureRequest(
            description=request.description,
            include_diagrams=False,
            include_requirements_doc=False,
            include_tech_specs=True,
            include_deployment_plan=False
        )

        # Call the design function
        response = await design_architecture_sync(arch_request)

        if response.error:
            return ExecuteResponse(success=False, error=response.error)

        # Format the response as expected by orchestrator
        result_text = f"Architecture Design Complete:\n\n"

        if response.design:
            result_text += f"Design: {json.dumps(response.design, indent=2)}\n\n"

        if response.templates:
            result_text += f"Templates: {json.dumps(response.templates, indent=2)}\n\n"

        if response.recommendations:
            result_text += f"Recommendations:\n" + "\n".join(f"- {rec}" for rec in response.recommendations)

        return ExecuteResponse(success=True, result=result_text)

    except Exception as e:
        logger.error(f"Error executing architecture task: {str(e)}")
        return ExecuteResponse(success=False, error=str(e))

@app.post("/design_architecture", response_model=ArchitectureResponse)
async def design_architecture(request: ArchitectureRequest):
    return await design_architecture_sync(request)

@app.get("/design_architecture/stream")
async def design_architecture_stream(
    description: str,
    requirements: Optional[str] = None,
    technology_stack: Optional[str] = None,
    scale_requirements: Optional[str] = None
):
    async def generate():
        try:
            # Parse requirements and tech stack
            req_list = requirements.split(',') if requirements else []
            tech_list = technology_stack.split(',') if technology_stack else []

            request = ArchitectureRequest(
                description=description,
                requirements=req_list,
                technology_stack=tech_list,
                scale_requirements=scale_requirements
            )

            # Stream progress updates
            yield f"data: {json.dumps({'type': 'progress', 'message': '🤔 Analyzing requirements...', 'step': 1, 'total': 10})}\n\n"
            await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'progress', 'message': '📋 Gathering system context...', 'step': 2, 'total': 10})}\n\n"
            await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'progress', 'message': '🏗️ Designing system architecture...', 'step': 3, 'total': 10})}\n\n"

            # Use NIM adapter for architecture design
            adapter = NIMAdapter(role="architect")

            system_prompt = get_agent_prompt("architecture")

            messages = [{"role": "system", "content": system_prompt}]

            # Build user prompt
            user_prompt = f"""
            Design a software architecture for the following requirements:

            Description: {request.description}

            Requirements:
            {chr(10).join(f"- {req}" for req in (request.requirements or []))}

            Technology Stack:
            {chr(10).join(f"- {tech}" for tech in (request.technology_stack or []))}

            Scale Requirements: {request.scale_requirements or 'Not specified'}

            Provide a comprehensive architectural design.
            """

            messages.append({"role": "user", "content": user_prompt})

            yield f"data: {json.dumps({'type': 'progress', 'message': '🧠 Aetherium is thinking through the architecture...', 'step': 4, 'total': 10})}\n\n"

            response = adapter.call_model(messages, temperature=0.3)

            yield f"data: {json.dumps({'type': 'progress', 'message': '📊 Processing architectural specifications...', 'step': 5, 'total': 10})}\n\n"

            # Parse JSON response
            try:
                result = json.loads(response.text)

                # Stream each section
                if 'design' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '🏗️ System Architecture', 'content': result['design']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'diagrams' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '📈 Architecture Diagrams', 'content': result['diagrams']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'requirements' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '📋 Requirements Analysis', 'content': result['requirements']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'dependencies' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '📦 Technology Stack', 'content': result['dependencies']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'frameworks' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '🛠️ Frameworks & Tools', 'content': result['frameworks']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'deployment' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '🚀 Deployment Strategy', 'content': result['deployment']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'security' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '🔒 Security Architecture', 'content': result['security']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'testing' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '🧪 Testing Strategy', 'content': result['testing']})}\n\n"
                    await asyncio.sleep(0.3)

                if 'recommendations' in result:
                    yield f"data: {json.dumps({'type': 'section', 'title': '💡 Implementation Recommendations', 'content': result['recommendations']})}\n\n"
                    await asyncio.sleep(0.3)

                yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Architecture design complete!', 'step': 10, 'total': 10})}\n\n"

                # Send final result
                yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

            except json.JSONDecodeError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to parse response: {str(e)}'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

async def design_architecture_sync(request: ArchitectureRequest):
    try:
        logger.info(f"Designing architecture for: {request.description}")

        # Use NIM adapter for architecture design
        adapter = NIMAdapter(role="architect")

        system_prompt = get_agent_prompt("architecture")

        messages = [{"role": "system", "content": system_prompt}]

        # Build user prompt
        user_prompt = f"""
        Design a software architecture for the following requirements:

        Description: {request.description}

        Requirements:
        {chr(10).join(f"- {req}" for req in (request.requirements or []))}

        Technology Stack:
        {chr(10).join(f"- {tech}" for tech in (request.technology_stack or []))}

        Scale Requirements: {request.scale_requirements or 'Not specified'}

        Provide a comprehensive architectural design including system components,
        data flows, technology choices, and implementation recommendations.
        """

        if request.conversation_history:
            # Add conversation history
            history_text = "Previous conversation:\n"
            for msg in request.conversation_history[-10:]:
                role = "User" if msg.get("type") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            user_prompt = f"{history_text}\n\n{user_prompt}"

        messages.append({"role": "user", "content": user_prompt})

        response = adapter.call_model(messages, temperature=0.3)

        # Parse JSON response
        try:
            result = json.loads(response.text)
            return ArchitectureResponse(**result)
        except json.JSONDecodeError:
            # Fallback if not JSON
            return ArchitectureResponse(
                design={"description": response.text},
                templates={},
                recommendations=["Review the architectural design provided"]
            )

    except Exception as e:
        logger.error(f"Error in architecture design: {str(e)}")
        return ArchitectureResponse(
            design={},
            templates={},
            recommendations=[],
            error=str(e)
        )

@app.post("/generate_templates", response_model=TemplateResponse)
async def generate_templates(request: TemplateRequest):
    try:
        logger.info(f"Generating templates for: {request.project_type}")

        # Use NIM adapter for template generation
        adapter = NIMAdapter(role="builders")

        system_prompt = """
        You are a code generation expert. Generate complete project templates with proper structure.
        Include all necessary files, configuration, and boilerplate code.
        Output in JSON format with keys: files (dict of filename->content), structure (dict describing the project structure)
        """

        messages = [{"role": "system", "content": system_prompt}]

        user_prompt = f"""
        Generate a complete project template for:

        Project Type: {request.project_type}
        Features: {', '.join(request.features)}
        Technology Stack: {', '.join(request.technology_stack)}

        Include:
        - Project structure
        - Configuration files
        - Main application files
        - Tests
        - Documentation
        - Docker files if applicable
        - CI/CD configuration

        Provide production-ready code with best practices.
        """

        messages.append({"role": "user", "content": user_prompt})

        response = adapter.call_model(messages, temperature=0.2)

        # Parse JSON response
        try:
            result = json.loads(response.text)
            return TemplateResponse(**result)
        except json.JSONDecodeError:
            return TemplateResponse(
                files={"README.md": response.text},
                structure={"type": "basic", "description": "Generated template"},
                error="Failed to parse structured response"
            )

    except Exception as e:
        logger.error(f"Error generating templates: {str(e)}")
        return TemplateResponse(
            files={},
            structure={},
            error=str(e)
        )

@app.post("/scaffold_project")
async def scaffold_project(request: TemplateRequest):
    """Scaffold a complete project structure"""
    try:
        logger.info(f"Scaffolding project: {request.project_type}")

        # Generate templates first
        template_response = await generate_templates(request)

        if template_response.error:
            return {"error": template_response.error}

        # Save files to artifacts directory
        artifacts_dir = os.environ.get("ARTIFACTS_DIR")
        if artifacts_dir and os.path.exists(artifacts_dir):
            project_dir = os.path.join(artifacts_dir, f"{request.project_type}_project")
            os.makedirs(project_dir, exist_ok=True)

            saved_files = []
            for file_path, content in template_response.files.items():
                try:
                    full_path = os.path.join(project_dir, file_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    saved_files.append(file_path)
                    logger.info(f"Saved file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save {file_path}: {e}")

            return {
                "success": True,
                "project_directory": project_dir,
                "files_created": saved_files,
                "structure": template_response.structure
            }
        else:
            return {
                "success": False,
                "error": "Artifacts directory not configured",
                "files": template_response.files,
                "structure": template_response.structure
            }

    except Exception as e:
        logger.error(f"Error scaffolding project: {str(e)}")
        return {"error": str(e)}

@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return information about the architecture agent"""
    level = (detail or "").lower()
    responses = {
        "short": "Architecture agent for designing scalable systems and generating project templates",
        "medium": "Specialized agent for software architecture design, system scaffolding, and template generation. Handles complex project structures and provides best practices for large-scale applications.",
        "detailed": "The Architecture Agent provides comprehensive software architecture services including system design, component modeling, technology stack recommendations, project scaffolding, and template generation. It specializes in large-scale system design with considerations for scalability, maintainability, security, and operational excellence."
    }

    return {
        "level": level,
        "response": responses.get(level, responses["short"]),
        "capabilities": [
            "System architecture design",
            "Component modeling",
            "Technology stack recommendations",
            "Project scaffolding",
            "Template generation",
            "Scalability planning",
            "Security architecture",
            "Deployment design"
        ]
    }