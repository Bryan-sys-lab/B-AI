import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible so adapters and
# other modules that read environment variables (e.g. providers/*) will see
# the values when they're imported or instantiated.
load_dotenv()

# Ensure repo root is on sys.path before any other imports so local modules
# (e.g., scraper) and top-level packages resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    # Intentional import-time sys.path modification to ensure repo root is
    # available in containerized or subdirectory run contexts. Lint rule E402
    # (module level import not at top) is intentionally suppressed here.
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import List, Optional, Dict, Any  # noqa: E402
import logging  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from providers.system_prompt import AGENT_PROMPTS  # noqa: E402

app = FastAPI(title="Research and Best Practices Specialist Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchRequest(BaseModel):
    topic: str
    technology_stack: Optional[List[str]] = None
    project_type: Optional[str] = None
    depth: str = "comprehensive"  # "basic", "comprehensive", "expert"
    include_code_examples: bool = True
    include_sources: bool = True
    context: Optional[str] = None

class BestPracticesRequest(BaseModel):
    technology: str
    use_case: Optional[str] = None
    current_year: int = 2024
    include_trends: bool = True

class SynthesisRequest(BaseModel):
    research_topic: str
    target_audience: str = "developers"
    format: str = "structured"  # "structured", "narrative", "bullet_points"

class ResearchResponse(BaseModel):
    research_summary: str
    best_practices: List[str]
    code_examples: List[Dict[str, str]]
    sources: List[str]
    recommendations: List[str]
    confidence_score: float
    error: Optional[str] = None

# Legacy models for backward compatibility
class ScrapeRequest(BaseModel):
    url: str
    selectors: Optional[Dict[str, str]] = None
    verify_ssl: bool = True
    timeout: int = 30
    use_selenium: bool = False
    wait_for_element: Optional[str] = None
    javascript_delay: int = 2

class ScrapeResponse(BaseModel):
    result: Dict[str, Any]
    error: Optional[str] = None

class BulkScrapeRequest(BaseModel):
    urls: List[str]
    selectors: Optional[Dict[str, str]] = None
    verify_ssl: bool = True
    timeout: int = 30
    use_selenium: bool = False
    wait_for_element: Optional[str] = None
    javascript_delay: int = 2

class BulkScrapeResponse(BaseModel):
    results: List[Dict[str, Any]]
    error: Optional[str] = None

class StructuredScrapeRequest(BaseModel):
    url: str
    schema: Dict[str, Any]
    verify_ssl: bool = True

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """Conduct comprehensive research on best practices and industry standards"""
    try:
        logger.info(f"Conducting research on: {request.topic}")

        # Use NIM adapter for research
        adapter = NIMAdapter(role="researcher")

        # Build comprehensive research prompt
        system_prompt = AGENT_PROMPTS.get("web_scraper", """
        You are a Research and Best Practices Specialist with internet access. Your role is to:
        1. Research current industry standards and best practices
        2. Analyze technology trends and implementation approaches
        3. Provide evidence-based recommendations
        4. Synthesize information from authoritative sources
        5. Focus on maintainability, scalability, and modern development practices

        Always provide specific, actionable insights with source attribution.
        """)

        # Build research query
        research_query = f"""
        Conduct comprehensive research on: {request.topic}

        Technology Stack: {', '.join(request.technology_stack) if request.technology_stack else 'Not specified'}
        Project Type: {request.project_type or 'General software development'}
        Depth Level: {request.depth}

        Please provide:
        1. Current industry best practices and standards
        2. Technology recommendations with rationale
        3. Implementation patterns and approaches
        4. Code examples demonstrating best practices
        5. Common pitfalls to avoid
        6. Future trends and considerations

        Focus on practical, implementable solutions with real-world examples.
        Include specific technologies, frameworks, and tools with their use cases.
        """

        if request.context:
            research_query += f"\n\nAdditional Context: {request.context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": research_query}
        ]

        response = adapter.call_model(messages, temperature=0.3)

        # Parse and structure the research response
        research_data = await _parse_research_response(response.text, request)

        return ResearchResponse(**research_data)

    except Exception as e:
        logger.error(f"Error in research: {str(e)}")
        return ResearchResponse(
            research_summary="",
            best_practices=[],
            code_examples=[],
            sources=[],
            recommendations=[],
            confidence_score=0.0,
            error=str(e)
        )

@app.post("/best_practices", response_model=ResearchResponse)
async def get_best_practices(request: BestPracticesRequest):
    """Get best practices for a specific technology"""
    try:
        logger.info(f"Getting best practices for: {request.technology}")

        adapter = NIMAdapter(role="researcher")

        system_prompt = AGENT_PROMPTS.get("web_scraper", "You are a research specialist.")

        query = f"""
        Provide comprehensive best practices for {request.technology}.

        Use Case: {request.use_case or 'General development'}
        Current Year: {request.current_year}
        Include Trends: {'Yes' if request.include_trends else 'No'}

        Cover:
        1. Current best practices and patterns
        2. Common anti-patterns to avoid
        3. Performance optimization techniques
        4. Security considerations
        5. Testing strategies
        6. Deployment and maintenance practices
        7. Future trends and evolution

        Provide specific, actionable recommendations with examples.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = adapter.call_model(messages, temperature=0.2)

        research_data = await _parse_research_response(response.text, ResearchRequest(
            topic=f"Best practices for {request.technology}",
            technology_stack=[request.technology],
            depth="comprehensive"
        ))

        return ResearchResponse(**research_data)

    except Exception as e:
        logger.error(f"Error getting best practices: {str(e)}")
        return ResearchResponse(
            research_summary="",
            best_practices=[],
            code_examples=[],
            sources=[],
            recommendations=[],
            confidence_score=0.0,
            error=str(e)
        )

@app.post("/synthesize", response_model=ResearchResponse)
async def synthesize_research(request: SynthesisRequest):
    """Synthesize research findings into coherent recommendations"""
    try:
        logger.info(f"Synthesizing research on: {request.research_topic}")

        adapter = NIMAdapter(role="researcher")

        system_prompt = AGENT_PROMPTS.get("web_scraper", "You are a research specialist.")

        synthesis_query = f"""
        Synthesize research findings on: {request.research_topic}

        Target Audience: {request.target_audience}
        Format: {request.format}

        Existing Knowledge: {request.existing_knowledge or 'None provided'}

        Please provide a comprehensive synthesis that:
        1. Integrates multiple perspectives and sources
        2. Identifies key patterns and trends
        3. Provides clear, actionable recommendations
        4. Addresses potential conflicts or trade-offs
        5. Considers the target audience's needs and context

        Structure the response according to the requested format ({request.format}).
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": synthesis_query}
        ]

        response = adapter.call_model(messages, temperature=0.2)

        research_data = await _parse_research_response(response.text, ResearchRequest(
            topic=request.research_topic,
            depth="comprehensive"
        ))

        return ResearchResponse(**research_data)

    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}")
        return ResearchResponse(
            research_summary="",
            best_practices=[],
            code_examples=[],
            sources=[],
            recommendations=[],
            confidence_score=0.0,
            error=str(e)
        )

async def _parse_research_response(response_text: str, request: ResearchRequest) -> dict:
    """Parse Aetherium response into structured research data"""
    try:
        # Try to extract JSON from response
        import json
        import re

        # Look for JSON blocks in the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return {
                    "research_summary": data.get("research_summary", response_text[:500]),
                    "best_practices": data.get("best_practices", []),
                    "code_examples": data.get("code_examples", []),
                    "sources": data.get("sources", []),
                    "recommendations": data.get("recommendations", []),
                    "confidence_score": data.get("confidence_score", 0.8),
                    "error": None
                }
            except json.JSONDecodeError:
                pass

        # Fallback: extract sections from text
        sections = {
            "research_summary": "",
            "best_practices": [],
            "code_examples": [],
            "sources": [],
            "recommendations": [],
            "confidence_score": 0.7,
            "error": None
        }

        # Extract summary (first paragraph or first 300 chars)
        lines = response_text.split('\n')
        summary = ""
        for line in lines[:5]:
            if line.strip() and not line.startswith('#'):
                summary += line + " "
                if len(summary) > 300:
                    break
        sections["research_summary"] = summary.strip() or response_text[:300]

        # Extract recommendations (look for numbered or bulleted lists)
        rec_pattern = r'(\d+\.|\*|-)\s*([^\n]+)'
        recommendations = re.findall(rec_pattern, response_text)
        sections["recommendations"] = [rec[1].strip() for rec in recommendations[:5]]

        # Extract code examples (look for code blocks)
        code_pattern = r'```(?:\w+)?\n(.*?)\n```'
        code_blocks = re.findall(code_pattern, response_text, re.DOTALL)
        sections["code_examples"] = [{"language": "text", "code": block.strip(), "description": f"Example {i+1}"} for i, block in enumerate(code_blocks[:3])]

        return sections

    except Exception as e:
        logger.error(f"Error parsing research response: {str(e)}")
        return {
            "research_summary": response_text[:500],
            "best_practices": [],
            "code_examples": [],
            "sources": [],
            "recommendations": ["Review the research findings provided"],
            "confidence_score": 0.5,
            "error": None
        }

@app.post("/execute")
async def execute_task(request: dict):
    try:
        description = request.get("description", "").strip()
        logger.info(f"Executing research task: {description}")

        # Determine if this is a research task
        research_keywords = ["research", "best practices", "standards", "patterns", "recommendations", "guide", "tutorial", "how to"]
        is_research_task = any(keyword in description.lower() for keyword in research_keywords)

        if is_research_task:
            # Extract topic from description
            topic = description
            # Remove common prefixes
            for prefix in ["research", "find", "get", "show me", "tell me about"]:
                if topic.lower().startswith(prefix):
                    topic = topic[len(prefix):].strip()
                    break

            # Perform research
            research_request = ResearchRequest(
                topic=topic,
                depth="comprehensive",
                include_code_examples=True,
                include_sources=True
            )

            result = await conduct_research(research_request)
            return {
                "result": result.research_summary,
                "structured": {
                    "best_practices": result.best_practices,
                    "code_examples": result.code_examples,
                    "sources": result.sources,
                    "recommendations": result.recommendations
                },
                "success": True
            }
        else:
            # Fallback to general task execution
            adapter = NIMAdapter(role="researcher")
            system_prompt = AGENT_PROMPTS.get("web_scraper", "You are a helpful assistant.")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description}
            ]

            response = adapter.call_model(messages, temperature=0.3)
            return {
                "result": response.text,
                "success": True,
                "tokens": response.tokens,
                "latency_ms": response.latency_ms
            }

    except Exception as e:
        logger.error(f"Error executing research task: {str(e)}")
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