import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env as early as possible
load_dotenv()

# Ensure repo root is on sys.path before any other imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import Optional, Dict, Any  # noqa: E402
import logging  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from common.utils import is_running_in_container  # noqa: E402
from common.endpoints import add_health_endpoint  # noqa: E402

app = FastAPI(title="Task Classifier Agent")

add_health_endpoint(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None

class ClassificationResponse(BaseModel):
    type: str  # "task", "query", "about", "direct_response"
    complexity: str  # "simple", "medium", "complex"
    needs_decomposition: bool
    category: str  # "coding", "analysis", "question", "system", "other"
    confidence: float
    reasoning: str
    suggested_agents: Optional[list] = None
    direct_response: Optional[str] = None

@app.post("/classify", response_model=ClassificationResponse)
async def classify_task(request: ClassificationRequest):
    """Classify a user request and determine the appropriate handling strategy"""
    try:
        logger.info(f"Classifying request: {request.user_input[:100]}...")

        # Initialize NIM adapter for classification with task_classifier role
        adapter = NIMAdapter(role="task_classifier")

        # Build classification prompt
        prompt = f"""
You are an expert task classifier for a sophisticated Aetherium coding assistant system. Your role is to intelligently analyze user requests and route them to the most appropriate agents and processing strategies.

SYSTEM CONTEXT:
- This is a multi-agent coding system with specialized agents for different tasks
- Available agents: fix_implementation, debugger, review, testing, security, performance, deployment, monitoring, feedback, architecture, task_classifier, web_scraper
- The system handles coding, debugging, testing, deployment, and analysis tasks

USER REQUEST: "{request.user_input}"

CONTEXT: {json.dumps(request.context) if request.context else "None"}

CLASSIFICATION FRAMEWORK:

1. **TYPE** (choose exactly one):
   - "task": Requires actual work, implementation, or execution (coding, debugging, testing, deployment, analysis work)
   - "query": Questions about existing code, explanations, understanding, or analysis requests
   - "about": Questions about system capabilities, what the Aetherium can do, or system information
   - "direct_response": Simple conversational responses, greetings, or acknowledgments

2. **COMPLEXITY** (choose exactly one):
   - "simple": Can be handled directly or with minimal processing (single-step tasks)
   - "medium": Requires some analysis but can be handled by one primary agent
   - "complex": Needs multiple steps, agent coordination, or significant planning

3. **CATEGORY** (choose exactly one):
   - "coding": Writing, modifying, generating, or implementing code
   - "architecture": System design, architecture planning, structural design, or high-level planning
   - "analysis": Understanding, explaining, reviewing, or analyzing existing code
   - "question": General questions, clarifications, or information requests
   - "system": About the Aetherium system itself, capabilities, or how it works
   - "other": Miscellaneous or unclear requests

4. **DECOMPOSITION NEEDS**: true if the task requires breaking down into multiple subtasks that different agents should handle

5. **AGENT SUGGESTIONS**: For tasks needing decomposition, suggest 1-3 most appropriate agents from the available list

CLASSIFICATION RULES:
- "Hello world" or simple code examples → coding task (not direct_response)
- Questions about code behavior, bugs, or understanding → analysis or task
- Questions about what the Aetherium can do → about
- Simple greetings like "hi", "thanks" → direct_response
- Any request involving writing/modifying code → coding task
- System design or planning requests → architecture task
- Debugging or fixing issues → task with debugger agent
- Testing requests → task with testing agent
- Deployment or infrastructure → task with deployment agent
- Follow-up queries like "explain more", "tell me more", "what about", "how does it work", "can you elaborate" → query with context_retrieval
- Conversational continuations like "and then", "also", "furthermore", "additionally" → task with context_continuation

RESPONSE FORMAT (JSON only):
{{
    "type": "task|query|about|direct_response",
    "complexity": "simple|medium|complex",
    "needs_decomposition": true|false,
    "category": "coding|architecture|analysis|question|system|other",
    "confidence": 0.0-1.0,
    "reasoning": "Clear explanation of why this classification was chosen",
    "suggested_agents": ["agent1", "agent2"],
    "direct_response": "Response text only if type is direct_response, otherwise null"
}}

Be precise and context-aware in your classification.
"""

        messages = [{"role": "user", "content": prompt}]

        # Call the model
        response = adapter.call_model(messages, temperature=0.1)  # Low temperature for consistent classification

        # Parse the response
        if hasattr(response, 'text'):
            result_text = response.text.strip()
        else:
            result_text = str(response).strip()

        # Try to parse JSON response
        try:
            # Strip markdown code blocks if present
            cleaned_text = result_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]  # Remove ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```
            cleaned_text = cleaned_text.strip()

            classification = json.loads(cleaned_text)

            # Validate required fields
            required_fields = ["type", "complexity", "needs_decomposition", "category", "confidence", "reasoning"]
            if all(field in classification for field in required_fields):
                # Ensure suggested_agents is a list
                if "suggested_agents" not in classification:
                    classification["suggested_agents"] = []
                if "direct_response" not in classification:
                    classification["direct_response"] = None

                logger.info(f"Classification result: {classification['type']} ({classification['category']})")
                return ClassificationResponse(**classification)
            else:
                logger.warning(f"Missing required fields in classification: {classification}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification JSON: {e}")
            logger.error(f"Raw response: {result_text}")

        # Fallback classification if Aetherium fails
        return await fallback_classification(request.user_input, request.context)

    except Exception as e:
        logger.error(f"Error in task classification: {str(e)}")
        return await fallback_classification(request.user_input, request.context)

async def fallback_classification(user_input: str, context: Optional[Dict[str, Any]] = None):
    """Fallback classification when Aetherium fails"""
    logger.info("Using fallback classification")

    lower_input = user_input.lower()

    # Handle obvious system queries first
    system_keywords = [
        "what are you", "who are you", "what can you do", "your capabilities",
        "tell me about yourself", "what do you do", "help", "about you",
        "system info", "your features", "what are your features"
    ]

    if any(keyword in lower_input for keyword in system_keywords):
        return ClassificationResponse(
            type="about",
            complexity="simple",
            needs_decomposition=False,
            category="system",
            confidence=0.95,
            reasoning="User is asking about system capabilities or identity",
            suggested_agents=[],
            direct_response=None
        )

    # Handle follow-up queries that need context
    followup_keywords = [
        "explain more", "tell me more", "what about", "how does it work",
        "can you elaborate", "give me details", "expand on", "elaborate",
        "what do you mean", "clarify", "more details", "further explanation",
        "and then", "also", "furthermore", "additionally", "besides that"
    ]

    if any(keyword in lower_input for keyword in followup_keywords):
        return ClassificationResponse(
            type="query",
            complexity="simple",
            needs_decomposition=False,
            category="followup",
            confidence=0.9,
            reasoning="Follow-up query that needs conversation context to provide relevant response",
            suggested_agents=[],
            direct_response=None
        )

    # Handle simple conversational queries (but not when clearly part of coding context)
    simple_queries = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "how's it going", "what's up", "thanks", "thank you"
    ]

    # Don't treat "hello" as conversational if it's clearly part of coding context
    is_hello_world_context = "hello world" in lower_input or (
        "hello" in lower_input and (
            "function" in lower_input or "print" in lower_input or "output" in lower_input
        )
    )

    if any(query in lower_input for query in simple_queries) and not is_hello_world_context:
        return ClassificationResponse(
            type="direct_response",
            complexity="simple",
            needs_decomposition=False,
            category="conversation",
            confidence=0.9,
            reasoning="Simple conversational greeting or acknowledgment",
            suggested_agents=[],
            direct_response=get_conversational_response(lower_input)
        )

    # Check for architecture/design keywords
    architecture_keywords = ["design", "architecture", "system design", "architect", "structure", "diagram", "schema", "model", "plan", "blueprint"]
    if any(keyword in lower_input for keyword in architecture_keywords):
        return ClassificationResponse(
            type="task",
            complexity="medium",
            needs_decomposition=True,
            category="architecture",
            confidence=0.8,
            reasoning="Contains architecture/design-related keywords, likely needs system design",
            suggested_agents=["architecture"],
            direct_response=None
        )

    # Check for coding keywords
    coding_keywords = ["write", "create", "implement", "build", "develop", "code", "function", "class", "fix", "debug", "test", "print", "output", "script", "program"]
    if any(keyword in lower_input for keyword in coding_keywords):
        return ClassificationResponse(
            type="task",
            complexity="medium",
            needs_decomposition=True,
            category="coding",
            confidence=0.7,
            reasoning="Contains coding-related keywords, likely needs implementation",
            suggested_agents=["fix_implementation"],
            direct_response=None
        )

    # Default to task - be more permissive and accept any type of prompt
    return ClassificationResponse(
        type="task",
        complexity="medium",
        needs_decomposition=True,
        category="coding",
        confidence=0.8,
        reasoning="Accepting any type of prompt as a task to be processed by the system",
        suggested_agents=["fix_implementation"],
        direct_response=None
    )

def get_conversational_response(input_text: str) -> str:
    """Generate appropriate conversational responses"""
    if any(word in input_text for word in ["hello", "hi", "hey"]):
        return "Hello! I'm here to help you with coding tasks, debugging, testing, and more. What would you like to work on?"
    elif "how are you" in input_text or "how's it going" in input_text:
        return "I'm doing well, thank you! Ready to help with your coding projects. What can I assist you with today?"
    elif any(word in input_text for word in ["thanks", "thank you"]):
        return "You're welcome! Let me know if you need help with anything else."
    else:
        return "Hi there! I'm your Aetherium coding assistant. I can help you write code, debug issues, run tests, and much more. What would you like to work on?"

@app.post("/execute")
async def execute_task(request: dict):
    """Execute a task by classifying it and providing routing information"""
    try:
        description = request.get("description", "").strip()
        logger.info(f"Executing classification task: {description}")

        # Simple canned response still allowed for trivial 'hello'
        if description.lower() == "hello":
            return {"result": "Hello, world!", "success": True}

        # For classification tasks, delegate to the classify endpoint
        if "classify" in description.lower() or "categorize" in description.lower():
            # This is a meta-classification request
            return {
                "result": "I am the task classifier agent. I can help classify user requests and route them to appropriate agents.",
                "success": True
            }

        # For general tasks, classify the request
        classification_request = ClassificationRequest(
            user_input=description,
            context=request.get("context")
        )

        classification = await classify_task(classification_request)

        return {
            "result": f"Request classified as: {classification.type} ({classification.category})",
            "classification": {
                "type": classification.type,
                "complexity": classification.complexity,
                "category": classification.category,
                "needs_decomposition": classification.needs_decomposition,
                "suggested_agents": classification.suggested_agents,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning
            },
            "success": True
        }

    except Exception as e:
        logger.error(f"Error executing classification task: {str(e)}")
        return {"error": str(e)}

@app.get("/about")
def about():
    """Return information about the task classifier agent"""
    return {
        "name": "Task Classifier Agent",
        "description": "Intelligent classification of user requests for the Aetherium system",
        "capabilities": [
            "Request type classification (task, query, about, direct_response)",
            "Complexity assessment (simple, medium, complex)",
            "Agent suggestion for task decomposition",
            "Context-aware classification"
        ],
        "model": "NVIDIA NIM",
        "endpoint": "/classify"
    }