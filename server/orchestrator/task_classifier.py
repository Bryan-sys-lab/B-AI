"""
Aetherium-powered task classifier for intelligent routing of user requests.
Determines whether requests need task decomposition, direct responses, or special handling.
"""

import json
import logging
from typing import Dict, Any, Optional
from providers.nim_adapter import NIMAdapter

logger = logging.getLogger(__name__)

class TaskClassifier:
    """Aetherium-powered classifier for determining how to handle user requests"""

    def __init__(self):
        self.adapter = NIMAdapter()

    async def classify_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify a user request and determine the appropriate handling strategy.

        Returns:
            {
                "type": "task|query|about|direct_response",
                "complexity": "simple|medium|complex",
                "needs_decomposition": bool,
                "category": "coding|analysis|question|system|other",
                "confidence": float,
                "reasoning": str,
                "suggested_agents": [list of agent names if applicable],
                "direct_response": str (if type is direct_response)
            }
        """

        # Handle obvious system queries first
        system_keywords = [
            "what are you", "who are you", "what can you do", "your capabilities",
            "tell me about yourself", "what do you do", "help", "about you",
            "system info", "your features", "what are your features"
        ]

        lower_input = user_input.lower().strip()
        if any(keyword in lower_input for keyword in system_keywords):
            return {
                "type": "about",
                "complexity": "simple",
                "needs_decomposition": False,
                "category": "system",
                "confidence": 0.95,
                "reasoning": "User is asking about system capabilities or identity",
                "suggested_agents": [],
                "direct_response": None
            }

        # Handle simple conversational queries (but not when clearly part of coding context)
        simple_queries = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "thanks", "thank you"
        ]

        # Don't treat "hello" as conversational if it's clearly part of "hello world" programming context
        is_hello_world_context = "hello world" in lower_input or ("hello" in lower_input and ("function" in lower_input or "print" in lower_input or "output" in lower_input))

        if any(query in lower_input for query in simple_queries) and not is_hello_world_context:
            return {
                "type": "direct_response",
                "complexity": "simple",
                "needs_decomposition": False,
                "category": "conversation",
                "confidence": 0.9,
                "reasoning": "Simple conversational greeting or acknowledgment",
                "suggested_agents": [],
                "direct_response": self._get_conversational_response(lower_input)
            }

        # Use Aetherium to classify more complex requests
        return await self._classify_with_ai(user_input, context)

    async def _classify_with_ai(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use Aetherium to classify complex requests"""

        prompt = f"""
Analyze this user request and classify it according to the following categories:

User Request: "{user_input}"

Context: {json.dumps(context) if context else "None"}

Classification Criteria:

1. **Type**:
   - "task": Requires actual work (coding, debugging, testing, deployment)
   - "query": Questions about existing code, explanations, analysis
   - "about": Questions about system capabilities
   - "direct_response": Simple responses that don't need processing

2. **Complexity**:
   - "simple": Can be answered directly or with minimal processing
   - "medium": Requires some analysis but not full task decomposition
   - "complex": Needs multiple steps and agent coordination

3. **Category**:
   - "coding": Writing, modifying, or generating code
   - "analysis": Understanding, explaining, or reviewing code
   - "question": General questions or clarifications
   - "system": About the Aetherium system itself
   - "other": Miscellaneous

4. **Needs Decomposition**: Whether this requires breaking down into subtasks

Respond with JSON in this exact format:
{{
    "type": "task|query|about|direct_response",
    "complexity": "simple|medium|complex",
    "needs_decomposition": true|false,
    "category": "coding|analysis|question|system|other",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification",
    "suggested_agents": ["agent1", "agent2"] (empty list if not applicable),
    "direct_response": "Response text if type is direct_response, otherwise null"
}}
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.adapter.call_model(messages)

            if hasattr(response, 'text'):
                result_text = response.text.strip()
            else:
                result_text = str(response).strip()

            # Try to parse JSON response
            try:
                classification = json.loads(result_text)
                # Validate required fields
                required_fields = ["type", "complexity", "needs_decomposition", "category", "confidence", "reasoning"]
                if all(field in classification for field in required_fields):
                    # Ensure suggested_agents is a list
                    if "suggested_agents" not in classification:
                        classification["suggested_agents"] = []
                    if "direct_response" not in classification:
                        classification["direct_response"] = None
                    return classification
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Aetherium classification response: {result_text}")

        except Exception as e:
            logger.error(f"Error in Aetherium classification: {e}")

        # Fallback classification
        return self._fallback_classification(user_input)

    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        """Fallback classification when Aetherium fails"""
        lower_input = user_input.lower()

        # Check for coding keywords
        coding_keywords = ["write", "create", "implement", "build", "develop", "code", "function", "class", "fix", "debug", "test", "print", "output", "script", "program"]
        if any(keyword in lower_input for keyword in coding_keywords):
            return {
                "type": "task",
                "complexity": "medium",
                "needs_decomposition": True,
                "category": "coding",
                "confidence": 0.7,
                "reasoning": "Contains coding-related keywords, likely needs implementation",
                "suggested_agents": ["fix_implementation"],
                "direct_response": None
            }

        # Default to query/analysis
        return {
            "type": "query",
            "complexity": "simple",
            "needs_decomposition": False,
            "category": "question",
            "confidence": 0.6,
            "reasoning": "General question or request, using fallback classification",
            "suggested_agents": [],
            "direct_response": None
        }

    def _get_conversational_response(self, input_text: str) -> str:
        """Generate appropriate conversational responses"""
        if any(word in input_text for word in ["hello", "hi", "hey"]):
            return "Hello! I'm here to help you with coding tasks, debugging, testing, and more. What would you like to work on?"
        elif "how are you" in input_text or "how's it going" in input_text:
            return "I'm doing well, thank you! Ready to help with your coding projects. What can I assist you with today?"
        elif any(word in input_text for word in ["thanks", "thank you"]):
            return "You're welcome! Let me know if you need help with anything else."
        else:
            return "Hi there! I'm your Aetherium coding assistant. I can help you write code, debug issues, run tests, and much more. What would you like to work on?"

# Global classifier instance
task_classifier = TaskClassifier()