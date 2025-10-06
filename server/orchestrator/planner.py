import json
import os
import sys

# Ensure repo root is available on sys.path for imports when running inside
# containerized contexts.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    # Intentional: ensure repo root is available when running inside a container
    sys.path.insert(0, os.path.abspath(os.path.join(_repo_root, "..")))  # noqa: E402

from providers.nim_adapter import NIMAdapter  # noqa: E402
from orchestrator.response_parser import safe_parse_with_validation, validate_plan_structure  # noqa: E402


class Planner:
    def __init__(self):
        # Use NVIDIA NIM for intelligent task decomposition
        self.adapter = NIMAdapter()

    def assess_complexity(self, task_description: str) -> dict:
        """Assess task complexity to determine decomposition strategy"""
        desc_lower = task_description.lower().strip()

        # Simple greetings - no decomposition
        simple_greetings = ["hello", "hi", "greetings", "hello!", "hi!", "greetings!"]
        if desc_lower in simple_greetings:
            return {"level": "simple", "max_subtasks": 1, "strategy": "direct"}

        # Complexity indicators
        word_count = len(task_description.split())
        sentence_count = task_description.count('.') + task_description.count('!') + task_description.count('?')
        has_complex_verbs = any(word in desc_lower for word in
            ['create', 'build', 'implement', 'develop', 'design', 'architect', 'system', 'application'])
        has_multiple_steps = any(word in desc_lower for word in
            ['and', 'then', 'after', 'finally', 'next', 'step', 'phase'])

        complexity_score = 0
        complexity_score += min(word_count // 20, 3)  # 0-3 points for length
        complexity_score += min(sentence_count, 3)    # 0-3 points for sentences
        complexity_score += 2 if has_complex_verbs else 0  # 2 points for complex verbs
        complexity_score += 1 if has_multiple_steps else 0  # 1 point for multi-step indicators

        if complexity_score <= 2:
            return {"level": "simple", "max_subtasks": 2, "strategy": "minimal"}
        elif complexity_score <= 5:
            return {"level": "medium", "max_subtasks": 5, "strategy": "balanced"}
        else:
            return {"level": "complex", "max_subtasks": 10, "strategy": "detailed"}

    async def decompose_task(self, task_description: str, context: dict = None) -> dict:
        print(f"Planner: Decomposing task: {task_description}")

        # Assess complexity
        complexity = self.assess_complexity(task_description)
        print(f"Planner: Complexity assessment - Level: {complexity['level']}, Max subtasks: {complexity['max_subtasks']}, Strategy: {complexity['strategy']}")

        # Handle simple cases directly
        if complexity['level'] == 'simple':
            print(f"Planner: Simple task detected, using direct assignment")
            return {
                "subtasks": [
                    {"description": task_description, "agent": "fix_implementation", "priority": 5, "confidence": 1.0}
                ]
            }
        context_str = ""
        if context:
            context_str = f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"

        # Adjust prompt based on complexity
        if complexity['strategy'] == 'minimal':
            decomposition_guidance = f"Create 1-2 high-level subtasks that can be handled by individual agents. Avoid overly granular steps."
        elif complexity['strategy'] == 'balanced':
            decomposition_guidance = f"Create 3-5 focused subtasks that represent logical phases of the task."
        else:  # detailed
            decomposition_guidance = f"Create up to {complexity['max_subtasks']} specific, actionable subtasks that break down the task comprehensively."

        prompt = f"""
You are an intelligent task planner for a code agent system. Decompose the following task into appropriate subtasks based on its complexity.

Task Complexity Assessment: {complexity['level']} ({complexity['strategy']} strategy)
{decomposition_guidance}

Available agents: fix_implementation, debugger, review, deployment, monitoring, testing, security, performance, comparator, feedback.

Task: {task_description}{context_str}

Output a JSON object with:
- "subtasks": list of objects (max {complexity['max_subtasks']}), each with:
  - "description" (string): Clear, actionable subtask
  - "agent" (string from available agents)
  - "priority" (int, 1-10, where 10 is highest)
  - "confidence" (float, 0.0-1.0: how well the agent fits this subtask)

Guidelines:
- For simple tasks: Use 1-2 comprehensive subtasks
- For medium tasks: Use 3-5 focused subtasks
- For complex tasks: Use detailed decomposition but avoid micro-steps
- Ensure subtasks are meaningful units that agents can actually execute
- Assign realistic confidence scores based on agent expertise
"""
        messages = [{"role": "user", "content": prompt}]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Planner: Calling NIM adapter for task decomposition (attempt {attempt + 1}/{max_retries})")
                response = self.adapter.call_model(messages)
                print(f"Planner: NIM response received: {response.text[:200]}...")

                # Use robust parsing with validation
                print(f"Planner: Attempting robust parsing of response (length: {len(response.text)})")
                plan, parse_method, is_valid = safe_parse_with_validation(
                    response.text,
                    validate_plan_structure
                )

                print(f"Planner: Parse result - method: {parse_method}, valid: {is_valid}, plan: {plan is not None}")
                if plan and is_valid:
                    print(f"Planner: Successfully parsed plan with {len(plan.get('subtasks', []))} subtasks using {parse_method}")
                    return plan
                else:
                    print(f"Planner: Parsing failed (method: {parse_method}, valid: {is_valid}), trying again...")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1 * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        print(f"Planner: All parsing attempts failed, using intelligent fallback")
                        # Intelligent fallback based on task analysis
                        return self._create_fallback_plan(task_description, complexity)
            except Exception as e:
                print(f"Planner: Exception on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    print(f"Planner: All attempts failed, using intelligent fallback")
                    return self._create_fallback_plan(task_description, complexity)

        # This should never be reached, but just in case
        return self._create_fallback_plan(task_description, complexity)

    def _create_fallback_plan(self, task_description: str, complexity: dict) -> dict:
        """Create an intelligent fallback plan based on task analysis"""
        desc_lower = task_description.lower()

        # Analyze task for better agent assignment
        if any(word in desc_lower for word in ['design', 'architecture', 'system', 'diagram', 'blueprint']):
            agent = 'architecture'
            confidence = 0.9
        elif any(word in desc_lower for word in ['test', 'testing', 'pytest', 'unittest']):
            agent = 'testing'
            confidence = 0.9
        elif any(word in desc_lower for word in ['deploy', 'deployment', 'docker', 'kubernetes']):
            agent = 'deployment'
            confidence = 0.9
        elif any(word in desc_lower for word in ['security', 'vulnerability', 'auth', 'encrypt']):
            agent = 'security'
            confidence = 0.9
        elif any(word in desc_lower for word in ['performance', 'optimize', 'speed', 'memory']):
            agent = 'performance'
            confidence = 0.9
        elif any(word in desc_lower for word in ['debug', 'fix', 'error', 'bug']):
            agent = 'fix_implementation'
            confidence = 0.8
        else:
            agent = 'fix_implementation'
            confidence = 0.7

        # For complex tasks, create multiple subtasks
        if complexity['level'] == 'complex':
            return {
                "subtasks": [
                    {
                        "description": f"Analyze and plan: {task_description}",
                        "agent": agent,
                        "priority": 8,
                        "confidence": confidence
                    },
                    {
                        "description": f"Implement the core functionality: {task_description}",
                        "agent": agent,
                        "priority": 7,
                        "confidence": confidence
                    },
                    {
                        "description": f"Test and validate the implementation: {task_description}",
                        "agent": "testing",
                        "priority": 6,
                        "confidence": 0.8
                    }
                ]
            }
        elif complexity['level'] == 'medium':
            return {
                "subtasks": [
                    {
                        "description": f"Implement: {task_description}",
                        "agent": agent,
                        "priority": 7,
                        "confidence": confidence
                    },
                    {
                        "description": f"Review and improve: {task_description}",
                        "agent": "review",
                        "priority": 6,
                        "confidence": 0.8
                    }
                ]
            }
        else:
            return {
                "subtasks": [
                    {
                        "description": task_description,
                        "agent": agent,
                        "priority": 5,
                        "confidence": confidence
                    }
                ]
            }