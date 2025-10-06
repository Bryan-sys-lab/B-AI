import json
from providers.nim_adapter import NIMAdapter


class Router:
    def __init__(self):
        # Use NVIDIA NIM for intelligent routing with confidence scoring
        self.adapter = NIMAdapter()
        self.available_agents = ["fix_implementation_agent", "debugger_agent", "review_agent", "deployment_agent", "monitoring_agent", "testing_agent", "security_agent", "performance_agent", "comparator", "feedback_agent", "web_scraper", "task_classifier", "architecture", "knowledge_agent"]

    def consolidate_subtasks(self, subtasks: list) -> list:
        """Consolidate tiny or similar subtasks to prevent agent overload"""
        if len(subtasks) <= 3:
            return subtasks  # No need to consolidate small numbers

        consolidated = []
        current_batch = []
        batch_agent = None

        for subtask in subtasks:
            desc = subtask['description']
            word_count = len(desc.split())

            # If subtask is very small (micro-task) or same agent as batch
            if word_count < 8 or (batch_agent and subtask.get('agent') == batch_agent):
                current_batch.append(subtask)
                if not batch_agent:
                    batch_agent = subtask.get('agent', 'fix_implementation_agent')
            else:
                # Process current batch if it exists
                if current_batch:
                    if len(current_batch) == 1:
                        consolidated.append(current_batch[0])
                    else:
                        # Merge batch into single subtask
                        merged_desc = " and ".join([s['description'] for s in current_batch])
                        merged_subtask = {
                            "description": merged_desc,
                            "agent": batch_agent,
                            "priority": max([s.get('priority', 5) for s in current_batch]),
                            "confidence": sum([s.get('confidence', 0.5) for s in current_batch]) / len(current_batch)
                        }
                        consolidated.append(merged_subtask)
                    current_batch = []
                    batch_agent = None

                # Add current subtask
                consolidated.append(subtask)

        # Process remaining batch
        if current_batch:
            if len(current_batch) == 1:
                consolidated.append(current_batch[0])
            else:
                merged_desc = " and ".join([s['description'] for s in current_batch])
                merged_subtask = {
                    "description": merged_desc,
                    "agent": batch_agent or 'fix_implementation_agent',
                    "priority": max([s.get('priority', 5) for s in current_batch]),
                    "confidence": sum([s.get('confidence', 0.5) for s in current_batch]) / len(current_batch)
                }
                consolidated.append(merged_subtask)

        print(f"Router: Consolidated {len(subtasks)} subtasks into {len(consolidated)}")
        return consolidated

    async def route_subtasks(self, subtasks: list, task_context: dict = None) -> list:
        # First consolidate tiny subtasks
        consolidated_subtasks = self.consolidate_subtasks(subtasks)

        routed = []
        for subtask in consolidated_subtasks:
            agent = "fix_implementation_agent"  # Default agent
            confidence = 0.5  # Default confidence

            # Check for simple greetings and force fix_implementation
            desc_lower = subtask['description'].lower().strip()
            if desc_lower in ["hello", "hi", "greetings", "hello!", "hi!", "greetings!"]:
                agent = "fix_implementation_agent"
                confidence = 1.0
            # Prioritize planner's agent suggestion (highest priority)
            elif subtask.get('agent') and subtask['agent'] in self.available_agents:
                agent = subtask['agent']
                confidence = subtask.get('confidence', 0.8)
                print(f"Router: Using planner's agent suggestion: {agent} with confidence {confidence}")
            elif task_context and "classification" in task_context and task_context["classification"].get("suggested_agents"):
                # Use task classifier's suggested agents (medium priority)
                suggested_agents = task_context["classification"]["suggested_agents"]
                if suggested_agents and len(suggested_agents) > 0:
                    # Find first valid agent from suggestions
                    for suggested_agent in suggested_agents:
                        if suggested_agent in self.available_agents:
                            agent = suggested_agent
                            confidence = 0.9  # High confidence from classifier
                            print(f"Router: Using task classifier suggestion: {agent} with confidence {confidence}")
                            break
                    else:
                        # No valid agents found, fall through to keyword routing
                        pass  # Will fall through to else block
                else:
                    # Fallback to keyword-based routing
                    pass  # Will fall through to else block
            else:
                # Keyword-based routing as fallback
                desc_lower = subtask['description'].lower()

                # Define keyword mappings
                keyword_mappings = {
                    "fix_implementation": ["fix", "implement", "create", "write", "code", "function", "class", "bug", "error"],
                    "debugger": ["debug", "trace", "exception", "stack", "breakpoint", "pdb", "logging"],
                    "review": ["review", "quality", "best practices", "standards", "lint", "style"],
                    "testing": ["test", "pytest", "unittest", "coverage", "assert", "fixture"],
                    "security": ["security", "vulnerability", "exploit", "auth", "encrypt", "safe"],
                    "performance": ["performance", "optimize", "speed", "memory", "profile", "benchmark"],
                    "deployment": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "build"],
                    "monitoring": ["monitor", "log", "alert", "metrics", "observability"],
                    "feedback": ["feedback", "improve", "suggestion", "rate", "comment"],
                    "comparator": ["compare", "diff", "merge", "consensus", "validate"],
                    "architecture": ["design", "architecture", "system design", "architect", "structure", "diagram", "schema", "model", "plan", "blueprint"],
                    "memory_agent": ["knowledge", "memory", "learn", "remember", "store", "retrieve", "relationship", "graph", "history", "conversation"]
                }

                # Find best matching agent based on keywords
                best_agent = "fix_implementation_agent"  # default
                max_matches = 0
                for agent, keywords in keyword_mappings.items():
                    matches = sum(1 for keyword in keywords if keyword in desc_lower)
                    if matches > max_matches:
                        max_matches = matches
                        best_agent = agent

                # If keyword matching found a good match, use it with high confidence
                if max_matches > 0:
                    agent = best_agent
                    confidence = min(0.9, 0.5 + (max_matches * 0.1))  # Higher confidence for more matches
                    print(f"Router: Keyword-based routing to {agent} with confidence {confidence} (matches: {max_matches})")
                else:
                    # Use NIM for intelligent routing
                    prompt = f"""
You are an intelligent router for a multi-purpose agent system. Assign the best agent for this subtask with a confidence score.

Available agents and their expertise:
- fix_implementation: Code fixes, implementations, bug fixes, creating new functions/projects
- debugger: Debugging, error analysis, troubleshooting
- review: Code review, quality assessment, best practices
- deployment: Deployment, CI/CD, infrastructure
- monitoring: System monitoring, logging, observability
- testing: Unit tests, integration tests, test automation
- security: Security analysis, vulnerability assessment
- performance: Performance optimization, profiling
- comparator: Comparing outputs, validation, consensus
- feedback: User feedback, improvement suggestions
- memory_agent: Knowledge management, memory storage/retrieval, conversation history, knowledge graphs

Subtask: {subtask['description']}

Output JSON: {{"agent": "agent_name", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
                    messages = [{"role": "user", "content": prompt}]
                    max_routing_retries = 2
                    for routing_attempt in range(max_routing_retries):
                        try:
                            print(f"Router: Calling NIM adapter for subtask (attempt {routing_attempt + 1}): {subtask['description'][:50]}...")
                            response = self.adapter.call_model(messages)
                            print(f"Router: NIM response received: {response.text[:100]}...")
                            choice = json.loads(response.text)
                            agent = choice.get("agent", "fix_implementation")
                            # Validate that the agent is available
                            if agent not in self.available_agents:
                                agent = "fix_implementation_agent"  # Fallback to safe default
                            confidence = choice.get("confidence", 0.5)
                            print(f"Router: NIM routed to agent {agent} with confidence {confidence}")
                            break  # Success, exit retry loop
                        except Exception as e:
                            print(f"Router: NIM routing failed on attempt {routing_attempt + 1}: {e}")
                            if routing_attempt < max_routing_retries - 1:
                                import time
                                time.sleep(0.5 * (2 ** routing_attempt))  # Exponential backoff
                                continue
                            else:
                                # All retries failed, use keyword-based routing
                                print(f"Router: All NIM routing attempts failed, using keyword-based routing")
                                agent = best_agent
                                confidence = 0.7
            routed.append({
                "description": subtask["description"],
                "agent": agent,
                "confidence": confidence,
                "priority": subtask.get("priority", 5)
            })
        return routed