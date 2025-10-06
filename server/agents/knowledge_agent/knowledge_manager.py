import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

class KnowledgeManager:
    def __init__(self):
        self.vector_store_url = os.environ.get("VECTOR_STORE_URL", "http://localhost:8019")
        self.conversation_store = {}  # In-memory for now, could be persisted to database

    async def initialize(self):
        """Initialize the knowledge manager"""
        logger.info("Initializing Knowledge Manager")
        # Test connection to vector store
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_store_url}/health")
                if response.status_code == 200:
                    logger.info("Vector store connection successful")
                else:
                    logger.warning("Vector store health check failed")
        except Exception as e:
            logger.error(f"Failed to connect to vector store: {e}")

    async def store_knowledge(self, content: str, metadata: Dict[str, Any], knowledge_type: str) -> str:
        """Store knowledge in the vector store"""
        try:
            # Add knowledge type to metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata["knowledge_type"] = knowledge_type
            enhanced_metadata["timestamp"] = asyncio.get_event_loop().time()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_store_url}/add_text",
                    json={"text": content, "metadata": enhanced_metadata}
                )
                if response.status_code == 200:
                    result = response.json()
                    knowledge_id = str(result.get("id", "unknown"))
                    logger.info(f"Stored knowledge with ID: {knowledge_id}")
                    return knowledge_id
                else:
                    logger.error(f"Failed to store knowledge: {response.status_code}")
                    raise Exception(f"Vector store error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            raise

    async def retrieve_knowledge(self, query: str, knowledge_type: Optional[str] = None,
                               limit: int = 5, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge based on query"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_store_url}/search_text",
                    json={"query": query, "k": limit}
                )
                if response.status_code == 200:
                    result = response.json()
                    results = result.get("results", [])

                    # Filter by knowledge type if specified
                    if knowledge_type:
                        results = [r for r in results if r.get("metadata", {}).get("knowledge_type") == knowledge_type]

                    # Enhance results with relevance scoring and context
                    enhanced_results = []
                    for r in results:
                        enhanced_result = r.copy()
                        enhanced_result["relevance_score"] = r.get("score", 0)
                        enhanced_result["knowledge_type"] = r.get("metadata", {}).get("knowledge_type", "unknown")
                        enhanced_results.append(enhanced_result)

                    logger.info(f"Retrieved {len(enhanced_results)} knowledge items")
                    return enhanced_results
                else:
                    logger.error(f"Failed to retrieve knowledge: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []

    async def get_recommendations(self, agent_name: str, task_description: str,
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get knowledge-based recommendations for an agent"""
        try:
            # Search for similar past tasks and successful approaches
            query = f"agent:{agent_name} task:{task_description}"
            past_experiences = await self.retrieve_knowledge(query, "task_outcome", 10)

            recommendations = []

            # Analyze past experiences for patterns
            successful_approaches = [exp for exp in past_experiences if exp.get("metadata", {}).get("success") == True]

            if successful_approaches:
                # Extract common successful patterns
                approaches = [exp.get("metadata", {}).get("approach", "") for exp in successful_approaches]
                common_approaches = self._find_common_patterns(approaches)

                for approach in common_approaches[:3]:  # Top 3 recommendations
                    recommendations.append({
                        "type": "successful_approach",
                        "content": approach,
                        "confidence": 0.8,
                        "source": "past_experience"
                    })

            # Search for related knowledge
            related_knowledge = await self.retrieve_knowledge(task_description, None, 5)
            for knowledge in related_knowledge[:2]:  # Top 2 related items
                recommendations.append({
                    "type": "related_knowledge",
                    "content": knowledge.get("metadata", {}).get("text", "")[:200],
                    "confidence": knowledge.get("relevance_score", 0) * 0.5,
                    "source": "vector_store"
                })

            logger.info(f"Generated {len(recommendations)} recommendations for {agent_name}")
            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    async def store_conversation_history(self, task_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Store conversation history for a task"""
        try:
            self.conversation_store[task_id] = messages
            logger.info(f"Stored conversation history for task {task_id}: {len(messages)} messages")
            return True
        except Exception as e:
            logger.error(f"Error storing conversation history: {e}")
            return False

    async def retrieve_conversation_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a task"""
        try:
            return self.conversation_store.get(task_id, [])
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    async def update_conversation_history(self, task_id: str, new_messages: List[Dict[str, Any]]) -> bool:
        """Update conversation history for a task"""
        try:
            existing = self.conversation_store.get(task_id, [])
            existing.extend(new_messages)
            # Keep only last 50 messages to prevent memory issues
            if len(existing) > 50:
                existing = existing[-50:]
            self.conversation_store[task_id] = existing
            logger.info(f"Updated conversation history for task {task_id}: now {len(existing)} messages")
            return True
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")
            return False

    async def learn_from_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from completed task outcomes"""
        try:
            task_id = task_data.get("task_id", "unknown")
            success = task_data.get("success", False)
            agent_name = task_data.get("agent_name", "unknown")
            description = task_data.get("description", "")
            output = task_data.get("output", "")

            # Store task outcome as knowledge
            knowledge_content = f"Task: {description}\nAgent: {agent_name}\nSuccess: {success}\nOutput: {output}"
            metadata = {
                "task_id": task_id,
                "agent_name": agent_name,
                "success": success,
                "task_type": "outcome",
                "learnings": task_data.get("learnings", [])
            }

            knowledge_id = await self.store_knowledge(knowledge_content, metadata, "task_outcome")

            # Extract learnings for future recommendations
            learnings = {
                "knowledge_stored": knowledge_id,
                "patterns_identified": self._extract_patterns(task_data),
                "recommendations_generated": 0
            }

            logger.info(f"Learned from task {task_id}: stored knowledge {knowledge_id}")
            return learnings

        except Exception as e:
            logger.error(f"Error learning from task: {e}")
            return {"error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored knowledge"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_store_url}/metadata")
                if response.status_code == 200:
                    metadata = response.json()
                    total_items = len(metadata)

                    # Count by knowledge type
                    type_counts = {}
                    for item_id, item_metadata in metadata.items():
                        k_type = item_metadata.get("knowledge_type", "unknown")
                        type_counts[k_type] = type_counts.get(k_type, 0) + 1

                    return {
                        "total_knowledge_items": total_items,
                        "knowledge_types": type_counts,
                        "conversation_histories": len(self.conversation_store)
                    }
                else:
                    return {"error": "Failed to get vector store metadata"}
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {"error": str(e)}

    def _find_common_patterns(self, approaches: List[str]) -> List[str]:
        """Find common patterns in successful approaches"""
        if not approaches:
            return []

        # Simple pattern extraction - in a real implementation, this could use NLP
        patterns = []
        common_phrases = ["use", "implement", "create", "apply", "follow", "consider"]

        for approach in approaches:
            for phrase in common_phrases:
                if phrase in approach.lower():
                    patterns.append(approach.strip())
                    break

        return list(set(patterns))  # Remove duplicates

    def _extract_patterns(self, task_data: Dict[str, Any]) -> List[str]:
        """Extract patterns from task data for learning"""
        patterns = []

        if task_data.get("success"):
            patterns.append("successful_execution")
        else:
            patterns.append("failed_execution")

        agent_name = task_data.get("agent_name", "")
        if agent_name:
            patterns.append(f"agent_{agent_name}_used")

        description = task_data.get("description", "").lower()
        if "debug" in description:
            patterns.append("debugging_task")
        elif "implement" in description:
            patterns.append("implementation_task")
        elif "review" in description:
            patterns.append("review_task")

        return patterns