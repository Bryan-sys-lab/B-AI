import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
import json
import httpx

from .knowledge_manager import KnowledgeManager
from .relationship_tracker import RelationshipTracker
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES, get_agent_prompt

app = FastAPI(title="Knowledge Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
knowledge_manager = KnowledgeManager()
relationship_tracker = RelationshipTracker()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting knowledge agent startup event")
    await knowledge_manager.initialize()
    await relationship_tracker.initialize()
    logger.info("Knowledge agent startup event completed")

@app.get("/health")
def health():
    return {"status": "ok"}

# Request/Response Models
class StoreKnowledgeRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
    knowledge_type: str = "general"  # general, task_outcome, agent_relationship, conversation

class RetrieveKnowledgeRequest(BaseModel):
    query: str
    knowledge_type: Optional[str] = None
    limit: int = 5
    context: Optional[Dict[str, Any]] = None

class StoreRelationshipRequest(BaseModel):
    source_agent: str
    target_agent: str
    relationship_type: str  # collaborates_with, depends_on, conflicts_with, etc.
    strength: float = 1.0
    context: Optional[Dict[str, Any]] = None

class GetRecommendationsRequest(BaseModel):
    agent_name: str
    task_description: str
    context: Optional[Dict[str, Any]] = None

class ConversationHistoryRequest(BaseModel):
    task_id: str
    messages: List[Dict[str, Any]]
    operation: str = "store"  # store, retrieve, update

@app.post("/store_knowledge")
async def store_knowledge(request: StoreKnowledgeRequest):
    """Store knowledge in the vector store with metadata"""
    try:
        knowledge_id = await knowledge_manager.store_knowledge(
            request.content,
            request.metadata,
            request.knowledge_type
        )
        logger.info(f"Stored knowledge: {knowledge_id} of type {request.knowledge_type}")
        return {"knowledge_id": knowledge_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Error storing knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve_knowledge")
async def retrieve_knowledge(request: RetrieveKnowledgeRequest):
    """Retrieve relevant knowledge based on query"""
    try:
        results = await knowledge_manager.retrieve_knowledge(
            request.query,
            request.knowledge_type,
            request.limit,
            request.context
        )
        logger.info(f"Retrieved {len(results)} knowledge items for query: {request.query[:50]}...")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error retrieving knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store_relationship")
async def store_relationship(request: StoreRelationshipRequest):
    """Store relationship between agents"""
    try:
        relationship_id = await relationship_tracker.store_relationship(
            request.source_agent,
            request.target_agent,
            request.relationship_type,
            request.strength,
            request.context
        )
        logger.info(f"Stored relationship: {request.source_agent} -> {request.target_agent} ({request.relationship_type})")
        return {"relationship_id": relationship_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Error storing relationship: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_recommendations")
async def get_recommendations(request: GetRecommendationsRequest):
    """Get knowledge-based recommendations for an agent"""
    try:
        recommendations = await knowledge_manager.get_recommendations(
            request.agent_name,
            request.task_description,
            request.context
        )
        logger.info(f"Generated {len(recommendations)} recommendations for {request.agent_name}")
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation_history")
async def manage_conversation_history(request: ConversationHistoryRequest):
    """Manage conversation history for tasks"""
    try:
        if request.operation == "store":
            result = await knowledge_manager.store_conversation_history(
                request.task_id,
                request.messages
            )
        elif request.operation == "retrieve":
            result = await knowledge_manager.retrieve_conversation_history(request.task_id)
        elif request.operation == "update":
            result = await knowledge_manager.update_conversation_history(
                request.task_id,
                request.messages
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")

        logger.info(f"Conversation history {request.operation} for task {request.task_id}: {len(request.messages)} messages")
        return {"result": result, "status": "success"}
    except Exception as e:
        logger.error(f"Error managing conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge_stats")
async def get_knowledge_stats():
    """Get statistics about stored knowledge"""
    try:
        stats = await knowledge_manager.get_stats()
        relationship_stats = await relationship_tracker.get_stats()
        return {
            "knowledge_stats": stats,
            "relationship_stats": relationship_stats
        }
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn_from_task")
async def learn_from_task(task_data: Dict[str, Any]):
    """Learn from completed task outcomes"""
    try:
        learning_results = await knowledge_manager.learn_from_task(task_data)
        logger.info(f"Learned from task: {task_data.get('task_id', 'unknown')}")
        return {"learning_results": learning_results, "status": "learned"}
    except Exception as e:
        logger.error(f"Error learning from task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_task(request: dict):
    """Main execution endpoint for the knowledge agent"""
    try:
        logger.info(f"Executing knowledge task: {request}")
        desc = (request.get("description", "") or "").strip()

        # Handle different types of knowledge requests
        if "store" in desc.lower() and "knowledge" in desc.lower():
            # Extract content and metadata from description
            content = request.get("content", desc)
            metadata = request.get("metadata", {"source": "agent_request"})
            knowledge_type = request.get("knowledge_type", "general")

            knowledge_id = await knowledge_manager.store_knowledge(content, metadata, knowledge_type)
            return {"result": f"Stored knowledge with ID: {knowledge_id}", "success": True}

        elif "retrieve" in desc.lower() or "search" in desc.lower():
            query = request.get("query", desc)
            knowledge_type = request.get("knowledge_type")
            limit = request.get("limit", 5)

            results = await knowledge_manager.retrieve_knowledge(query, knowledge_type, limit)
            return {"result": f"Found {len(results)} relevant knowledge items", "knowledge": results, "success": True}

        elif "relationship" in desc.lower():
            # Handle relationship queries
            relationships = await relationship_tracker.get_relationships()
            return {"result": f"Found {len(relationships)} agent relationships", "relationships": relationships, "success": True}

        elif "recommend" in desc.lower():
            agent_name = request.get("agent_name", "unknown")
            task_desc = request.get("task_description", desc)

            recommendations = await knowledge_manager.get_recommendations(agent_name, task_desc)
            return {"result": f"Generated {len(recommendations)} recommendations", "recommendations": recommendations, "success": True}

        else:
            # Default knowledge assistance
            return {
                "result": "I am the Knowledge Agent. I can help you store, retrieve, and manage knowledge across the system. Available operations: store_knowledge, retrieve_knowledge, manage_relationships, get_recommendations.",
                "success": True
            }

    except Exception as e:
        logger.error(f"Error executing knowledge task: {str(e)}")
        return {"error": str(e)}

@app.get("/execute/stream")
async def execute_task_stream(
    description: str,
    conversation_history: Optional[str] = None
):
    """Streaming execution endpoint"""
    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'message': '🧠 Processing knowledge request...', 'step': 1, 'total': 3})}\n\n"
            await asyncio.sleep(0.3)

            # Parse request
            request = {"description": description}
            if conversation_history:
                try:
                    request["conversation_history"] = json.loads(conversation_history)
                except:
                    pass

            yield f"data: {json.dumps({'type': 'progress', 'message': '🔍 Analyzing knowledge needs...', 'step': 2, 'total': 3})}\n\n"
            await asyncio.sleep(0.3)

            # Execute task
            result = await execute_task(request)

            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Knowledge operation completed!', 'step': 3, 'total': 3})}\n\n"

            # Send final result
            response_data = {
                "result": result.get("result", "Knowledge operation completed"),
                "success": result.get("success", True),
                "data": result
            }

            yield f"data: {json.dumps({'type': 'complete', 'result': response_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return a canned "about" response at three levels: short, medium, detailed."""
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    responses = {
        "short": "I am the Knowledge Agent, managing memory and relationships across the system.",
        "medium": "I handle long-term memory storage, agent relationship tracking, and provide knowledge recommendations to improve system performance.",
        "detailed": "As the Knowledge Agent, I maintain the system's knowledge graph by storing task outcomes, tracking agent relationships, managing conversation history, and providing intelligent recommendations. I use vector search for semantic knowledge retrieval and learn from system interactions to improve future performance."
    }

    resp = responses.get(level, responses["short"])
    return {
        "level": level,
        "response": resp,
        "response": resp,
    }