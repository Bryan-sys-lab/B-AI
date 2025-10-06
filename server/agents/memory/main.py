import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
import httpx

from providers.nim_adapter import NIMAdapter
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

app = FastAPI(title="Memory/Knowledge Graph Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryRequest(BaseModel):
    operation: str  # 'store', 'retrieve', 'search', 'link', 'graph'
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class KnowledgeGraph:
    """Manages relationships between agents, tasks, and knowledge"""

    def __init__(self):
        self.relationships = {}  # entity_id -> {related_entities}
        self.entity_metadata = {}  # entity_id -> metadata
        self.knowledge_base = {}  # key -> knowledge_item

    def add_entity(self, entity_id: str, entity_type: str, metadata: dict = None):
        """Add an entity to the knowledge graph"""
        if entity_id not in self.entity_metadata:
            self.entity_metadata[entity_id] = {
                'type': entity_type,
                'metadata': metadata or {},
                'created_at': asyncio.get_event_loop().time(),
                'relationships': []
            }
        return entity_id

    def add_relationship(self, entity1: str, entity2: str, relationship_type: str, metadata: dict = None):
        """Add a relationship between two entities"""
        if entity1 not in self.relationships:
            self.relationships[entity1] = {}
        if entity2 not in self.relationships:
            self.relationships[entity2] = {}

        self.relationships[entity1][entity2] = {
            'type': relationship_type,
            'metadata': metadata or {},
            'created_at': asyncio.get_event_loop().time()
        }

        # Bidirectional relationship
        self.relationships[entity2][entity1] = {
            'type': relationship_type,
            'metadata': metadata or {},
            'created_at': asyncio.get_event_loop().time()
        }

    def get_related_entities(self, entity_id: str, relationship_type: str = None):
        """Get entities related to the given entity"""
        if entity_id not in self.relationships:
            return []

        related = []
        for related_id, rel_data in self.relationships[entity_id].items():
            if relationship_type is None or rel_data['type'] == relationship_type:
                related.append({
                    'entity_id': related_id,
                    'relationship': rel_data
                })

        return related

    def store_knowledge(self, key: str, knowledge: dict):
        """Store knowledge item"""
        self.knowledge_base[key] = {
            'data': knowledge,
            'stored_at': asyncio.get_event_loop().time()
        }

    def retrieve_knowledge(self, key: str):
        """Retrieve knowledge item"""
        return self.knowledge_base.get(key)

    def search_knowledge(self, query: str):
        """Search knowledge base for relevant items"""
        results = []
        for key, item in self.knowledge_base.items():
            if query.lower() in key.lower() or query.lower() in json.dumps(item['data']).lower():
                results.append({
                    'key': key,
                    'data': item['data'],
                    'relevance_score': 1.0  # Simple matching for now
                })
        return results

# Global knowledge graph instance
knowledge_graph = KnowledgeGraph()

async def call_vector_store(operation: str, data: dict):
    """Call the vector store service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if operation == 'store':
                response = await client.post("http://localhost:8019/add_text", json=data)
            elif operation == 'search':
                response = await client.post("http://localhost:8019/search_text", json=data)
            elif operation == 'retrieve':
                # Vector store doesn't have direct retrieve, use search
                response = await client.post("http://localhost:8019/search_text", json=data)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Vector store error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        logger.error(f"Vector store call failed: {e}")
        return None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/execute")
async def execute_memory_operation(request: MemoryRequest):
    """Execute memory and knowledge graph operations"""
    try:
        operation = request.operation
        data = request.data
        context = request.context or {}

        logger.info(f"Memory operation: {operation} with data keys: {list(data.keys())}")

        if operation == 'store_conversation':
            # Store conversation history
            task_id = data.get('task_id')
            conversation = data.get('conversation', [])

            if task_id and conversation:
                # Store in knowledge graph
                entity_id = f"task_{task_id}"
                knowledge_graph.add_entity(entity_id, 'task', {'conversation_length': len(conversation)})

                # Store conversation as knowledge
                key = f"conversation_{task_id}"
                knowledge_graph.store_knowledge(key, {
                    'task_id': task_id,
                    'conversation': conversation,
                    'stored_at': asyncio.get_event_loop().time()
                })

                # Also store in vector store for semantic search
                conversation_text = "\n".join([
                    f"{msg.get('type', 'unknown')}: {msg.get('content', '')}"
                    for msg in conversation[-20:]  # Last 20 messages
                ])

                vector_data = {
                    'text': conversation_text,
                    'metadata': {
                        'type': 'conversation',
                        'task_id': task_id,
                        'message_count': len(conversation),
                        'last_message': conversation[-1].get('content', '')[:200] if conversation else ''
                    }
                }

                vector_result = await call_vector_store('store', vector_data)

                return MemoryResponse(
                    success=True,
                    result={
                        'stored': True,
                        'entity_id': entity_id,
                        'vector_stored': vector_result is not None
                    }
                )

        elif operation == 'retrieve_conversation':
            # Retrieve conversation history
            task_id = data.get('task_id')
            if task_id:
                key = f"conversation_{task_id}"
                knowledge = knowledge_graph.retrieve_knowledge(key)
                if knowledge:
                    return MemoryResponse(success=True, result=knowledge)

        elif operation == 'store_agent_output':
            # Store agent output and link to task
            task_id = data.get('task_id')
            agent_name = data.get('agent_name')
            output = data.get('output', {})

            if task_id and agent_name:
                # Add entities
                task_entity = f"task_{task_id}"
                agent_entity = f"agent_{agent_name}"

                knowledge_graph.add_entity(task_entity, 'task')
                knowledge_graph.add_entity(agent_entity, 'agent', {'name': agent_name})

                # Add relationship
                knowledge_graph.add_relationship(
                    task_entity, agent_entity, 'executed_by',
                    {'output_summary': str(output)[:500]}
                )

                # Store output as knowledge
                key = f"output_{task_id}_{agent_name}"
                knowledge_graph.store_knowledge(key, {
                    'task_id': task_id,
                    'agent_name': agent_name,
                    'output': output,
                    'stored_at': asyncio.get_event_loop().time()
                })

                # Store in vector store
                output_text = json.dumps(output)
                vector_data = {
                    'text': output_text[:5000],  # Limit size
                    'metadata': {
                        'type': 'agent_output',
                        'task_id': task_id,
                        'agent_name': agent_name,
                        'output_size': len(output_text)
                    }
                }

                vector_result = await call_vector_store('store', vector_data)

                return MemoryResponse(
                    success=True,
                    result={
                        'stored': True,
                        'task_entity': task_entity,
                        'agent_entity': agent_entity,
                        'vector_stored': vector_result is not None
                    }
                )

        elif operation == 'search_knowledge':
            # Search knowledge base
            query = data.get('query', '')
            if query:
                # Search local knowledge graph
                local_results = knowledge_graph.search_knowledge(query)

                # Search vector store
                vector_results = await call_vector_store('search', {'query': query, 'k': 5})

                return MemoryResponse(
                    success=True,
                    result={
                        'local_results': local_results,
                        'vector_results': vector_results.get('results', []) if vector_results else []
                    }
                )

        elif operation == 'get_agent_relationships':
            # Get relationships for an agent
            agent_name = data.get('agent_name')
            if agent_name:
                entity_id = f"agent_{agent_name}"
                relationships = knowledge_graph.get_related_entities(entity_id)
                return MemoryResponse(success=True, result={'relationships': relationships})

        elif operation == 'get_task_context':
            # Get full context for a task
            task_id = data.get('task_id')
            if task_id:
                task_entity = f"task_{task_id}"

                # Get conversation
                conv_key = f"conversation_{task_id}"
                conversation = knowledge_graph.retrieve_knowledge(conv_key)

                # Get agent outputs
                related_entities = knowledge_graph.get_related_entities(task_entity, 'executed_by')
                agent_outputs = []
                for rel in related_entities:
                    agent_entity = rel['entity_id']
                    if agent_entity.startswith('agent_'):
                        agent_name = agent_entity.replace('agent_', '')
                        output_key = f"output_{task_id}_{agent_name}"
                        output = knowledge_graph.retrieve_knowledge(output_key)
                        if output:
                            agent_outputs.append(output)

                return MemoryResponse(
                    success=True,
                    result={
                        'conversation': conversation,
                        'agent_outputs': agent_outputs,
                        'relationships': related_entities
                    }
                )

        return MemoryResponse(success=False, error=f"Unknown operation: {operation}")

    except Exception as e:
        logger.error(f"Memory operation failed: {e}")
        return MemoryResponse(success=False, error=str(e))

@app.get("/graph/stats")
async def get_graph_stats():
    """Get knowledge graph statistics"""
    try:
        stats = {
            'entities': len(knowledge_graph.entity_metadata),
            'relationships': sum(len(rels) for rels in knowledge_graph.relationships.values()) // 2,  # Divide by 2 for bidirectional
            'knowledge_items': len(knowledge_graph.knowledge_base),
            'entity_types': {}
        }

        for entity_id, metadata in knowledge_graph.entity_metadata.items():
            entity_type = metadata.get('type', 'unknown')
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1

        return stats

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        return {"error": str(e)}

@app.get("/about")
def about(detail: Optional[str] = "short"):
    """Return agent about information"""
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}

    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {
        "level": level,
        "response": resp,
        "specialization": "Memory and Knowledge Graph Management"
    }