from abc import ABC, abstractmethod
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables at import time
except ImportError:
    pass  # dotenv not available, rely on system env
try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except Exception:
    # Provide a minimal shim around structlog backed by the stdlib logging
    import logging
    from types import SimpleNamespace

    logging.basicConfig(level=logging.INFO)

    def _noop(*args, **kwargs):
        return None

    stdlib_ns = SimpleNamespace(
        filter_by_level=_noop,
        add_logger_name=_noop,
        add_log_level=_noop,
        PositionalArgumentsFormatter=_noop,
        LoggerFactory=lambda *a, **k: None,
        BoundLogger=logging.Logger,
    )

    processors_ns = SimpleNamespace(
        TimeStamper=_noop,
        StackInfoRenderer=_noop,
        format_exc_info=_noop,
        UnicodeDecoder=_noop,
        JSONRenderer=_noop,
    )

    structlog = SimpleNamespace(
        configure=lambda *a, **k: None,
        stdlib=stdlib_ns,
        processors=processors_ns,
        get_logger=lambda name=None: logging.getLogger(name or __name__),
    )
    _STRUCTLOG_AVAILABLE = False

# Make OpenTelemetry optional at import time. Some deployment images may not
# include opentelemetry; in that case provide a no-op tracer so the code can
# run without the dependency while preserving tracing calls where available.
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    # Initialize OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    _OTEL_AVAILABLE = True
except Exception:
    # Fallback no-op tracer/spans and minimal Status/StatusCode placeholders
    _OTEL_AVAILABLE = False

    class StatusCode:
        UNSET = 0
        OK = 1
        ERROR = 2

    class Status:
        def __init__(self, code=StatusCode.UNSET, description=None):
            self.status_code = code
            self.description = description

    class _NoopSpan:
        def __init__(self, name: str = "noop"):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, *args, **kwargs):
            return None

        def add_event(self, *args, **kwargs):
            return None

        def set_status(self, *args, **kwargs):
            return None

    class _NoopTracer:
        def start_span(self, name: str):
            return _NoopSpan(name)

    tracer = _NoopTracer()

# Configure structured logging
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

@dataclass
class ModelResponse:
    text: str
    tokens: int
    tool_calls: List[Dict[str, Any]]
    structured_response: Dict[str, Any]
    confidence: float
    latency_ms: int
    error: Optional[str] = None

class BaseAdapter(ABC):
    def __init__(self, api_key_env: str, endpoint: str, max_retries: int = 3, backoff_factor: float = 2.0, role: str = "default", opa_url: str = "http://localhost:8181/v1/data/authz/allow", provider_id: str = None):
        # Initialize logger/tracer early so helper methods (like _get_secret)
        # can safely log during construction.
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.tracer = tracer

        self.api_key = self._get_secret(api_key_env)
        # Masked presence logging: log whether a key was found without printing
        # the value. This helps debugging startups without leaking secrets.
        if self.api_key:
            masked = (self.api_key[:4] + "..." + self.api_key[-4:]) if len(self.api_key) > 8 else "***"
            # Only log presence at debug level and mask actual key; provides
            # helpful startup telemetry while avoiding secrets in logs.
            self.logger.debug(f"Loaded API key for {api_key_env}: {masked}")
        else:
            # Don't raise here; some services instantiate adapters at startup
            # even if the API key isn't configured for that environment. Log
            # a warning and defer failure until the adapter is actually used.
            self.logger.warning(f"Secret {api_key_env} not found in environment")

        self.endpoint = endpoint
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.role = role
        self.opa_url = opa_url
        self.provider_id = provider_id or self._get_provider_id()

    def _get_secret(self, key_name: str) -> Optional[str]:
        # Try environment variable first
        secret = os.getenv(key_name)
        if secret:
            return secret
        # TODO: Implement integration with secret manager (e.g., AWS Secrets Manager, Vault)
        # For now, return None if not in env
        self.logger.warning(f"Secret {key_name} not found in environment")
        return None

    def _get_provider_id(self) -> str:
        """Get provider ID based on class name"""
        class_name = self.__class__.__name__.lower()
        if "nim" in class_name:
            return "nvidia_nim"
        elif "mistral" in class_name:
            return "mistral"
        elif "deepseek" in class_name:
            return "deepseek"
        elif "openrouter" in class_name:
            return "openrouter"
        elif "huggingface" in class_name:
            return "huggingface_local"
        elif "ollama" in class_name:
            return "ollama"
        elif "scaleway" in class_name:
            return "scaleway"
        elif "together" in class_name:
            return "together"
        else:
            return f"{class_name}_adapter"

    def _check_opa_policy(self, input_data: Dict[str, Any]) -> bool:
        try:
            payload = {
                "input": {
                    "role": self.role,
                    "adapter": self.__class__.__name__,
                    "request": input_data
                }
            }
            response = requests.post(self.opa_url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            allowed = result.get("result", False)
            self.logger.info("OPA policy check", allowed=allowed, role=self.role)
            return allowed
        except Exception as e:
            self.logger.warning("OPA policy check failed, allowing request", error=str(e), role=self.role)
            return True  # Fail open when OPA is not available

    def _call_api(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _call_api")

    def _normalize_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        raise NotImplementedError("Subclasses must implement _normalize_response")

    def _truncate_history(self, messages: List[Dict], max_messages: int = 20) -> List[Dict]:
        """Truncate message history to prevent token overflow."""
        if len(messages) <= max_messages:
            return messages
        # Keep the first message (system prompt if present) and the last max_messages-1
        truncated = []
        if messages and messages[0].get("role") == "system":
            truncated.append(messages[0])
            remaining = messages[1:][- (max_messages - 1):]
        else:
            remaining = messages[-max_messages:]
        truncated.extend(remaining)
        self.logger.info("Truncated message history", original_count=len(messages), truncated_count=len(truncated), role=self.role)
        return truncated

    def _extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships for knowledge graph."""
        # Simple entity extraction (can be enhanced with NLP models)
        entities = []
        relations = []

        # Basic pattern matching for code-related entities
        import re

        # Extract function names, classes, files
        functions = re.findall(r'def\s+(\w+)', text)
        classes = re.findall(r'class\s+(\w+)', text)
        files = re.findall(r'(\w+\.(py|js|ts|java|cpp))', text)

        entities.extend([{"type": "function", "name": f} for f in functions])
        entities.extend([{"type": "class", "name": c} for c in classes])
        entities.extend([{"type": "file", "name": f[0]} for f in files])

        # Extract relationships (simple co-occurrence)
        words = re.findall(r'\b\w+\b', text.lower())
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                if abs(i - j) <= 5:  # Close proximity
                    relations.append({
                        "from": word1,
                        "to": word2,
                        "type": "related",
                        "strength": 1.0 / (j - i)  # Closer = stronger
                    })

        return {"entities": entities, "relations": relations}

    def _store_in_vector_db(self, content: str, metadata: Dict[str, Any]):
        """Store content in vector database for RAG."""
        try:
            # Assuming vector_store has add_document method
            from vector_store import vector_store  # Import your vector store
            doc = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            vector_store.add_document(doc)
            self.logger.info("Stored in vector DB", content_length=len(content), role=self.role)
        except Exception as e:
            self.logger.warning("Failed to store in vector DB", error=str(e), role=self.role)

    def _retrieve_from_vector_db(self, query: str, limit: int = 3) -> List[str]:
        """Retrieve relevant context from vector database."""
        try:
            from vector_store import vector_store
            results = vector_store.search(query, limit=limit)
            contexts = [doc["content"] for doc in results]
            self.logger.info("Retrieved from vector DB", query=query, results_count=len(contexts), role=self.role)
            return contexts
        except Exception as e:
            self.logger.warning("Failed to retrieve from vector DB", error=str(e), role=self.role)
            return []

    def _update_knowledge_graph(self, entities: List[Dict], relations: List[Dict]):
        """Update in-memory knowledge graph."""
        try:
            # Simple in-memory graph (can be replaced with Neo4j/GraphDB)
            if not hasattr(self, '_knowledge_graph'):
                self._knowledge_graph = {"nodes": {}, "edges": []}

            # Add nodes
            for entity in entities:
                key = f"{entity['type']}:{entity['name']}"
                if key not in self._knowledge_graph["nodes"]:
                    self._knowledge_graph["nodes"][key] = entity

            # Add edges
            for relation in relations:
                edge = {
                    "from": relation["from"],
                    "to": relation["to"],
                    "type": relation["type"],
                    "strength": relation.get("strength", 1.0)
                }
                self._knowledge_graph["edges"].append(edge)

            self.logger.info("Updated knowledge graph", nodes_count=len(self._knowledge_graph["nodes"]), edges_count=len(self._knowledge_graph["edges"]), role=self.role)
        except Exception as e:
            self.logger.warning("Failed to update knowledge graph", error=str(e), role=self.role)

    def _query_knowledge_graph(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query knowledge graph for related information."""
        try:
            if not hasattr(self, '_knowledge_graph'):
                return {}

            # Simple graph traversal
            related = []
            visited = set()

            def traverse(current, depth):
                if depth > max_depth or current in visited:
                    return
                visited.add(current)

                for edge in self._knowledge_graph["edges"]:
                    if edge["from"] == current:
                        related.append({
                            "entity": edge["to"],
                            "relation": edge["type"],
                            "strength": edge["strength"]
                        })
                        traverse(edge["to"], depth + 1)

            traverse(entity, 0)
            self.logger.info("Queried knowledge graph", entity=entity, related_count=len(related), role=self.role)
            return {"related_entities": related}
        except Exception as e:
            self.logger.warning("Failed to query knowledge graph", error=str(e), role=self.role)
            return {}

    def call_model(self, messages: List[Dict], **kwargs) -> ModelResponse:
        if not self.api_key:
            raise ValueError("API key not configured for this adapter; set the appropriate environment variable")

        # Check for cached response first
        cached_response = self._check_prompt_cache(messages, **kwargs)
        if cached_response:
            self.logger.info("Using cached response", provider=self.__class__.__name__, role=self.role, cache_hit=True)
            return cached_response

        # Truncate history to prevent token overflow
        messages = self._truncate_history(messages)

        # Retrieve relevant context from vector DB for RAG
        current_query = ""
        if messages:
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                current_query = user_messages[-1]["content"]

        rag_context = ""
        if current_query:
            relevant_docs = self._retrieve_from_vector_db(current_query, limit=2)
            if relevant_docs:
                rag_context = "\n\nRelevant past context:\n" + "\n".join(relevant_docs)

        # Query knowledge graph for related entities
        kg_context = ""
        if current_query:
            kg_results = self._query_knowledge_graph(current_query)
            if kg_results.get("related_entities"):
                related = kg_results["related_entities"][:3]  # Top 3
                kg_context = "\n\nRelated concepts: " + ", ".join([f"{r['entity']} ({r['relation']})" for r in related])

        # Enhance messages with RAG and KG context
        enhanced_messages = messages.copy()
        if rag_context or kg_context:
            context_message = {
                "role": "system",
                "content": f"Use the following context to provide better responses:{rag_context}{kg_context}"
            }
            # Insert after existing system message or at beginning
            insert_pos = 0
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                insert_pos = 1
            enhanced_messages.insert(insert_pos, context_message)

        with self.tracer.start_span(f"{self.__class__.__name__}.call_model") as span:
            span.set_attribute("role", self.role)
            span.set_attribute("endpoint", self.endpoint)

            # OPA policy check
            policy_input = {"messages": enhanced_messages, "kwargs": kwargs}
            if not self._check_opa_policy(policy_input):
                span.set_status(Status(StatusCode.ERROR, "OPA policy denied"))
                self.logger.error("Request denied by OPA policy", role=self.role)
                return ModelResponse(
                    text="",
                    tokens=0,
                    tool_calls=[],
                    structured_response={},
                    confidence=0.0,
                    latency_ms=0,
                    error="Access denied by policy"
                )

            start_time = time.time()
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    self.logger.info("Calling API", attempt=attempt, role=self.role)
                    raw_response = self._call_api(enhanced_messages, **kwargs)
                    response = self._normalize_response(raw_response)
                    latency = int((time.time() - start_time) * 1000)
                    response.latency_ms = latency

                    span.set_attribute("latency_ms", latency)
                    span.set_attribute("tokens", response.tokens)
                    span.set_attribute("success", True)

                    # Emit metrics
                    self._emit_metrics(latency, response.tokens, True, self._estimate_cost(response.tokens))
                    self.logger.info("Model call successful", latency_ms=latency, tokens=response.tokens, role=self.role)

                    # Store response in prompt cache
                    try:
                        from .prompt_cache import prompt_cache_manager

                        model_name = kwargs.get("model", getattr(self, "default_model", "unknown"))
                        cache_response = {
                            "text": response.text,
                            "tool_calls": response.tool_calls,
                            "structured_response": response.structured_response,
                            "confidence": response.confidence,
                            "error": response.error
                        }

                        prompt_cache_manager.store_response(
                            provider=self.__class__.__name__.lower().replace("adapter", ""),
                            model=model_name,
                            role=self.role,
                            messages=enhanced_messages,
                            response=cache_response,
                            tokens_used=response.tokens,
                            latency_ms=latency,
                            cost_estimate=self._estimate_cost(response.tokens)
                        )
                        self.logger.info("Stored response in prompt cache", provider=self.__class__.__name__, role=self.role)
                    except Exception as cache_error:
                        self.logger.warning("Failed to store response in cache", error=str(cache_error), role=self.role)

                    # Store conversation in vector DB for future RAG
                    if current_query and response.text:
                        conversation_content = f"Query: {current_query}\nResponse: {response.text}"
                        metadata = {
                            "role": self.role,
                            "provider": self.__class__.__name__,
                            "tokens": response.tokens,
                            "latency_ms": latency
                        }
                        self._store_in_vector_db(conversation_content, metadata)

                        # Extract and store entities/relations in knowledge graph
                        combined_text = current_query + " " + response.text
                        kg_data = self._extract_entities_and_relations(combined_text)
                        self._update_knowledge_graph(kg_data["entities"], kg_data["relations"])

                    return response
                except Exception as e:
                    attempt += 1
                    span.add_event(f"Attempt {attempt} failed", {"error": str(e)})
                    if attempt > self.max_retries:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        self._emit_metrics(int((time.time() - start_time) * 1000), 0, False, 0)
                        self.logger.error("All retries exhausted", error=str(e), attempts=attempt, role=self.role)
                        raise e
                    wait_time = self.backoff_factor ** attempt
                    self.logger.warning("API call failed, retrying", attempt=attempt, error=str(e), wait_time=wait_time, role=self.role)
                    time.sleep(wait_time)

    def _emit_metrics(self, latency: int, tokens: int, success: bool, cost: float):
        self.logger.info("Metrics emitted", latency_ms=latency, tokens=tokens, success=success, cost=cost, role=self.role)

        # Update provider metrics in database with retry logic
        try:
            import asyncio
            from sqlalchemy import select, update, func
            from orchestrator.database import async_session, ProviderMetrics, execute_with_retry

            async def update_metrics():
                async with async_session() as session:
                    # Get current metrics
                    result = await session.execute(
                        select(ProviderMetrics).where(ProviderMetrics.provider_id == self.provider_id)
                    )
                    metrics = result.scalar_one_or_none()

                    if metrics:
                        # Update metrics
                        metrics.total_requests += 1
                        metrics.tokens_used += tokens
                        metrics.cost_estimate += cost
                        metrics.last_used = func.now()

                        if success:
                            # Recalculate success rate
                            current_successes = int(metrics.success_rate * (metrics.total_requests - 1) / 100)
                            new_successes = current_successes + 1
                            metrics.success_rate = (new_successes / metrics.total_requests) * 100
                        else:
                            # Failed request - recalculate success rate
                            current_successes = int(metrics.success_rate * (metrics.total_requests - 1) / 100)
                            metrics.success_rate = (current_successes / metrics.total_requests) * 100

                        # Update latency (simple average)
                        if metrics.latency == 0:
                            metrics.latency = latency
                        else:
                            metrics.latency = (metrics.latency + latency) / 2

                        await session.commit()
                        self.logger.info(f"Updated metrics for provider {self.provider_id}: requests={metrics.total_requests}, success_rate={metrics.success_rate:.1f}%")

            # Run the async update with retry logic
            async def update_with_retry():
                await execute_with_retry(update_metrics)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    asyncio.create_task(update_with_retry())
                else:
                    loop.run_until_complete(update_with_retry())
            except RuntimeError:
                # No event loop, create a new one
                asyncio.run(update_with_retry())

        except Exception as e:
            self.logger.warning("Failed to update provider metrics in database", error=str(e), provider_id=self.provider_id)

    def _estimate_cost(self, tokens: int) -> float:
        # Placeholder, implement per provider
        return tokens * 0.0001  # example

    def _check_prompt_cache(self, messages: List[Dict], **kwargs) -> Optional[ModelResponse]:
        """Check if we have a cached response for this prompt."""
        try:
            from .prompt_cache import prompt_cache_manager

            # Get model name from kwargs or use default
            model = kwargs.get("model", getattr(self, "default_model", "unknown"))
            provider = self.__class__.__name__.lower().replace("adapter", "")

            # Check cache
            cached = prompt_cache_manager.get_cached_response(
                provider=provider,
                model=model,
                role=self.role,
                messages=messages
            )

            if cached:
                self.logger.info("Using cached response", provider=provider, role=self.role, cache_hit=True, tokens_used=cached.get('tokens_used', 0))
                # Convert cached response back to ModelResponse format
                response_data = cached["response"]
                return ModelResponse(
                    text=response_data.get("text", ""),
                    tokens=cached["tokens_used"],
                    tool_calls=response_data.get("tool_calls", []),
                    structured_response=response_data.get("structured_response", {}),
                    confidence=response_data.get("confidence", 1.0),
                    latency_ms=cached["latency_ms"],
                    error=response_data.get("error")
                )

        except Exception as e:
            self.logger.warning("Failed to check prompt cache", error=str(e), role=self.role)

        return None