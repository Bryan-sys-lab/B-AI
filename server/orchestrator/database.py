from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Text, JSON, DateTime, Float, Integer, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.exc import OperationalError
import uuid
from typing import Optional
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///codeagent.db")

# Configure SQLite with better concurrency settings
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,  # Allow multi-threaded access
        "timeout": 30.0,  # Connection timeout
        "isolation_level": None,  # Autocommit mode for better concurrency
    }

engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Disable echo in production
    connect_args=connect_args,
    pool_pre_ping=True,  # Check connection health
    pool_size=10,  # Connection pool size
    max_overflow=20,  # Max overflow connections
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def execute_with_retry(operation, max_retries=3, delay=0.1):
    """Execute database operation with retry logic for SQLite locking issues"""
    for attempt in range(max_retries):
        try:
            return await operation()
        except OperationalError as e:
            if "database is locked" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                    raise
            else:
                # Re-raise non-locking errors immediately
                raise
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="active")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id"), nullable=True)
    user_id: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    plan: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Subtask(Base):
    __tablename__ = "subtasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"))
    agent_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    assigned_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"))
    subtask_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("subtasks.id"), nullable=True)
    user_id: Mapped[str] = mapped_column(String(255))
    rating: Mapped[int] = mapped_column(Integer, default=5)  # 1-10 scale
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    improvement_suggestions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(50))  # 'primary' or 'fallback'
    purpose: Mapped[str] = mapped_column(Text)
    models: Mapped[dict] = mapped_column(JSON)  # list of available models
    status: Mapped[str] = mapped_column(String(50), default="standby")  # 'active', 'standby', 'error', 'inactive'
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # provider-specific config
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ProviderMetrics(Base):
    __tablename__ = "provider_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider_id: Mapped[str] = mapped_column(String(36), ForeignKey("providers.id"))
    latency: Mapped[float] = mapped_column(Float, default=0.0)
    success_rate: Mapped[float] = mapped_column(Float, default=100.0)
    total_requests: Mapped[int] = mapped_column(Float, default=0)
    active_requests: Mapped[int] = mapped_column(Float, default=0)
    cost_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_used: Mapped[int] = mapped_column(Float, default=0)
    last_used: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(50))  # 'master_agent', 'fix_implementation_agent', etc.
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="idle")  # 'idle', 'busy', 'error'
    health: Mapped[str] = mapped_column(String(50), default="healthy")  # 'healthy', 'warning', 'error'
    current_task: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class AgentMetrics(Base):
    __tablename__ = "agent_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"))
    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    average_response_time: Mapped[float] = mapped_column(Float, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_activity: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text)
    branch: Mapped[str] = mapped_column(String(255), default="main")
    status: Mapped[str] = mapped_column(String(50), default="synced")  # 'synced', 'behind', 'ahead', 'conflict'
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    size: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    commits: Mapped[int] = mapped_column(Integer, default=0)
    contributors: Mapped[int] = mapped_column(Integer, default=0)
    last_sync: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RepositoryFile(Base):
    __tablename__ = "repository_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    repository_id: Mapped[str] = mapped_column(String(36), ForeignKey("repositories.id"))
    path: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(50))  # 'file' or 'folder'
    size: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    last_modified: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    content: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # For file content
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class SecurityPolicy(Base):
    __tablename__ = "security_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100))  # 'code', 'infrastructure', 'data', 'access'
    severity: Mapped[str] = mapped_column(String(50))  # 'critical', 'high', 'medium', 'low'
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SecurityScan(Base):
    __tablename__ = "security_scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    target_type: Mapped[str] = mapped_column(String(50))  # 'repository', 'code', 'infrastructure'
    target_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(50), default="pending")  # 'pending', 'running', 'completed', 'failed'
    findings: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class ObservabilityMetric(Base):
    __tablename__ = "observability_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(100))  # 'performance', 'usage', 'error', 'system'
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(50))
    tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(100))  # 'coding', 'analysis', 'generation', 'review'
    content: Mapped[str] = mapped_column(Text)
    variables: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class IntelligenceAnalysis(Base):
    __tablename__ = "intelligence_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    target_type: Mapped[str] = mapped_column(String(50))  # 'code', 'repository', 'task'
    target_id: Mapped[str] = mapped_column(String(36))
    analysis_type: Mapped[str] = mapped_column(String(100))  # 'complexity', 'quality', 'patterns', 'suggestions'
    result: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Integration(Base):
    __tablename__ = "integrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(100))  # 'github', 'gitlab', 'slack', 'discord', 'webhook'
    description: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="inactive")  # 'active', 'inactive', 'error'
    last_sync: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PromptCache(Base):
    __tablename__ = "prompt_cache"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA256 hash of prompt
    provider: Mapped[str] = mapped_column(String(100))  # 'nim', 'deepseek', 'mistral', etc.
    model: Mapped[str] = mapped_column(String(100))  # specific model name
    role: Mapped[str] = mapped_column(String(50))  # 'default', 'builders', etc.
    prompt_hash: Mapped[str] = mapped_column(String(64))  # hash of full prompt content
    prompt_content: Mapped[dict] = mapped_column(JSON)  # full prompt messages
    response: Mapped[dict] = mapped_column(JSON)  # full Aetherium response
    tokens_used: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_estimate: Mapped[float] = mapped_column(Float)
    hit_count: Mapped[int] = mapped_column(Integer, default=1)  # how many times this cache entry was used
    last_used: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)  # TTL expiration
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)