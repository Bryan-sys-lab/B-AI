from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Text, JSON, DateTime, Float, ForeignKey, Integer
from sqlalchemy.sql import func
import uuid
from typing import Optional
import os
from orchestrator.database import Base, Task, Subtask

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///codeagent.db")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"))
    subtask_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("subtasks.id"), nullable=True)
    agent_name: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"))
    subtask_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("subtasks.id"), nullable=True)
    agent_name: Mapped[str] = mapped_column(String(255))
    metric_name: Mapped[str] = mapped_column(String(255))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class ImprovementTask(Base):
    __tablename__ = "improvement_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    description: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=1)  # 1-5 scale
    category: Mapped[str] = mapped_column(String(255))  # e.g., prompt_optimization, model_fine_tuning
    status: Mapped[str] = mapped_column(String(50), default="pending")
    generated_from: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # source analysis
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class FineTuningCandidate(Base):
    __tablename__ = "fine_tuning_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), ForeignKey("tasks.id"))
    transcript_id: Mapped[str] = mapped_column(String(36), ForeignKey("transcripts.id"))
    input_text: Mapped[str] = mapped_column(Text)
    expected_output: Mapped[str] = mapped_column(Text)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)