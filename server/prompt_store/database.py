from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship
from sqlalchemy import String, Text, JSON, DateTime, Float, Integer, ForeignKey, Boolean
from sqlalchemy.sql import func
import uuid
from typing import Optional
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@postgres:5432/codeagent")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    current_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("prompt_versions.id"), nullable=True)

    # Relationships
    versions: Mapped[list["PromptVersion"]] = relationship("PromptVersion", back_populates="prompt", cascade="all, delete-orphan")
    current_version: Mapped[Optional["PromptVersion"]] = relationship("PromptVersion", foreign_keys=[current_version_id])

class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id: Mapped[str] = mapped_column(String(36), ForeignKey("prompts.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # For small prompts
    minio_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # For large prompts stored in MinIO
    prompt_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Tags, model info, etc.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prompt: Mapped["Prompt"] = relationship("Prompt", back_populates="versions")
    metrics: Mapped[list["PromptMetrics"]] = relationship("PromptMetrics", back_populates="version", cascade="all, delete-orphan")

class PromptMetrics(Base):
    __tablename__ = "prompt_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("prompt_versions.id"), nullable=False)
    execution_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    last_used_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    version: Mapped["PromptVersion"] = relationship("PromptVersion", back_populates="metrics")

    @property
    def success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def avg_latency_ms(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.total_latency_ms / self.execution_count

    @property
    def avg_cost(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.total_cost / self.execution_count

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)