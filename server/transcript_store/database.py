from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, Text, JSON, DateTime, Integer, Enum
from sqlalchemy.sql import func
import uuid
from typing import Optional
import os
import enum

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@postgres:5432/codeagent")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class TranscriptType(enum.Enum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    ERROR = "error"

class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type: Mapped[TranscriptType] = mapped_column(Enum(TranscriptType), nullable=False, index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    transcript_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Additional info like model, duration, etc.
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # For small transcripts
    minio_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # For large transcripts stored in MinIO
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)