from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, and_, or_, func
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

from transcript_store.database import async_session, init_db, Transcript, TranscriptType
from transcript_store.minio_client import minio_client
from transcript_store.models import TranscriptCreate, TranscriptResponse, TranscriptSearch, DatasetExport, RetentionPolicy

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Transcript Store Service", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CONTENT_SIZE_THRESHOLD = 10240  # 10KB - store in MinIO if larger

# Helper functions
def should_store_in_minio(content: str) -> bool:
    return len(content.encode('utf-8')) > CONTENT_SIZE_THRESHOLD

async def get_transcript_content(transcript: Transcript) -> str:
    if transcript.minio_key:
        return minio_client.download_transcript(transcript.minio_key)
    return transcript.content or ""

async def store_transcript_content(transcript_id: str, content: str) -> tuple[str, Optional[str]]:
    if should_store_in_minio(content):
        minio_key = minio_client.upload_transcript(content, transcript_id)
        return "", minio_key
    return content, None

# API Endpoints

@app.post("/transcripts", response_model=TranscriptResponse)
async def create_transcript(transcript_data: TranscriptCreate):
    """Store a new transcript (append-only)"""
    async with async_session() as session:
        # Generate ID first
        import uuid
        transcript_id = str(uuid.uuid4())

        # Store content
        content, minio_key = await store_transcript_content(transcript_id, transcript_data.content)

        # Create transcript
        transcript = Transcript(
            id=transcript_id,
            type=transcript_data.type,
            agent_id=transcript_data.agent_id,
            task_id=transcript_data.task_id,
            session_id=transcript_data.session_id,
            timestamp=transcript_data.timestamp or datetime.now(timezone.utc),
            transcript_metadata=transcript_data.metadata,
            content=content,
            minio_key=minio_key
        )
        session.add(transcript)
        await session.commit()

        return TranscriptResponse(
            id=transcript.id,
            type=transcript.type,
            agent_id=transcript.agent_id,
            task_id=transcript.task_id,
            session_id=transcript.session_id,
            timestamp=transcript.timestamp,
            metadata=transcript.transcript_metadata,
            content=transcript_data.content,  # Return original content
            created_at=transcript.created_at
        )

@app.get("/transcripts", response_model=List[TranscriptResponse])
async def search_transcripts(
    type: Optional[str] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
):
    """Search and filter transcripts"""
    async with async_session() as session:
        query = select(Transcript)

        # Build filters
        conditions = []
        if type:
            conditions.append(Transcript.type == TranscriptType(type))
        if agent_id:
            conditions.append(Transcript.agent_id == agent_id)
        if task_id:
            conditions.append(Transcript.task_id == task_id)
        if session_id:
            conditions.append(Transcript.session_id == session_id)
        if start_time:
            conditions.append(Transcript.timestamp >= start_time)
        if end_time:
            conditions.append(Transcript.timestamp <= end_time)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Transcript.timestamp.desc()).offset(offset).limit(limit)

        result = await session.execute(query)
        transcripts = result.scalars().all()

        responses = []
        for transcript in transcripts:
            responses.append(TranscriptResponse(
                id=transcript.id,
                type=transcript.type,
                agent_id=transcript.agent_id,
                task_id=transcript.task_id,
                session_id=transcript.session_id,
                timestamp=transcript.timestamp,
                metadata=transcript.transcript_metadata,
                content=await get_transcript_content(transcript),
                created_at=transcript.created_at
            ))

        return responses

@app.get("/transcripts/{transcript_id}", response_model=TranscriptResponse)
async def get_transcript(transcript_id: str):
    """Get a specific transcript"""
    async with async_session() as session:
        result = await session.execute(
            select(Transcript).where(Transcript.id == transcript_id)
        )
        transcript = result.scalar_one_or_none()
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")

        return TranscriptResponse(
            id=transcript.id,
            type=transcript.type,
            agent_id=transcript.agent_id,
            task_id=transcript.task_id,
            session_id=transcript.session_id,
            timestamp=transcript.timestamp,
            metadata=transcript.transcript_metadata,
            content=await get_transcript_content(transcript),
            created_at=transcript.created_at
        )

@app.post("/export")
async def export_dataset(export_params: DatasetExport):
    """Export transcripts for dataset generation (self-improvement)"""
    async with async_session() as session:
        query = select(Transcript)

        # Build filters similar to search
        conditions = []
        if export_params.type:
            conditions.append(Transcript.type == export_params.type)
        if export_params.agent_id:
            conditions.append(Transcript.agent_id == export_params.agent_id)
        if export_params.task_id:
            conditions.append(Transcript.task_id == export_params.task_id)
        if export_params.start_time:
            conditions.append(Transcript.timestamp >= export_params.start_time)
        if export_params.end_time:
            conditions.append(Transcript.timestamp <= export_params.end_time)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Transcript.timestamp)

        result = await session.execute(query)
        transcripts = result.scalars().all()

        # Format for dataset (JSONL for training)
        dataset_lines = []
        for transcript in transcripts:
            content = await get_transcript_content(transcript)
            dataset_lines.append({
                "id": transcript.id,
                "type": transcript.type.value,
                "agent_id": transcript.agent_id,
                "task_id": transcript.task_id,
                "session_id": transcript.session_id,
                "timestamp": transcript.timestamp.isoformat(),
                "metadata": transcript.transcript_metadata,
                "content": content
            })

        # For now, return as JSON array. In production, could stream or save to file
        return {"dataset": dataset_lines, "count": len(dataset_lines)}

@app.post("/retention")
async def apply_retention_policy(policy: RetentionPolicy):
    """Apply retention policy - delete transcripts older than specified days"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.days_to_keep)

    async with async_session() as session:
        query = select(Transcript).where(Transcript.created_at < cutoff_date)
        if policy.type:
            query = query.where(Transcript.type == policy.type)

        result = await session.execute(query)
        old_transcripts = result.scalars().all()

        deleted_count = 0
        for transcript in old_transcripts:
            if transcript.minio_key:
                minio_client.delete_transcript(transcript.minio_key)
            await session.delete(transcript)
            deleted_count += 1

        await session.commit()

        return {"message": f"Deleted {deleted_count} transcripts older than {policy.days_to_keep} days"}

@app.get("/stats")
async def get_stats():
    """Get basic statistics about stored transcripts"""
    async with async_session() as session:
        # Total count
        total_result = await session.execute(select(func.count(Transcript.id)))
        total_count = total_result.scalar()

        # Count by type
        type_counts = {}
        for ttype in TranscriptType:
            count_result = await session.execute(
                select(func.count(Transcript.id)).where(Transcript.type == ttype)
            )
            type_counts[ttype.value] = count_result.scalar()

        # Oldest and newest
        oldest_result = await session.execute(
            select(Transcript.created_at).order_by(Transcript.created_at).limit(1)
        )
        oldest = oldest_result.scalar()

        newest_result = await session.execute(
            select(Transcript.created_at).order_by(Transcript.created_at.desc()).limit(1)
        )
        newest = newest_result.scalar()

        return {
            "total_transcripts": total_count,
            "by_type": type_counts,
            "oldest_transcript": oldest,
            "newest_transcript": newest
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/about")
def about():
    """About endpoint"""
    return {
        "service": "Transcript Store",
        "version": "1.0.0",
        "description": "Append-only transcript storage for audit, troubleshooting, and self-improvement"
    }