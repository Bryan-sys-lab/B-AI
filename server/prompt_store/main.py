from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func, desc
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from prompt_store.database import async_session, init_db, Prompt, PromptVersion, PromptMetrics
from prompt_store.minio_client import minio_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Prompt Store Service", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PromptCreate(BaseModel):
    name: str
    description: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None

class PromptUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class PromptResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_by: str
    created_at: datetime
    updated_at: datetime
    current_version_id: Optional[str] = None
    version_count: int = 0

class PromptVersionCreate(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class PromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    version_number: int
    content: str
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool
    created_at: datetime

class PromptMetricsResponse(BaseModel):
    id: str
    prompt_version_id: str
    execution_count: int
    success_rate: float
    avg_latency_ms: float
    avg_cost: float
    last_used_at: Optional[datetime] = None

class MetricsRecord(BaseModel):
    prompt_version_id: str
    success: bool
    latency_ms: float
    cost: float

# Constants
CONTENT_SIZE_THRESHOLD = 10240  # 10KB - store in MinIO if larger

# Helper functions
def should_store_in_minio(content: str) -> bool:
    return len(content.encode('utf-8')) > CONTENT_SIZE_THRESHOLD

async def get_prompt_content(version: PromptVersion) -> str:
    if version.minio_key:
        return minio_client.download_prompt(version.minio_key)
    return version.content or ""

async def store_prompt_content(prompt_id: str, version_number: int, content: str) -> tuple[str, Optional[str]]:
    if should_store_in_minio(content):
        minio_key = minio_client.upload_prompt(content, prompt_id, version_number)
        return "", minio_key
    return content, None

# API Endpoints

@app.post("/prompts", response_model=PromptResponse)
async def create_prompt(prompt_data: PromptCreate):
    """Create a new prompt with initial version"""
    async with async_session() as session:
        # Check if prompt name already exists
        result = await session.execute(
            select(Prompt).where(Prompt.name == prompt_data.name)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Prompt name already exists")

        # Create prompt
        prompt = Prompt(
            name=prompt_data.name,
            description=prompt_data.description,
            created_by="system"  # TODO: Add user authentication
        )
        session.add(prompt)
        await session.flush()

        # Create initial version
        content, minio_key = await store_prompt_content(
            prompt.id, 1, prompt_data.content
        )

        version = PromptVersion(
            prompt_id=prompt.id,
            version_number=1,
            content=content,
            minio_key=minio_key,
            prompt_metadata=prompt_data.metadata,
            is_active=True
        )
        session.add(version)
        await session.flush()

        # Update prompt's current version
        prompt.current_version_id = version.id
        await session.commit()

        return PromptResponse(
            id=prompt.id,
            name=prompt.name,
            description=prompt.description,
            created_by=prompt.created_by,
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            current_version_id=prompt.current_version_id,
            version_count=1
        )

@app.get("/prompts", response_model=List[PromptResponse])
async def list_prompts(skip: int = 0, limit: int = 100):
    """List all prompts"""
    async with async_session() as session:
        result = await session.execute(
            select(Prompt).offset(skip).limit(limit)
        )
        prompts = result.scalars().all()

        responses = []
        for prompt in prompts:
            # Count versions
            version_result = await session.execute(
                select(func.count(PromptVersion.id)).where(PromptVersion.prompt_id == prompt.id)
            )
            version_count = version_result.scalar()

            responses.append(PromptResponse(
                id=prompt.id,
                name=prompt.name,
                description=prompt.description,
                created_by=prompt.created_by,
                created_at=prompt.created_at,
                updated_at=prompt.updated_at,
                current_version_id=prompt.current_version_id,
                version_count=version_count
            ))

        return responses

@app.get("/prompts/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """Get a specific prompt"""
    async with async_session() as session:
        result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt = result.scalar_one_or_none()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        # Count versions
        version_result = await session.execute(
            select(func.count(PromptVersion.id)).where(PromptVersion.prompt_id == prompt.id)
        )
        version_count = version_result.scalar()

        return PromptResponse(
            id=prompt.id,
            name=prompt.name,
            description=prompt.description,
            created_by=prompt.created_by,
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            current_version_id=prompt.current_version_id,
            version_count=version_count
        )

@app.put("/prompts/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, prompt_data: PromptUpdate):
    """Update prompt metadata"""
    async with async_session() as session:
        result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt = result.scalar_one_or_none()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        if prompt_data.name:
            prompt.name = prompt_data.name
        if prompt_data.description is not None:
            prompt.description = prompt_data.description

        await session.commit()

        # Count versions
        version_result = await session.execute(
            select(func.count(PromptVersion.id)).where(PromptVersion.prompt_id == prompt.id)
        )
        version_count = version_result.scalar()

        return PromptResponse(
            id=prompt.id,
            name=prompt.name,
            description=prompt.description,
            created_by=prompt.created_by,
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            current_version_id=prompt.current_version_id,
            version_count=version_count
        )

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt and all its versions"""
    async with async_session() as session:
        result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt = result.scalar_one_or_none()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        # Delete associated MinIO objects
        version_result = await session.execute(
            select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
        )
        versions = version_result.scalars().all()
        for version in versions:
            if version.minio_key:
                minio_client.delete_prompt(version.minio_key)

        # Delete prompt (cascade will delete versions and metrics)
        await session.delete(prompt)
        await session.commit()

        return {"message": "Prompt deleted successfully"}

@app.post("/prompts/{prompt_id}/versions", response_model=PromptVersionResponse)
async def create_prompt_version(prompt_id: str, version_data: PromptVersionCreate):
    """Create a new version of a prompt"""
    async with async_session() as session:
        # Check if prompt exists
        prompt_result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt = prompt_result.scalar_one_or_none()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        # Get next version number
        version_result = await session.execute(
            select(func.max(PromptVersion.version_number)).where(PromptVersion.prompt_id == prompt_id)
        )
        max_version = version_result.scalar() or 0
        next_version = max_version + 1

        # Store content
        content, minio_key = await store_prompt_content(
            prompt_id, next_version, version_data.content
        )

        # Create version
        version = PromptVersion(
            prompt_id=prompt_id,
            version_number=next_version,
            content=content,
            minio_key=minio_key,
            prompt_metadata=version_data.metadata,
            is_active=True
        )
        session.add(version)
        await session.flush()

        # Update prompt's current version
        prompt.current_version_id = version.id
        await session.commit()

        return PromptVersionResponse(
            id=version.id,
            prompt_id=version.prompt_id,
            version_number=version.version_number,
            content=await get_prompt_content(version),
            metadata=version.prompt_metadata,
            is_active=version.is_active,
            created_at=version.created_at
        )

@app.get("/prompts/{prompt_id}/versions", response_model=List[PromptVersionResponse])
async def list_prompt_versions(prompt_id: str):
    """List all versions of a prompt"""
    async with async_session() as session:
        # Check if prompt exists
        prompt_result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        if not prompt_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Prompt not found")

        result = await session.execute(
            select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
            .order_by(desc(PromptVersion.version_number))
        )
        versions = result.scalars().all()

        responses = []
        for version in versions:
            responses.append(PromptVersionResponse(
                id=version.id,
                prompt_id=version.prompt_id,
                version_number=version.version_number,
                content=await get_prompt_content(version),
                metadata=version.prompt_metadata,
                is_active=version.is_active,
                created_at=version.created_at
            ))

        return responses

@app.get("/prompts/{prompt_id}/versions/{version_id}", response_model=PromptVersionResponse)
async def get_prompt_version(prompt_id: str, version_id: str):
    """Get a specific version of a prompt"""
    async with async_session() as session:
        result = await session.execute(
            select(PromptVersion).where(
                PromptVersion.id == version_id,
                PromptVersion.prompt_id == prompt_id
            )
        )
        version = result.scalar_one_or_none()
        if not version:
            raise HTTPException(status_code=404, detail="Prompt version not found")

        return PromptVersionResponse(
            id=version.id,
            prompt_id=version.prompt_id,
            version_number=version.version_number,
            content=await get_prompt_content(version),
            metadata=version.prompt_metadata,
            is_active=version.is_active,
            created_at=version.created_at
        )

@app.post("/prompts/{prompt_id}/rollback/{version_id}")
async def rollback_prompt(prompt_id: str, version_id: str):
    """Rollback prompt to a specific version"""
    async with async_session() as session:
        # Check if prompt exists
        prompt_result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        prompt = prompt_result.scalar_one_or_none()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        # Check if version exists
        version_result = await session.execute(
            select(PromptVersion).where(
                PromptVersion.id == version_id,
                PromptVersion.prompt_id == prompt_id
            )
        )
        version = version_result.scalar_one_or_none()
        if not version:
            raise HTTPException(status_code=404, detail="Prompt version not found")

        # Update prompt's current version
        prompt.current_version_id = version_id
        await session.commit()

        return {"message": f"Prompt rolled back to version {version.version_number}"}

@app.post("/metrics")
async def record_metrics(metrics_data: MetricsRecord):
    """Record usage metrics for a prompt version"""
    async with async_session() as session:
        # Check if version exists
        version_result = await session.execute(
            select(PromptVersion).where(PromptVersion.id == metrics_data.prompt_version_id)
        )
        version = version_result.scalar_one_or_none()
        if not version:
            raise HTTPException(status_code=404, detail="Prompt version not found")

        # Get or create metrics record
        metrics_result = await session.execute(
            select(PromptMetrics).where(PromptMetrics.prompt_version_id == metrics_data.prompt_version_id)
        )
        metrics = metrics_result.scalar_one_or_none()

        if not metrics:
            metrics = PromptMetrics(prompt_version_id=metrics_data.prompt_version_id)
            session.add(metrics)

        # Update metrics
        metrics.execution_count += 1
        if metrics_data.success:
            metrics.success_count += 1
        metrics.total_latency_ms += metrics_data.latency_ms
        metrics.total_cost += metrics_data.cost
        metrics.last_used_at = datetime.utcnow()

        await session.commit()

        return {"message": "Metrics recorded successfully"}

@app.get("/prompts/{prompt_id}/metrics", response_model=List[PromptMetricsResponse])
async def get_prompt_metrics(prompt_id: str):
    """Get metrics for all versions of a prompt"""
    async with async_session() as session:
        # Check if prompt exists
        prompt_result = await session.execute(
            select(Prompt).where(Prompt.id == prompt_id)
        )
        if not prompt_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Prompt not found")

        result = await session.execute(
            select(PromptMetrics).join(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
        )
        metrics_list = result.scalars().all()

        responses = []
        for metrics in metrics_list:
            responses.append(PromptMetricsResponse(
                id=metrics.id,
                prompt_version_id=metrics.prompt_version_id,
                execution_count=metrics.execution_count,
                success_rate=metrics.success_rate,
                avg_latency_ms=metrics.avg_latency_ms,
                avg_cost=metrics.avg_cost,
                last_used_at=metrics.last_used_at
            ))

        return responses

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/about")
def about():
    """About endpoint"""
    return {
        "service": "Prompt Store",
        "version": "1.0.0",
        "description": "Versioned store for Aetherium prompts with metrics tracking"
    }