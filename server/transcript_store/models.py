from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TranscriptType(str, Enum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    ERROR = "error"

class TranscriptCreate(BaseModel):
    type: TranscriptType
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None  # If not provided, use current time
    metadata: Optional[Dict[str, Any]] = None
    content: str

class TranscriptResponse(BaseModel):
    id: str
    type: TranscriptType
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    content: str
    created_at: datetime

class TranscriptSearch(BaseModel):
    type: Optional[TranscriptType] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
    offset: int = 0

class DatasetExport(BaseModel):
    type: Optional[TranscriptType] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    format: str = "json"  # json, csv, etc.

class RetentionPolicy(BaseModel):
    days_to_keep: int
    type: Optional[TranscriptType] = None  # If specified, only apply to this type