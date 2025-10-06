import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import select, update, func, and_, create_engine
from orchestrator.database import PromptCache
from sqlalchemy.orm import sessionmaker
import os

# Create sync engine for cache operations
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///codeagent.db").replace("+aiosqlite", "")
sync_engine = create_engine(DATABASE_URL, echo=False)
sync_session = sessionmaker(sync_engine, expire_on_commit=False)


class PromptCacheManager:
    """Manages prompt-response caching to avoid redundant API calls and save tokens."""

    def __init__(self, default_ttl_hours: int = 24):
        self.default_ttl_hours = default_ttl_hours

    def _generate_cache_key(self, provider: str, model: str, role: str, messages: List[Dict]) -> str:
        """Generate a unique cache key for the prompt."""
        # Create a deterministic string representation of the prompt
        prompt_data = {
            "provider": provider,
            "model": model,
            "role": role,
            "messages": messages
        }

        # Sort keys for consistency and convert to JSON string
        prompt_json = json.dumps(prompt_data, sort_keys=True, separators=(',', ':'))

        # Generate SHA256 hash
        return hashlib.sha256(prompt_json.encode('utf-8')).hexdigest()

    def _generate_prompt_hash(self, messages: List[Dict]) -> str:
        """Generate hash of just the prompt content for duplicate detection."""
        prompt_json = json.dumps(messages, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(prompt_json.encode('utf-8')).hexdigest()

    def get_cached_response(self, provider: str, model: str, role: str, messages: List[Dict]) -> Optional[Dict[str, Any]]:
        """Check if we have a cached response for this prompt."""
        cache_key = self._generate_cache_key(provider, model, role, messages)

        with sync_session() as session:
            # Check for exact cache key match
            result = session.execute(
                select(PromptCache).where(
                    and_(
                        PromptCache.cache_key == cache_key,
                        PromptCache.expires_at > datetime.utcnow()
                    )
                )
            )
            cache_entry = result.scalar_one_or_none()

            if cache_entry:
                # Update hit count and last used
                cache_entry.hit_count += 1
                cache_entry.last_used = datetime.utcnow()
                session.commit()

                return {
                    "response": cache_entry.response,
                    "tokens_used": cache_entry.tokens_used,
                    "latency_ms": cache_entry.latency_ms,
                    "cost_estimate": cache_entry.cost_estimate,
                    "cached": True,
                    "hit_count": cache_entry.hit_count
                }

        return None

    def store_response(self, provider: str, model: str, role: str, messages: List[Dict],
                            response: Dict[str, Any], tokens_used: int, latency_ms: int,
                            cost_estimate: float, ttl_hours: Optional[int] = None) -> str:
        """Store a prompt-response pair in the cache."""
        cache_key = self._generate_cache_key(provider, model, role, messages)
        prompt_hash = self._generate_prompt_hash(messages)

        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours or self.default_ttl_hours)

        with sync_session() as session:
            # Check if this exact prompt is already cached
            existing = session.execute(
                select(PromptCache).where(PromptCache.cache_key == cache_key)
            )
            existing_entry = existing.scalar_one_or_none()

            if existing_entry:
                # Update existing entry
                existing_entry.response = response
                existing_entry.tokens_used = tokens_used
                existing_entry.latency_ms = latency_ms
                existing_entry.cost_estimate = cost_estimate
                existing_entry.hit_count += 1
                existing_entry.last_used = datetime.utcnow()
                existing_entry.expires_at = expires_at
                session.commit()
            else:
                # Create new cache entry
                cache_entry = PromptCache(
                    cache_key=cache_key,
                    provider=provider,
                    model=model,
                    role=role,
                    prompt_hash=prompt_hash,
                    prompt_content=messages,
                    response=response,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    cost_estimate=cost_estimate,
                    expires_at=expires_at
                )
                session.add(cache_entry)
                session.commit()

        return cache_key

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sync_session() as session:
            # Total entries
            total_result = session.execute(
                select(func.count(PromptCache.id))
            )
            total_entries = total_result.scalar()

            # Active entries (not expired)
            active_result = session.execute(
                select(func.count(PromptCache.id)).where(
                    PromptCache.expires_at > datetime.utcnow()
                )
            )
            active_entries = active_result.scalar()

            # Total hits
            hits_result = session.execute(
                select(func.sum(PromptCache.hit_count))
            )
            total_hits = hits_result.scalar() or 0

            # Total tokens saved
            tokens_result = session.execute(
                select(func.sum(PromptCache.tokens_used * (PromptCache.hit_count - 1)))
            )
            tokens_saved = tokens_result.scalar() or 0

            # Total cost saved
            cost_result = session.execute(
                select(func.sum(PromptCache.cost_estimate * (PromptCache.hit_count - 1)))
            )
            cost_saved = cost_result.scalar() or 0

            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "expired_entries": total_entries - active_entries,
                "total_hits": total_hits,
                "tokens_saved": tokens_saved,
                "cost_saved": cost_saved
            }

    def clear_expired_cache(self) -> int:
        """Remove expired cache entries. Returns number of entries removed."""
        with sync_session() as session:
            result = session.execute(
                select(PromptCache).where(PromptCache.expires_at <= datetime.utcnow())
            )
            expired_entries = result.scalars().all()

            count = len(expired_entries)
            for entry in expired_entries:
                session.delete(entry)

            session.commit()
            return count

    def clear_all_cache(self) -> int:
        """Clear all cache entries. Returns number of entries removed."""
        with sync_session() as session:
            result = session.execute(select(func.count(PromptCache.id)))
            count = result.scalar()

            session.execute(PromptCache.__table__.delete())
            session.commit()

            return count

    def find_similar_prompts(self, messages: List[Dict], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar prompts based on prompt hash."""
        prompt_hash = self._generate_prompt_hash(messages)

        with sync_session() as session:
            result = session.execute(
                select(PromptCache).where(
                    and_(
                        PromptCache.prompt_hash == prompt_hash,
                        PromptCache.expires_at > datetime.utcnow()
                    )
                ).order_by(PromptCache.last_used.desc()).limit(limit)
            )
            similar_entries = result.scalars().all()

            return [{
                "cache_key": entry.cache_key,
                "provider": entry.provider,
                "model": entry.model,
                "hit_count": entry.hit_count,
                "last_used": entry.last_used.isoformat(),
                "response_preview": str(entry.response)[:200] + "..." if len(str(entry.response)) > 200 else str(entry.response)
            } for entry in similar_entries]


# Global cache manager instance
prompt_cache_manager = PromptCacheManager()