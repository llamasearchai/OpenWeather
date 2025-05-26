import sqlite3
import json
import pickle
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import asyncio
import aiosqlite

from openweather.core.config import settings

logger = logging.getLogger(__name__)

class WeatherCache:
    """Advanced caching system with SQLite backend and analytics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.CACHE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS weather_cache (
                    id INTEGER PRIMARY KEY,
                    cache_key TEXT UNIQUE,
                    location TEXT,
                    data BLOB,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS llm_cache (
                    id INTEGER PRIMARY KEY,
                    query_hash TEXT UNIQUE,
                    prompt TEXT,
                    response TEXT,
                    provider TEXT,
                    model TEXT,
                    tokens_used INTEGER,
                    created_at TIMESTAMP,
                    embedding BLOB
                );
                
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT,
                    event_data JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_weather_location ON weather_cache(location);
                CREATE INDEX IF NOT EXISTS idx_weather_expires ON weather_cache(expires_at);
                CREATE INDEX IF NOT EXISTS idx_llm_provider ON llm_cache(provider);
                CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type);
            """)
            await db.commit()

    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data."""
        async with self._lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        "SELECT data, expires_at FROM weather_cache WHERE cache_key = ?",
                        (cache_key,)
                    )
                    row = await cursor.fetchone()
                    
                    if row:
                        data_blob, expires_at_str = row
                        expires_at = datetime.fromisoformat(expires_at_str)
                        
                        if datetime.now(timezone.utc) < expires_at:
                            # Update access statistics
                            await db.execute(
                                "UPDATE weather_cache SET access_count = access_count + 1, "
                                "last_accessed = ? WHERE cache_key = ?",
                                (datetime.now(timezone.utc).isoformat(), cache_key)
                            )
                            await db.commit()
                            
                            return pickle.loads(data_blob)
                        else:
                            # Clean up expired entry
                            await db.execute(
                                "DELETE FROM weather_cache WHERE cache_key = ?",
                                (cache_key,)
                            )
                            await db.commit()
                            
                return None
                
            except Exception as e:
                logger.error(f"Cache get error: {e}")
                return None

    async def set(
        self, 
        cache_key: str, 
        data: Dict[str, Any], 
        ttl: int = 3600,
        location: Optional[str] = None
    ):
        """Store data in cache."""
        async with self._lock:
            try:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                data_blob = pickle.dumps(data)
                
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        "INSERT OR REPLACE INTO weather_cache "
                        "(cache_key, location, data, created_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            cache_key,
                            location,
                            data_blob,
                            datetime.now(timezone.utc).isoformat(),
                            expires_at.isoformat()
                        )
                    )
                    await db.commit()
                    
            except Exception as e:
                logger.error(f"Cache set error: {e}")

    async def get_llm_response(self, query_hash: str) -> Optional[str]:
        """Get cached LLM response."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT response FROM llm_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                row = await cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logger.error(f"LLM cache get error: {e}")
            return None

    async def store_llm_response(
        self,
        query_hash: str,
        prompt: str,
        response: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        embedding: Optional[bytes] = None
    ):
        """Store LLM response in cache."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO llm_cache "
                    "(query_hash, prompt, response, provider, model, tokens_used, created_at, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        query_hash,
                        prompt,
                        response,
                        provider,
                        model,
                        tokens_used,
                        datetime.now(timezone.utc).isoformat(),
                        embedding
                    )
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"LLM cache store error: {e}")

    async def search_similar_queries(
        self, 
        query_embedding: bytes, 
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Search for similar cached queries using embeddings."""
        # Placeholder implementation - would use vector similarity in production
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT response, provider, model FROM llm_cache "
                    "WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 5"
                )
                rows = await cursor.fetchall()
                
                return [
                    {
                        "response": row[0],
                        "provider": row[1],
                        "model": row[2],
                        "similarity": 0.9  # Placeholder similarity score
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Similar query search error: {e}")
            return []

    async def log_analytics_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log analytics event."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO analytics_events (event_type, event_data) VALUES (?, ?)",
                    (event_type, json.dumps(event_data))
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Analytics logging error: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Weather cache stats
                cursor = await db.execute(
                    "SELECT COUNT(*), SUM(access_count) FROM weather_cache "
                    "WHERE expires_at > ?"
                    , (datetime.now(timezone.utc).isoformat(),)
                )
                weather_stats = await cursor.fetchone()
                
                # LLM cache stats
                cursor = await db.execute(
                    "SELECT COUNT(*), SUM(tokens_used), provider FROM llm_cache "
                    "GROUP BY provider"
                )
                llm_stats = await cursor.fetchall()
                
                return {
                    "weather_cache": {
                        "entries": weather_stats[0] or 0,
                        "total_hits": weather_stats[1] or 0
                    },
                    "llm_cache": {
                        provider: {"entries": count, "tokens": tokens}
                        for count, tokens, provider in llm_stats
                    }
                }
                
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

    async def cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM weather_cache WHERE expires_at < ?",
                    (datetime.now(timezone.utc).isoformat(),)
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}") 