"""
Configuration management for The Noiseless Newspaper.
Uses pydantic-settings for type-safe environment variable handling.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "The Noiseless Newspaper"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/noiseless.db"
    # For production: "postgresql+asyncpg://user:pass@host:5432/noiseless"

    # -------------------------------------------------------------------------
    # API Keys (all optional - uses mocks if not provided)
    # -------------------------------------------------------------------------
    # Academic sources
    semantic_scholar_api_key: str | None = None
    # OpenAlex is free, no key needed

    # News sources
    newsapi_key: str | None = None
    gnews_api_key: str | None = None

    # Embeddings
    openai_api_key: str | None = None
    voyage_api_key: str | None = None

    # LLM for summarization
    anthropic_api_key: str | None = None

    # -------------------------------------------------------------------------
    # Retrieval Settings
    # -------------------------------------------------------------------------
    # How many articles to fetch per source per topic per day
    articles_per_topic_per_source: int = 50

    # How many days of articles to keep in the database
    article_retention_days: int = 365

    # Minimum citation count for academic articles (cold start filter)
    min_citations_academic: int = 5

    # PageRank damping factor
    pagerank_damping: float = 0.85

    # PageRank iterations
    pagerank_iterations: int = 100

    # -------------------------------------------------------------------------
    # Embedding Settings
    # -------------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"  # Local fallback
    embedding_dimension: int = 384  # Matches all-MiniLM-L6-v2

    # Use OpenAI embeddings if API key is available
    use_openai_embeddings: bool = False
    openai_embedding_model: str = "text-embedding-3-small"

    # -------------------------------------------------------------------------
    # Ranking Weights
    # -------------------------------------------------------------------------
    # How much weight to give each signal (must sum to 1.0)
    weight_citation_score: float = 0.35  # PageRank-based citation score
    weight_recency: float = 0.20  # How recent the article is
    weight_topic_relevance: float = 0.30  # Embedding similarity to user's topic
    weight_user_votes: float = 0.15  # Aggregated user votes over time

    # Time decay for votes (how much more a 1-year vote counts vs 1-week)
    vote_weight_1_week: float = 0.15
    vote_weight_1_month: float = 0.35
    vote_weight_1_year: float = 0.50

    # -------------------------------------------------------------------------
    # Batch Job Settings
    # -------------------------------------------------------------------------
    # When to run the daily batch job (UTC)
    batch_job_hour: int = 4  # 4 AM UTC
    batch_job_minute: int = 0

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------
    # Requests per minute to external APIs
    arxiv_rate_limit: int = 30
    semantic_scholar_rate_limit: int = 100
    newsapi_rate_limit: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
