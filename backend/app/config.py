"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RankingWeights(BaseSettings):
    """Weights for the article ranking algorithm."""

    model_config = SettingsConfigDict(env_prefix="RANKING_")

    # Time decay parameters
    recency_half_life_hours: float = Field(
        default=24.0,
        description="Half-life for recency score exponential decay (hours)",
    )

    # Vote scoring parameters
    vote_decay_days: float = Field(
        default=7.0,
        description="Time constant for vote weight decay (days)",
    )

    # Lambda transition parameters (cold start -> mature)
    lambda_midpoint_votes: int = Field(
        default=10,
        description="Number of votes at which lambda = 0.5",
    )
    lambda_steepness: float = Field(
        default=0.5,
        description="Steepness of the sigmoid transition",
    )

    # Component weights for final score
    weight_votes: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_pagerank: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_topic_match: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("weight_votes", "weight_pagerank", "weight_recency", "weight_topic_match")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0 and 1")
        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Noiseless Newspaper"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False)
    environment: Literal["development", "staging", "production"] = "development"

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./noiseless_newspaper.db",
        description="Async database URL (SQLAlchemy format)",
    )

    # API Keys (all optional for local development)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    arxiv_api_key: str | None = Field(default=None)
    newsapi_key: str | None = Field(default=None)
    semantic_scholar_api_key: str | None = Field(default=None)

    # Scheduler
    fetch_interval_minutes: int = Field(
        default=60,
        description="Interval for fetching new articles from sources",
    )
    pagerank_update_interval_hours: int = Field(
        default=6,
        description="Interval for recalculating PageRank scores",
    )

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
    )

    # Pagination
    default_page_size: int = Field(default=20, ge=1, le=100)
    max_page_size: int = Field(default=100, ge=1, le=500)

    # Ranking weights (nested)
    ranking: RankingWeights = Field(default_factory=RankingWeights)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
