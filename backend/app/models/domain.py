"""
Pydantic domain models for the API layer.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VoteType(str, Enum):
    """Types of votes users can cast."""

    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    BOOKMARK = "bookmark"


class TopicPath(BaseModel):
    """Represents a path through the taxonomy."""

    model_config = ConfigDict(frozen=True)

    domain: str = Field(..., description="Top-level domain ID (e.g., 'ai-ml')")
    subtopic: str = Field(..., description="Mid-level subtopic ID (e.g., 'llms')")
    niche: str | None = Field(None, description="Bottom-level niche ID (e.g., 'architectures')")

    @property
    def path_string(self) -> str:
        """Return dot-separated path string."""
        if self.niche:
            return f"{self.domain}.{self.subtopic}.{self.niche}"
        return f"{self.domain}.{self.subtopic}"

    @classmethod
    def from_string(cls, path: str) -> "TopicPath":
        """Parse a dot-separated path string."""
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid topic path: {path}")
        return cls(
            domain=parts[0],
            subtopic=parts[1],
            niche=parts[2] if len(parts) > 2 else None,
        )


class Author(BaseModel):
    """Article author information."""

    model_config = ConfigDict(frozen=True)

    name: str
    affiliation: str | None = None
    url: str | None = None


class Article(BaseModel):
    """Core article model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source: str = Field(..., description="Source identifier (e.g., 'arxiv', 'newsapi')")
    external_id: str = Field(..., description="ID in the source system")
    title: str
    abstract: str | None = None
    url: str
    authors: list[Author] = Field(default_factory=list)
    topics: list[TopicPath] = Field(default_factory=list)
    published_at: datetime
    fetched_at: datetime
    citation_count: int = 0
    citation_ids: list[str] = Field(default_factory=list, description="IDs of cited articles")

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()


class ArticleRanking(BaseModel):
    """Article with ranking scores."""

    model_config = ConfigDict(from_attributes=True)

    article: Article
    final_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)
    vote_score: float = Field(..., ge=0.0, le=1.0)
    pagerank_score: float = Field(..., ge=0.0, le=1.0)
    topic_match_score: float = Field(..., ge=0.0, le=1.0)
    lambda_value: float = Field(..., ge=0.0, le=1.0, description="Cold-start transition factor")


class UserVote(BaseModel):
    """A user's vote on an article."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    article_id: UUID
    vote_type: VoteType
    created_at: datetime


class UserVoteCreate(BaseModel):
    """Request model for creating a vote."""

    article_id: UUID
    vote_type: VoteType


class TopicPreference(BaseModel):
    """User's preference for a topic."""

    topic_path: TopicPath
    weight: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    excluded: bool = False


class UserPreferences(BaseModel):
    """User's topic preferences and settings."""

    model_config = ConfigDict(from_attributes=True)

    user_id: UUID
    topic_preferences: list[TopicPreference] = Field(default_factory=list)
    daily_article_count: int = Field(default=5, ge=1, le=50)
    preferred_sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    updated_at: datetime


class UserPreferencesUpdate(BaseModel):
    """Request model for updating preferences."""

    topic_preferences: list[TopicPreference] | None = None
    daily_article_count: int | None = Field(None, ge=1, le=50)
    preferred_sources: list[str] | None = None
    excluded_sources: list[str] | None = None


class UserStats(BaseModel):
    """Statistics about a user's engagement."""

    user_id: UUID
    total_votes: int = 0
    upvotes: int = 0
    downvotes: int = 0
    bookmarks: int = 0
    articles_read: int = 0
    favorite_topics: list[TopicPath] = Field(default_factory=list)
    join_date: datetime
    last_active: datetime


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    items: list
    total: int
    page: int
    page_size: int
    total_pages: int

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


class ArticleListResponse(BaseModel):
    """Paginated list of ranked articles."""

    articles: list[ArticleRanking]
    total: int
    page: int
    page_size: int
    total_pages: int


class DailyArticleResponse(BaseModel):
    """Response for daily article recommendation."""

    articles: list[ArticleRanking]
    generated_at: datetime
    topic_coverage: dict[str, int] = Field(
        default_factory=dict,
        description="Count of articles per topic",
    )
