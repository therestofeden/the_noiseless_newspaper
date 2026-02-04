"""
Domain models for The Noiseless Newspaper.
These are the core business entities, independent of database/API representation.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class ArticleSource(str, Enum):
    """Sources from which articles can be retrieved."""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENLEX = "openalex"
    NEWSAPI = "newsapi"
    GNEWS = "gnews"
    RSS = "rss"
    MOCK = "mock"


class ContentType(str, Enum):
    """Type of content - affects ranking behavior."""
    RESEARCH = "research"  # Academic papers, preprints
    NEWS = "news"  # Current events, journalism
    ANALYSIS = "analysis"  # Long-form analysis, reports


class VotePeriod(str, Enum):
    """Time periods at which users vote on relevance."""
    ONE_WEEK = "1_week"
    ONE_MONTH = "1_month"
    ONE_YEAR = "1_year"


# =============================================================================
# Topic Taxonomy
# =============================================================================

class TopicNiche(BaseModel):
    """Deepest level of topic specificity."""
    id: str
    name: str
    keywords: list[str] = Field(default_factory=list)  # For search queries
    embedding: Optional[list[float]] = None  # Precomputed topic embedding


class TopicSubtopic(BaseModel):
    """Mid-level topic category."""
    id: str
    name: str
    niches: dict[str, TopicNiche] = Field(default_factory=dict)


class TopicCategory(BaseModel):
    """Top-level domain/category."""
    id: str
    name: str
    icon: str
    description: str
    content_type: ContentType = ContentType.RESEARCH
    subtopics: dict[str, TopicSubtopic] = Field(default_factory=dict)


class TopicPath(BaseModel):
    """A specific path through the taxonomy (e.g., ai-ml/llms/interpretability)."""
    category_id: str
    subtopic_id: str
    niche_id: str

    @property
    def path(self) -> str:
        return f"{self.category_id}/{self.subtopic_id}/{self.niche_id}"

    @classmethod
    def from_path(cls, path: str) -> "TopicPath":
        parts = path.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid topic path: {path}")
        return cls(category_id=parts[0], subtopic_id=parts[1], niche_id=parts[2])


# =============================================================================
# Articles
# =============================================================================

class Author(BaseModel):
    """Article author."""
    name: str
    affiliation: Optional[str] = None
    orcid: Optional[str] = None


class Article(BaseModel):
    """Core article entity."""
    id: str  # Unique ID (source-specific, e.g., arxiv:2401.12345)
    source: ArticleSource
    content_type: ContentType

    # Core metadata
    title: str
    abstract: Optional[str] = None
    summary: Optional[str] = None  # LLM-generated summary
    authors: list[Author] = Field(default_factory=list)
    published_at: datetime
    url: str  # Link to full article

    # Topic classification
    topic_path: Optional[str] = None  # Best matching topic path
    topic_relevance_score: float = 0.0  # Embedding similarity

    # Citation data (for research articles)
    citation_count: int = 0
    citation_velocity: float = 0.0  # Citations per month
    citing_paper_ids: list[str] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)

    # Computed scores
    pagerank_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0

    # Embedding
    embedding: Optional[list[float]] = None

    # Timestamps
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    scored_at: Optional[datetime] = None


class ArticleRanking(BaseModel):
    """A ranked article with all scoring components exposed."""
    article: Article
    rank: int

    # Score breakdown
    citation_score: float
    recency_score: float
    topic_relevance_score: float
    user_vote_score: float
    final_score: float

    # Debug info
    scoring_explanation: Optional[str] = None


# =============================================================================
# User Interactions
# =============================================================================

class UserVote(BaseModel):
    """A user's vote on an article's relevance."""
    user_id: str
    article_id: str
    period: VotePeriod
    score: int = Field(ge=1, le=5)  # 1-5 relevance score
    voted_at: datetime = Field(default_factory=datetime.utcnow)


class UserClick(BaseModel):
    """Record of a user clicking on / reading an article."""
    user_id: str
    article_id: str
    topic_path: str
    clicked_at: datetime = Field(default_factory=datetime.utcnow)


class UserPreferences(BaseModel):
    """User's topic preferences and settings."""
    user_id: str
    selected_topics: list[str] = Field(default_factory=list)  # Topic paths
    topic_frequencies: dict[str, str] = Field(default_factory=dict)  # path -> high/medium/low
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Aggregated Scores
# =============================================================================

class ArticleVoteStats(BaseModel):
    """Aggregated vote statistics for an article."""
    article_id: str

    # Vote counts by period
    votes_1_week: int = 0
    votes_1_month: int = 0
    votes_1_year: int = 0

    # Average scores by period
    avg_score_1_week: float = 0.0
    avg_score_1_month: float = 0.0
    avg_score_1_year: float = 0.0

    # Weighted combined score
    weighted_vote_score: float = 0.0


class UserSignalScore(BaseModel):
    """How well a user's votes predict long-term community consensus."""
    user_id: str
    correlation_score: float = 0.0  # -1 to 1
    total_votes: int = 0
    votes_with_outcome: int = 0  # Votes where we have long-term data
    computed_at: datetime = Field(default_factory=datetime.utcnow)
