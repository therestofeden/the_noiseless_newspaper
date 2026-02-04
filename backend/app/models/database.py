"""
SQLAlchemy database models for The Noiseless Newspaper.
Uses SQLAlchemy 2.0 async patterns.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.models.domain import ArticleSource, ContentType, VotePeriod


# =============================================================================
# Base
# =============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


# =============================================================================
# Articles
# =============================================================================

class DBArticle(Base):
    """Stored article with all metadata and scores."""
    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Core metadata
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    authors_json: Mapped[Optional[str]] = mapped_column(JSON)  # List of author dicts
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)

    # Topic classification
    topic_path: Mapped[Optional[str]] = mapped_column(String(255))
    topic_relevance_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Citation data
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    citation_velocity: Mapped[float] = mapped_column(Float, default=0.0)
    citing_paper_ids_json: Mapped[Optional[str]] = mapped_column(JSON)
    cited_paper_ids_json: Mapped[Optional[str]] = mapped_column(JSON)

    # Computed scores
    pagerank_score: Mapped[float] = mapped_column(Float, default=0.0)
    recency_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Embedding (stored as JSON array)
    embedding_json: Mapped[Optional[str]] = mapped_column(JSON)

    # Timestamps
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    votes: Mapped[list["DBVote"]] = relationship(back_populates="article", cascade="all, delete-orphan")
    clicks: Mapped[list["DBClick"]] = relationship(back_populates="article", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("ix_articles_topic_path", "topic_path"),
        Index("ix_articles_published_at", "published_at"),
        Index("ix_articles_final_score", "final_score"),
        Index("ix_articles_source_published", "source", "published_at"),
    )


class DBCitation(Base):
    """Citation relationship between articles (for PageRank)."""
    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    citing_article_id: Mapped[str] = mapped_column(String(255), ForeignKey("articles.id"), nullable=False)
    cited_article_id: Mapped[str] = mapped_column(String(255), nullable=False)  # May not be in our DB
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_citations_citing", "citing_article_id"),
        Index("ix_citations_cited", "cited_article_id"),
        Index("ix_citations_pair", "citing_article_id", "cited_article_id", unique=True),
    )


# =============================================================================
# Users
# =============================================================================

class DBUser(Base):
    """User account."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    preferences: Mapped[Optional["DBUserPreferences"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    votes: Mapped[list["DBVote"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    clicks: Mapped[list["DBClick"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class DBUserPreferences(Base):
    """User's topic preferences."""
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"), unique=True)

    # Stored as JSON
    selected_topics_json: Mapped[Optional[str]] = mapped_column(JSON)  # List of topic paths
    topic_frequencies_json: Mapped[Optional[str]] = mapped_column(JSON)  # Dict path -> frequency

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user: Mapped["DBUser"] = relationship(back_populates="preferences")


# =============================================================================
# Votes & Clicks
# =============================================================================

class DBVote(Base):
    """User vote on article relevance."""
    __tablename__ = "votes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"), nullable=False)
    article_id: Mapped[str] = mapped_column(String(255), ForeignKey("articles.id"), nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)  # 1_week, 1_month, 1_year
    score: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    voted_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    user: Mapped["DBUser"] = relationship(back_populates="votes")
    article: Mapped["DBArticle"] = relationship(back_populates="votes")

    __table_args__ = (
        Index("ix_votes_user", "user_id"),
        Index("ix_votes_article", "article_id"),
        Index("ix_votes_article_period", "article_id", "period"),
        Index("ix_votes_user_article_period", "user_id", "article_id", "period", unique=True),
    )


class DBClick(Base):
    """User click/read history."""
    __tablename__ = "clicks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"), nullable=False)
    article_id: Mapped[str] = mapped_column(String(255), ForeignKey("articles.id"), nullable=False)
    topic_path: Mapped[str] = mapped_column(String(255), nullable=False)
    clicked_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    user: Mapped["DBUser"] = relationship(back_populates="clicks")
    article: Mapped["DBArticle"] = relationship(back_populates="clicks")

    __table_args__ = (
        Index("ix_clicks_user", "user_id"),
        Index("ix_clicks_user_topic", "user_id", "topic_path"),
        Index("ix_clicks_user_time", "user_id", "clicked_at"),
    )


# =============================================================================
# Aggregated Stats (Materialized views, computed by batch jobs)
# =============================================================================

class DBArticleVoteStats(Base):
    """Pre-computed vote statistics for articles."""
    __tablename__ = "article_vote_stats"

    article_id: Mapped[str] = mapped_column(String(255), ForeignKey("articles.id"), primary_key=True)

    votes_1_week: Mapped[int] = mapped_column(Integer, default=0)
    votes_1_month: Mapped[int] = mapped_column(Integer, default=0)
    votes_1_year: Mapped[int] = mapped_column(Integer, default=0)

    avg_score_1_week: Mapped[float] = mapped_column(Float, default=0.0)
    avg_score_1_month: Mapped[float] = mapped_column(Float, default=0.0)
    avg_score_1_year: Mapped[float] = mapped_column(Float, default=0.0)

    weighted_vote_score: Mapped[float] = mapped_column(Float, default=0.0)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class DBUserSignalScore(Base):
    """Pre-computed user signal scores (how well their votes predict community consensus)."""
    __tablename__ = "user_signal_scores"

    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"), primary_key=True)
    correlation_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_votes: Mapped[int] = mapped_column(Integer, default=0)
    votes_with_outcome: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


# =============================================================================
# Database Connection
# =============================================================================

class Database:
    """Database connection manager."""

    def __init__(self, database_url: str):
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging
            future=True,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    async def create_tables(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables (use with caution!)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self):
        """Get a database session."""
        async with self.async_session() as session:
            yield session
