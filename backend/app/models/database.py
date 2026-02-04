"""
SQLAlchemy async database models.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings

if TYPE_CHECKING:
    pass


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""

    pass


class DBArticle(Base):
    """Database model for articles."""

    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    authors_json: Mapped[str] = mapped_column(Text, default="[]")
    topics_json: Mapped[str] = mapped_column(Text, default="[]")
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    citation_ids_json: Mapped[str] = mapped_column(Text, default="[]")

    # Cached ranking scores
    pagerank_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    last_pagerank_update: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    votes: Mapped[list["DBVote"]] = relationship("DBVote", back_populates="article")
    clicks: Mapped[list["DBClick"]] = relationship("DBClick", back_populates="article")
    vote_stats: Mapped["DBArticleVoteStats | None"] = relationship(
        "DBArticleVoteStats",
        back_populates="article",
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uix_source_external_id"),
        Index("ix_articles_published_pagerank", "published_at", "pagerank_score"),
    )


class DBUser(Base):
    """Database model for users."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    username: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    votes: Mapped[list["DBVote"]] = relationship("DBVote", back_populates="user")
    clicks: Mapped[list["DBClick"]] = relationship("DBClick", back_populates="user")
    preferences: Mapped["DBUserPreferences | None"] = relationship(
        "DBUserPreferences",
        back_populates="user",
        uselist=False,
    )


class DBVote(Base):
    """Database model for user votes on articles."""

    __tablename__ = "votes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    article_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vote_type: Mapped[str] = mapped_column(
        Enum("upvote", "downvote", "bookmark", name="vote_type_enum"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user: Mapped["DBUser"] = relationship("DBUser", back_populates="votes")
    article: Mapped["DBArticle"] = relationship("DBArticle", back_populates="votes")

    __table_args__ = (
        UniqueConstraint("user_id", "article_id", "vote_type", name="uix_user_article_vote"),
        Index("ix_votes_article_created", "article_id", "created_at"),
    )


class DBClick(Base):
    """Database model for tracking article clicks/reads."""

    __tablename__ = "clicks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    article_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    clicked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    dwell_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["DBUser"] = relationship("DBUser", back_populates="clicks")
    article: Mapped["DBArticle"] = relationship("DBArticle", back_populates="clicks")


class DBUserPreferences(Base):
    """Database model for user preferences."""

    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    topic_preferences_json: Mapped[str] = mapped_column(Text, default="[]")
    daily_article_count: Mapped[int] = mapped_column(Integer, default=5)
    preferred_sources_json: Mapped[str] = mapped_column(Text, default="[]")
    excluded_sources_json: Mapped[str] = mapped_column(Text, default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationship
    user: Mapped["DBUser"] = relationship("DBUser", back_populates="preferences")


class DBCitation(Base):
    """Database model for citation relationships between articles."""

    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    citing_article_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cited_article_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("citing_article_id", "cited_article_id", name="uix_citation"),
        Index("ix_citations_cited", "cited_article_id"),
    )


class DBArticleVoteStats(Base):
    """Aggregated vote statistics for articles (materialized for performance)."""

    __tablename__ = "article_vote_stats"

    article_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    upvote_count: Mapped[int] = mapped_column(Integer, default=0)
    downvote_count: Mapped[int] = mapped_column(Integer, default=0)
    bookmark_count: Mapped[int] = mapped_column(Integer, default=0)
    total_votes: Mapped[int] = mapped_column(Integer, default=0, index=True)
    weighted_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    last_vote_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationship
    article: Mapped["DBArticle"] = relationship("DBArticle", back_populates="vote_stats")


# Database engine and session factory
_engine = None
_session_factory = None


async def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            future=True,
        )
    return _engine


async def get_session_factory():
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )
    return _session_factory


async def init_db():
    """Initialize the database (create tables)."""
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
