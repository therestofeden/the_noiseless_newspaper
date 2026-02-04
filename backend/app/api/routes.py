"""
FastAPI routes for the Noiseless Newspaper API.
"""

import json
from datetime import datetime
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.taxonomy import get_taxonomy
from app.models.database import (
    DBArticle,
    DBArticleVoteStats,
    DBClick,
    DBUser,
    DBUserPreferences,
    DBVote,
    get_session_factory,
)
from app.models.domain import (
    Article,
    ArticleRanking,
    Author,
    DailyArticleResponse,
    TopicPath,
    TopicPreference,
    UserPreferences,
    UserPreferencesUpdate,
    UserStats,
    UserVote,
    UserVoteCreate,
    VoteType,
)
from app.services.citation_graph import get_citation_graph_service
from app.services.ranking import get_ranking_service
from app.sources.mock import get_mock_adapter

logger = structlog.get_logger(__name__)
router = APIRouter()


async def get_db_session() -> AsyncSession:
    """Dependency to get a database session."""
    session_factory = await get_session_factory()
    async with session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


# ============================================================================
# Taxonomy Routes
# ============================================================================


@router.get("/taxonomy")
async def get_taxonomy_endpoint():
    """
    Get the full taxonomy structure.

    Returns the complete 3-level taxonomy with domains, subtopics, and niches.
    """
    taxonomy = get_taxonomy()
    return {
        "taxonomy": taxonomy.to_dict(),
        "domains": list(taxonomy.domains.keys()),
    }


# ============================================================================
# User Preferences Routes
# ============================================================================


@router.get("/users/{user_id}/preferences", response_model=UserPreferences)
async def get_user_preferences(
    user_id: UUID,
    session: SessionDep,
):
    """
    Get a user's preferences.

    Creates default preferences if none exist.
    """
    # Ensure user exists
    user_result = await session.execute(
        select(DBUser).where(DBUser.id == str(user_id))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        # Create user if doesn't exist
        user = DBUser(id=str(user_id))
        session.add(user)
        await session.flush()

    # Get or create preferences
    prefs_result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == str(user_id))
    )
    prefs = prefs_result.scalar_one_or_none()

    if not prefs:
        prefs = DBUserPreferences(user_id=str(user_id))
        session.add(prefs)
        await session.commit()
        await session.refresh(prefs)

    # Parse JSON fields
    topic_prefs = json.loads(prefs.topic_preferences_json)
    preferred_sources = json.loads(prefs.preferred_sources_json)
    excluded_sources = json.loads(prefs.excluded_sources_json)

    return UserPreferences(
        user_id=user_id,
        topic_preferences=[
            TopicPreference(
                topic_path=TopicPath.from_string(tp["path"]),
                weight=tp.get("weight", 1.0),
                excluded=tp.get("excluded", False),
            )
            for tp in topic_prefs
        ],
        daily_article_count=prefs.daily_article_count,
        preferred_sources=preferred_sources,
        excluded_sources=excluded_sources,
        updated_at=prefs.updated_at,
    )


@router.put("/users/{user_id}/preferences", response_model=UserPreferences)
async def update_user_preferences(
    user_id: UUID,
    update: UserPreferencesUpdate,
    session: SessionDep,
):
    """
    Update a user's preferences.

    Only provided fields are updated.
    """
    # Ensure user exists
    user_result = await session.execute(
        select(DBUser).where(DBUser.id == str(user_id))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        user = DBUser(id=str(user_id))
        session.add(user)
        await session.flush()

    # Get or create preferences
    prefs_result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == str(user_id))
    )
    prefs = prefs_result.scalar_one_or_none()

    if not prefs:
        prefs = DBUserPreferences(user_id=str(user_id))
        session.add(prefs)

    # Update fields if provided
    if update.topic_preferences is not None:
        prefs.topic_preferences_json = json.dumps([
            {
                "path": tp.topic_path.path_string,
                "weight": tp.weight,
                "excluded": tp.excluded,
            }
            for tp in update.topic_preferences
        ])

    if update.daily_article_count is not None:
        prefs.daily_article_count = update.daily_article_count

    if update.preferred_sources is not None:
        prefs.preferred_sources_json = json.dumps(update.preferred_sources)

    if update.excluded_sources is not None:
        prefs.excluded_sources_json = json.dumps(update.excluded_sources)

    prefs.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(prefs)

    logger.info("Updated user preferences", user_id=str(user_id))

    # Return updated preferences
    return await get_user_preferences(user_id, session)


# ============================================================================
# Daily Article Routes
# ============================================================================


@router.get("/users/{user_id}/daily-article", response_model=DailyArticleResponse)
async def get_daily_article(
    user_id: UUID,
    session: SessionDep,
    count: Annotated[int | None, Query(ge=1, le=50)] = None,
):
    """
    Get personalized daily article recommendations for a user.

    Uses the ranking algorithm to select the best articles based on
    user preferences, recency, votes, and PageRank.
    """
    settings = get_settings()

    # Get user preferences
    prefs_result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == str(user_id))
    )
    prefs_db = prefs_result.scalar_one_or_none()

    user_prefs = None
    article_count = count or settings.default_page_size

    if prefs_db:
        topic_prefs = json.loads(prefs_db.topic_preferences_json)
        user_prefs = UserPreferences(
            user_id=user_id,
            topic_preferences=[
                TopicPreference(
                    topic_path=TopicPath.from_string(tp["path"]),
                    weight=tp.get("weight", 1.0),
                    excluded=tp.get("excluded", False),
                )
                for tp in topic_prefs
            ],
            daily_article_count=prefs_db.daily_article_count,
            preferred_sources=json.loads(prefs_db.preferred_sources_json),
            excluded_sources=json.loads(prefs_db.excluded_sources_json),
            updated_at=prefs_db.updated_at,
        )
        article_count = count or prefs_db.daily_article_count

    # Fetch articles from mock source (in production, query DB)
    mock_adapter = get_mock_adapter()

    # Filter by preferred domains if set
    preferred_domains = None
    if user_prefs and user_prefs.topic_preferences:
        preferred_domains = list(set(
            tp.topic_path.domain
            for tp in user_prefs.topic_preferences
            if not tp.excluded
        ))

    articles = await mock_adapter.fetch_articles(
        domains=preferred_domains or None,
        limit=article_count * 3,  # Fetch more to allow for filtering
    )

    if not articles:
        return DailyArticleResponse(
            articles=[],
            generated_at=datetime.utcnow(),
            topic_coverage={},
        )

    # Get ranking service and citation graph
    ranking_service = get_ranking_service()
    citation_service = get_citation_graph_service()

    # Build pagerank scores dict
    pagerank_scores = {
        article.id: citation_service.get_score(str(article.id))
        for article in articles
    }

    # Get vote stats from DB (simplified - in production would batch query)
    vote_stats: dict[UUID, tuple[int, int, list[datetime] | None]] = {}
    for article in articles:
        stats_result = await session.execute(
            select(DBArticleVoteStats).where(
                DBArticleVoteStats.article_id == str(article.id)
            )
        )
        stats = stats_result.scalar_one_or_none()
        if stats:
            vote_stats[article.id] = (stats.upvote_count, stats.downvote_count, None)
        else:
            vote_stats[article.id] = (0, 0, None)

    # Rank articles
    ranked = ranking_service.rank_articles(
        articles=articles,
        pagerank_scores=pagerank_scores,
        vote_stats=vote_stats,
        user_preferences=user_prefs,
    )

    # Take top N
    top_articles = ranked[:article_count]

    # Compute topic coverage
    topic_coverage: dict[str, int] = {}
    for ranking in top_articles:
        for topic in ranking.article.topics:
            domain = topic.domain
            topic_coverage[domain] = topic_coverage.get(domain, 0) + 1

    logger.info(
        "Generated daily articles",
        user_id=str(user_id),
        count=len(top_articles),
        topics=topic_coverage,
    )

    return DailyArticleResponse(
        articles=top_articles,
        generated_at=datetime.utcnow(),
        topic_coverage=topic_coverage,
    )


# ============================================================================
# Vote Routes
# ============================================================================


@router.post("/users/{user_id}/votes", response_model=UserVote, status_code=status.HTTP_201_CREATED)
async def create_vote(
    user_id: UUID,
    vote: UserVoteCreate,
    session: SessionDep,
):
    """
    Create a vote for an article.

    Supports upvotes, downvotes, and bookmarks.
    """
    # Ensure user exists
    user_result = await session.execute(
        select(DBUser).where(DBUser.id == str(user_id))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        user = DBUser(id=str(user_id))
        session.add(user)
        await session.flush()

    # Check if article exists in DB (for mock, we skip this check)
    # In production, would verify article exists

    # Check for existing vote of same type
    existing_result = await session.execute(
        select(DBVote).where(
            DBVote.user_id == str(user_id),
            DBVote.article_id == str(vote.article_id),
            DBVote.vote_type == vote.vote_type.value,
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User already has a {vote.vote_type.value} on this article",
        )

    # Create the vote
    db_vote = DBVote(
        user_id=str(user_id),
        article_id=str(vote.article_id),
        vote_type=vote.vote_type.value,
    )
    session.add(db_vote)

    # Update vote stats
    stats_result = await session.execute(
        select(DBArticleVoteStats).where(
            DBArticleVoteStats.article_id == str(vote.article_id)
        )
    )
    stats = stats_result.scalar_one_or_none()

    if not stats:
        stats = DBArticleVoteStats(article_id=str(vote.article_id))
        session.add(stats)

    if vote.vote_type == VoteType.UPVOTE:
        stats.upvote_count += 1
    elif vote.vote_type == VoteType.DOWNVOTE:
        stats.downvote_count += 1
    elif vote.vote_type == VoteType.BOOKMARK:
        stats.bookmark_count += 1

    stats.total_votes += 1
    stats.last_vote_at = datetime.utcnow()

    await session.commit()
    await session.refresh(db_vote)

    logger.info(
        "Created vote",
        user_id=str(user_id),
        article_id=str(vote.article_id),
        vote_type=vote.vote_type.value,
    )

    return UserVote(
        id=UUID(db_vote.id),
        user_id=user_id,
        article_id=vote.article_id,
        vote_type=vote.vote_type,
        created_at=db_vote.created_at,
    )


@router.get("/users/{user_id}/votes", response_model=list[UserVote])
async def get_user_votes(
    user_id: UUID,
    session: SessionDep,
    vote_type: VoteType | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """
    Get a user's votes, optionally filtered by type.
    """
    query = select(DBVote).where(DBVote.user_id == str(user_id))

    if vote_type:
        query = query.where(DBVote.vote_type == vote_type.value)

    query = query.order_by(DBVote.created_at.desc()).limit(limit).offset(offset)

    result = await session.execute(query)
    votes = result.scalars().all()

    return [
        UserVote(
            id=UUID(v.id),
            user_id=user_id,
            article_id=UUID(v.article_id),
            vote_type=VoteType(v.vote_type),
            created_at=v.created_at,
        )
        for v in votes
    ]


# ============================================================================
# User Stats Routes
# ============================================================================


@router.get("/users/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: UUID,
    session: SessionDep,
):
    """
    Get statistics about a user's engagement.
    """
    # Get user
    user_result = await session.execute(
        select(DBUser).where(DBUser.id == str(user_id))
    )
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Count votes by type
    vote_counts = await session.execute(
        select(
            DBVote.vote_type,
            func.count(DBVote.id).label("count"),
        )
        .where(DBVote.user_id == str(user_id))
        .group_by(DBVote.vote_type)
    )
    counts_by_type = {row[0]: row[1] for row in vote_counts.fetchall()}

    # Count clicks
    clicks_result = await session.execute(
        select(func.count(DBClick.id)).where(DBClick.user_id == str(user_id))
    )
    click_count = clicks_result.scalar() or 0

    # Get favorite topics (from votes)
    # In production, would aggregate topics from voted articles
    favorite_topics: list[TopicPath] = []

    return UserStats(
        user_id=user_id,
        total_votes=sum(counts_by_type.values()),
        upvotes=counts_by_type.get("upvote", 0),
        downvotes=counts_by_type.get("downvote", 0),
        bookmarks=counts_by_type.get("bookmark", 0),
        articles_read=click_count,
        favorite_topics=favorite_topics,
        join_date=user.created_at,
        last_active=user.last_active_at,
    )


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": get_settings().app_version,
    }
