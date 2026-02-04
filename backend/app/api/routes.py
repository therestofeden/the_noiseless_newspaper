"""
FastAPI routes for The Noiseless Newspaper API.

These endpoints power the frontend application:
- Topic taxonomy retrieval
- User preferences management
- Daily article selection
- Voting on article relevance
- User reading history and signal score
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.taxonomy import get_all_topic_paths, get_taxonomy_for_frontend, get_topic_info
from app.models.database import (
    DBArticle,
    DBClick,
    DBUser,
    DBUserPreferences,
    DBUserSignalScore,
    DBVote,
    Database,
)
from app.models.domain import TopicPath, VotePeriod

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class UserPreferencesRequest(BaseModel):
    """Request body for updating user preferences."""
    selected_topics: list[str] = Field(default_factory=list)
    topic_frequencies: dict[str, str] = Field(default_factory=dict)


class UserPreferencesResponse(BaseModel):
    """Response body for user preferences."""
    user_id: str
    selected_topics: list[str]
    topic_frequencies: dict[str, str]
    created_at: datetime
    updated_at: datetime


class ArticleResponse(BaseModel):
    """Response body for an article."""
    id: str
    title: str
    summary: Optional[str]
    abstract: Optional[str]
    source: str
    url: str
    published_at: datetime
    topic_path: Optional[str]
    topic_display: Optional[dict]
    citation_count: int
    final_score: float
    score_breakdown: Optional[dict]


class DailyArticleResponse(BaseModel):
    """Response body for the daily article."""
    article: ArticleResponse
    already_read: bool = False
    vote_due: Optional[str] = None  # Period for which vote is due


class VoteRequest(BaseModel):
    """Request body for submitting a vote."""
    article_id: str
    period: str  # "1_week", "1_month", "1_year"
    score: int = Field(ge=1, le=5)


class VoteResponse(BaseModel):
    """Response body for a submitted vote."""
    success: bool
    message: str


class UserStatsResponse(BaseModel):
    """Response body for user statistics."""
    user_id: str
    signal_score: float
    total_articles_read: int
    total_votes: int
    votes_by_period: dict[str, int]
    recent_reads: list[dict]


class SmartSuggestionsResponse(BaseModel):
    """Response body for smart topic suggestions."""
    suggestions: list[dict]  # List of topic info with relevance


# =============================================================================
# Dependency Injection
# =============================================================================

# Database instance (initialized in main.py)
_database: Optional[Database] = None


def set_database(db: Database):
    """Set the database instance for dependency injection."""
    global _database
    _database = db


async def get_db() -> AsyncSession:
    """Get database session."""
    if _database is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    async with _database.async_session() as session:
        yield session


async def get_or_create_user(session: AsyncSession, user_id: str) -> DBUser:
    """Get or create a user."""
    result = await session.execute(select(DBUser).where(DBUser.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        user = DBUser(id=user_id)
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user


# =============================================================================
# Taxonomy Routes
# =============================================================================

@router.get("/taxonomy")
async def get_taxonomy():
    """Get the full topic taxonomy for frontend rendering."""
    return get_taxonomy_for_frontend()


@router.get("/taxonomy/paths")
async def get_topic_paths():
    """Get all valid topic paths."""
    return {"paths": get_all_topic_paths()}


@router.get("/taxonomy/{path:path}")
async def get_topic(path: str):
    """Get information about a specific topic."""
    info = get_topic_info(path)
    if not info:
        raise HTTPException(status_code=404, detail="Topic not found")
    return info


# =============================================================================
# User Preferences Routes
# =============================================================================

@router.get("/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get user's topic preferences."""
    user = await get_or_create_user(session, user_id)

    result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()

    if not prefs:
        return UserPreferencesResponse(
            user_id=user_id,
            selected_topics=[],
            topic_frequencies={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    import json
    return UserPreferencesResponse(
        user_id=user_id,
        selected_topics=json.loads(prefs.selected_topics_json) if prefs.selected_topics_json else [],
        topic_frequencies=json.loads(prefs.topic_frequencies_json) if prefs.topic_frequencies_json else {},
        created_at=prefs.created_at,
        updated_at=prefs.updated_at,
    )


@router.put("/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    user_id: str,
    request: UserPreferencesRequest,
    session: AsyncSession = Depends(get_db),
):
    """Update user's topic preferences."""
    import json

    user = await get_or_create_user(session, user_id)

    # Validate topic paths
    valid_paths = set(get_all_topic_paths())
    for path in request.selected_topics:
        if path not in valid_paths:
            raise HTTPException(status_code=400, detail=f"Invalid topic path: {path}")

    result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()

    if prefs:
        prefs.selected_topics_json = json.dumps(request.selected_topics)
        prefs.topic_frequencies_json = json.dumps(request.topic_frequencies)
        prefs.updated_at = datetime.utcnow()
    else:
        prefs = DBUserPreferences(
            user_id=user_id,
            selected_topics_json=json.dumps(request.selected_topics),
            topic_frequencies_json=json.dumps(request.topic_frequencies),
        )
        session.add(prefs)

    await session.commit()
    await session.refresh(prefs)

    return UserPreferencesResponse(
        user_id=user_id,
        selected_topics=request.selected_topics,
        topic_frequencies=request.topic_frequencies,
        created_at=prefs.created_at,
        updated_at=prefs.updated_at,
    )


# =============================================================================
# Article Routes
# =============================================================================

@router.get("/users/{user_id}/daily-article", response_model=DailyArticleResponse)
async def get_daily_article(
    user_id: str,
    topic_path: str = Query(..., description="Topic path for today's article"),
    session: AsyncSession = Depends(get_db),
):
    """
    Get the single best article for a user on a given topic today.

    This is the core endpoint - one article per day, chosen by
    the ranking algorithm.
    """
    # Validate topic path
    if topic_path not in get_all_topic_paths():
        raise HTTPException(status_code=400, detail=f"Invalid topic path: {topic_path}")

    user = await get_or_create_user(session, user_id)

    # Get articles the user has already read today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await session.execute(
        select(DBClick.article_id)
        .where(DBClick.user_id == user_id)
        .where(DBClick.clicked_at >= today_start)
    )
    read_today = {row[0] for row in result.all()}

    # Get top article for this topic (excluding already read)
    query = (
        select(DBArticle)
        .where(DBArticle.topic_path == topic_path)
        .order_by(DBArticle.final_score.desc())
        .limit(10)
    )
    result = await session.execute(query)
    candidates = result.scalars().all()

    # Find first unread article
    article = None
    for candidate in candidates:
        if candidate.id not in read_today:
            article = candidate
            break

    if not article:
        # All articles read, return top one anyway
        if candidates:
            article = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="No articles available for this topic")

    # Check if user has pending votes
    vote_due = await _check_pending_votes(session, user_id, article.id)

    # Record the click
    click = DBClick(
        user_id=user_id,
        article_id=article.id,
        topic_path=topic_path,
    )
    session.add(click)
    await session.commit()

    return DailyArticleResponse(
        article=_format_article(article),
        already_read=article.id in read_today,
        vote_due=vote_due,
    )


@router.get("/users/{user_id}/suggestions", response_model=SmartSuggestionsResponse)
async def get_smart_suggestions(
    user_id: str,
    limit: int = Query(default=4, le=10),
    session: AsyncSession = Depends(get_db),
):
    """
    Get smart topic suggestions based on user's reading history.

    Uses click history and topic frequencies to rank suggestions.
    """
    import json

    user = await get_or_create_user(session, user_id)

    # Get user preferences
    result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()

    if not prefs or not prefs.selected_topics_json:
        return SmartSuggestionsResponse(suggestions=[])

    selected_topics = json.loads(prefs.selected_topics_json)
    frequencies = json.loads(prefs.topic_frequencies_json) if prefs.topic_frequencies_json else {}

    # Get click history
    result = await session.execute(
        select(DBClick.topic_path)
        .where(DBClick.user_id == user_id)
        .order_by(DBClick.clicked_at.desc())
        .limit(100)
    )
    click_history = [row[0] for row in result.all()]

    # Score each selected topic
    topic_scores = {}
    for topic_path in selected_topics:
        # Count clicks for this topic's parent
        parent_path = "/".join(topic_path.split("/")[:2])
        click_count = sum(1 for c in click_history if c.startswith(parent_path))

        # Apply frequency multiplier
        freq = frequencies.get(parent_path, "medium")
        freq_mult = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(freq, 1.0)

        # Add some randomness for variety
        import random
        topic_scores[topic_path] = (click_count + 1) * freq_mult + random.random() * 0.5

    # Sort and take top suggestions
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    top_topics = sorted_topics[:limit]

    suggestions = []
    for path, score in top_topics:
        info = get_topic_info(path)
        if info:
            suggestions.append({
                **info,
                "relevance_score": score,
            })

    return SmartSuggestionsResponse(suggestions=suggestions)


# =============================================================================
# Voting Routes
# =============================================================================

@router.post("/users/{user_id}/votes", response_model=VoteResponse)
async def submit_vote(
    user_id: str,
    request: VoteRequest,
    session: AsyncSession = Depends(get_db),
):
    """
    Submit a vote on an article's relevance.

    Votes are cast at different time periods (1 week, 1 month, 1 year)
    after reading the article.
    """
    user = await get_or_create_user(session, user_id)

    # Validate period
    try:
        period = VotePeriod(request.period)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid period: {request.period}")

    # Check if article exists
    result = await session.execute(
        select(DBArticle).where(DBArticle.id == request.article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    # Check for existing vote
    result = await session.execute(
        select(DBVote)
        .where(DBVote.user_id == user_id)
        .where(DBVote.article_id == request.article_id)
        .where(DBVote.period == period.value)
    )
    existing_vote = result.scalar_one_or_none()

    if existing_vote:
        # Update existing vote
        existing_vote.score = request.score
        existing_vote.voted_at = datetime.utcnow()
        message = "Vote updated"
    else:
        # Create new vote
        vote = DBVote(
            user_id=user_id,
            article_id=request.article_id,
            period=period.value,
            score=request.score,
        )
        session.add(vote)
        message = "Vote recorded"

    await session.commit()

    return VoteResponse(success=True, message=message)


@router.get("/users/{user_id}/pending-votes")
async def get_pending_votes(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    """
    Get articles that need votes based on time elapsed since reading.

    Returns articles read:
    - ~1 week ago (needs 1_week vote)
    - ~1 month ago (needs 1_month vote)
    - ~1 year ago (needs 1_year vote)
    """
    from datetime import timedelta

    user = await get_or_create_user(session, user_id)

    now = datetime.utcnow()
    pending = []

    # Define vote windows (with some tolerance)
    windows = [
        ("1_week", timedelta(days=6), timedelta(days=10)),
        ("1_month", timedelta(days=28), timedelta(days=35)),
        ("1_year", timedelta(days=360), timedelta(days=400)),
    ]

    for period, min_age, max_age in windows:
        # Find clicks in this window
        result = await session.execute(
            select(DBClick)
            .where(DBClick.user_id == user_id)
            .where(DBClick.clicked_at >= now - max_age)
            .where(DBClick.clicked_at <= now - min_age)
        )
        clicks = result.scalars().all()

        for click in clicks:
            # Check if already voted for this period
            result = await session.execute(
                select(DBVote)
                .where(DBVote.user_id == user_id)
                .where(DBVote.article_id == click.article_id)
                .where(DBVote.period == period)
            )
            existing_vote = result.scalar_one_or_none()

            if not existing_vote:
                # Get article details
                result = await session.execute(
                    select(DBArticle).where(DBArticle.id == click.article_id)
                )
                article = result.scalar_one_or_none()

                if article:
                    pending.append({
                        "article": _format_article(article),
                        "period": period,
                        "read_at": click.clicked_at.isoformat(),
                    })

    return {"pending_votes": pending}


# =============================================================================
# User Stats Routes
# =============================================================================

@router.get("/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get user's reading statistics and signal score."""
    from sqlalchemy import func

    user = await get_or_create_user(session, user_id)

    # Get signal score
    result = await session.execute(
        select(DBUserSignalScore).where(DBUserSignalScore.user_id == user_id)
    )
    signal_score_record = result.scalar_one_or_none()
    signal_score = signal_score_record.correlation_score if signal_score_record else 0.0

    # Count total articles read
    result = await session.execute(
        select(func.count(DBClick.id)).where(DBClick.user_id == user_id)
    )
    total_reads = result.scalar() or 0

    # Count total votes
    result = await session.execute(
        select(func.count(DBVote.id)).where(DBVote.user_id == user_id)
    )
    total_votes = result.scalar() or 0

    # Count votes by period
    votes_by_period = {}
    for period in VotePeriod:
        result = await session.execute(
            select(func.count(DBVote.id))
            .where(DBVote.user_id == user_id)
            .where(DBVote.period == period.value)
        )
        votes_by_period[period.value] = result.scalar() or 0

    # Get recent reads
    result = await session.execute(
        select(DBClick, DBArticle)
        .join(DBArticle, DBClick.article_id == DBArticle.id)
        .where(DBClick.user_id == user_id)
        .order_by(DBClick.clicked_at.desc())
        .limit(10)
    )
    recent = result.all()

    recent_reads = [
        {
            "article_id": click.article_id,
            "title": article.title,
            "topic_path": click.topic_path,
            "read_at": click.clicked_at.isoformat(),
        }
        for click, article in recent
    ]

    return UserStatsResponse(
        user_id=user_id,
        signal_score=signal_score,
        total_articles_read=total_reads,
        total_votes=total_votes,
        votes_by_period=votes_by_period,
        recent_reads=recent_reads,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _format_article(db_article: DBArticle) -> ArticleResponse:
    """Format a database article for API response."""
    topic_display = None
    if db_article.topic_path:
        info = get_topic_info(db_article.topic_path)
        if info:
            topic_display = {
                "icon": info["category"]["icon"],
                "category": info["category"]["name"],
                "subtopic": info["subtopic"]["name"],
                "niche": info["niche"]["name"],
            }

    return ArticleResponse(
        id=db_article.id,
        title=db_article.title,
        summary=db_article.summary,
        abstract=db_article.abstract,
        source=db_article.source,
        url=db_article.url,
        published_at=db_article.published_at,
        topic_path=db_article.topic_path,
        topic_display=topic_display,
        citation_count=db_article.citation_count,
        final_score=db_article.final_score,
        score_breakdown={
            "pagerank": db_article.pagerank_score,
            "recency": db_article.recency_score,
            "topic_relevance": db_article.topic_relevance_score,
        },
    )


async def _check_pending_votes(
    session: AsyncSession,
    user_id: str,
    article_id: str,
) -> Optional[str]:
    """Check if user has a pending vote for this article."""
    from datetime import timedelta

    now = datetime.utcnow()

    # Get when user read this article
    result = await session.execute(
        select(DBClick)
        .where(DBClick.user_id == user_id)
        .where(DBClick.article_id == article_id)
        .order_by(DBClick.clicked_at.asc())
        .limit(1)
    )
    click = result.scalar_one_or_none()

    if not click:
        return None

    age = now - click.clicked_at

    # Check each period
    periods = [
        ("1_week", timedelta(days=7)),
        ("1_month", timedelta(days=30)),
        ("1_year", timedelta(days=365)),
    ]

    for period_name, threshold in periods:
        if age >= threshold:
            # Check if already voted
            result = await session.execute(
                select(DBVote)
                .where(DBVote.user_id == user_id)
                .where(DBVote.article_id == article_id)
                .where(DBVote.period == period_name)
            )
            if not result.scalar_one_or_none():
                return period_name

    return None
