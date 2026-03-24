"""
FastAPI routes for The Noiseless Newspaper API.

Authentication
--------------
All ``/users/me/*`` endpoints require a valid Supabase JWT sent as:
    Authorization: Bearer <token>

In development (ENVIRONMENT=development, no SUPABASE_JWT_SECRET set) you may
instead pass:
    X-Debug-User-ID: your-test-user-id

Public endpoints (taxonomy) require no authentication.

Admin endpoints require:
    X-Admin-API-Key: <ADMIN_API_KEY>
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user_id, get_optional_user_id, require_admin
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
    vote_due: Optional[str] = None  # Period for which a vote is due


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
    suggestions: list[dict]  # List of topic info with relevance score


# =============================================================================
# Database dependency injection
# =============================================================================

_database: Optional[Database] = None


def set_database(db: Database) -> None:
    """Set the global database instance (called from main.py on startup)."""
    global _database
    _database = db


async def get_db() -> AsyncSession:
    """Yield an async database session."""
    if _database is None:
        raise HTTPException(status_code=500, detail="Database not initialised")
    async with _database.async_session() as session:
        yield session


async def get_or_create_user(session: AsyncSession, user_id: str) -> DBUser:
    """Return an existing user or create one on first encounter."""
    result = await session.execute(select(DBUser).where(DBUser.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        user = DBUser(id=user_id)
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user


# =============================================================================
# Public routes – no authentication required
# =============================================================================

@router.get("/taxonomy", tags=["taxonomy"])
async def get_taxonomy():
    """Return the full topic taxonomy for frontend rendering."""
    return get_taxonomy_for_frontend()


@router.get("/taxonomy/paths", tags=["taxonomy"])
async def get_topic_paths():
    """Return all valid topic paths."""
    return {"paths": get_all_topic_paths()}


@router.get("/taxonomy/{path:path}", tags=["taxonomy"])
async def get_topic(path: str):
    """Return information about a specific topic path."""
    info = get_topic_info(path)
    if not info:
        raise HTTPException(status_code=404, detail="Topic not found")
    return info


# =============================================================================
# User preference routes  –  require authentication
# =============================================================================

@router.get(
    "/users/me/preferences",
    response_model=UserPreferencesResponse,
    tags=["users"],
    summary="Get my topic preferences",
)
async def get_my_preferences(
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's topic preferences."""
    await get_or_create_user(session, user_id)

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
        topic_frequencies=(
            json.loads(prefs.topic_frequencies_json) if prefs.topic_frequencies_json else {}
        ),
        created_at=prefs.created_at,
        updated_at=prefs.updated_at,
    )


@router.put(
    "/users/me/preferences",
    response_model=UserPreferencesResponse,
    tags=["users"],
    summary="Update my topic preferences",
)
async def update_my_preferences(
    request: UserPreferencesRequest,
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Replace the authenticated user's topic preferences."""
    import json

    await get_or_create_user(session, user_id)

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
# Article routes  –  require authentication
# =============================================================================

@router.get(
    "/users/me/daily-article",
    response_model=DailyArticleResponse,
    tags=["articles"],
    summary="Get today's article for a topic",
)
async def get_my_daily_article(
    topic_path: str = Query(..., description="Topic path for today's article"),
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """
    Return the single best article for the authenticated user on a given topic today.

    This is the core endpoint – one article per day, chosen by the ranking algorithm.
    """
    if topic_path not in get_all_topic_paths():
        raise HTTPException(status_code=400, detail=f"Invalid topic path: {topic_path}")

    await get_or_create_user(session, user_id)

    # Articles already read today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await session.execute(
        select(DBClick.article_id)
        .where(DBClick.user_id == user_id)
        .where(DBClick.clicked_at >= today_start)
    )
    read_today = {row[0] for row in result.all()}

    # Top candidates for this topic
    result = await session.execute(
        select(DBArticle)
        .where(DBArticle.topic_path == topic_path)
        .order_by(DBArticle.final_score.desc())
        .limit(10)
    )
    candidates = result.scalars().all()

    article = None
    for candidate in candidates:
        if candidate.id not in read_today:
            article = candidate
            break

    if not article:
        if candidates:
            article = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="No articles available for this topic")

    vote_due = await _check_pending_votes(session, user_id, article.id)

    click = DBClick(user_id=user_id, article_id=article.id, topic_path=topic_path)
    session.add(click)
    await session.commit()

    return DailyArticleResponse(
        article=_format_article(article),
        already_read=article.id in read_today,
        vote_due=vote_due,
    )


@router.get(
    "/users/me/suggestions",
    response_model=SmartSuggestionsResponse,
    tags=["articles"],
    summary="Get smart topic suggestions",
)
async def get_my_suggestions(
    limit: int = Query(default=4, le=10),
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """
    Return personalised topic suggestions based on the user's reading history
    and declared topic frequencies.
    """
    import json

    await get_or_create_user(session, user_id)

    result = await session.execute(
        select(DBUserPreferences).where(DBUserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()

    if not prefs or not prefs.selected_topics_json:
        return SmartSuggestionsResponse(suggestions=[])

    selected_topics = json.loads(prefs.selected_topics_json)
    frequencies = (
        json.loads(prefs.topic_frequencies_json) if prefs.topic_frequencies_json else {}
    )

    result = await session.execute(
        select(DBClick.topic_path)
        .where(DBClick.user_id == user_id)
        .order_by(DBClick.clicked_at.desc())
        .limit(100)
    )
    click_history = [row[0] for row in result.all()]

    topic_scores: dict[str, float] = {}
    for path in selected_topics:
        parent_path = "/".join(path.split("/")[:2])
        click_count = sum(1 for c in click_history if c.startswith(parent_path))
        freq = frequencies.get(parent_path, "medium")
        freq_mult = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(freq, 1.0)

        import random
        topic_scores[path] = (click_count + 1) * freq_mult + random.random() * 0.5

    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    suggestions = []
    for path, score in sorted_topics[:limit]:
        info = get_topic_info(path)
        if info:
            suggestions.append({**info, "relevance_score": score})

    return SmartSuggestionsResponse(suggestions=suggestions)


# =============================================================================
# Voting routes  –  require authentication
# =============================================================================

@router.post(
    "/users/me/votes",
    response_model=VoteResponse,
    tags=["votes"],
    summary="Submit a relevance vote",
)
async def submit_my_vote(
    request: VoteRequest,
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """
    Submit a vote on an article's relevance.

    Votes are cast at three time periods (1 week, 1 month, 1 year) after
    first reading the article.  Later votes carry more weight in the
    ranking algorithm.
    """
    await get_or_create_user(session, user_id)

    try:
        period = VotePeriod(request.period)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid period: {request.period}")

    result = await session.execute(
        select(DBArticle).where(DBArticle.id == request.article_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Article not found")

    result = await session.execute(
        select(DBVote)
        .where(DBVote.user_id == user_id)
        .where(DBVote.article_id == request.article_id)
        .where(DBVote.period == period.value)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.score = request.score
        existing.voted_at = datetime.utcnow()
        message = "Vote updated"
    else:
        session.add(
            DBVote(
                user_id=user_id,
                article_id=request.article_id,
                period=period.value,
                score=request.score,
            )
        )
        message = "Vote recorded"

    await session.commit()
    return VoteResponse(success=True, message=message)


@router.get(
    "/users/me/pending-votes",
    tags=["votes"],
    summary="Get articles awaiting a relevance vote",
)
async def get_my_pending_votes(
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """
    Return articles that need a vote based on how long ago they were read.

    Checks for articles read approximately:
    - 1 week ago (needs 1_week vote)
    - 1 month ago (needs 1_month vote)
    - 1 year ago (needs 1_year vote)
    """
    from datetime import timedelta

    await get_or_create_user(session, user_id)

    now = datetime.utcnow()
    pending = []

    windows = [
        ("1_week",  timedelta(days=6),   timedelta(days=10)),
        ("1_month", timedelta(days=28),  timedelta(days=35)),
        ("1_year",  timedelta(days=360), timedelta(days=400)),
    ]

    for period, min_age, max_age in windows:
        result = await session.execute(
            select(DBClick)
            .where(DBClick.user_id == user_id)
            .where(DBClick.clicked_at >= now - max_age)
            .where(DBClick.clicked_at <= now - min_age)
        )
        for click in result.scalars().all():
            result2 = await session.execute(
                select(DBVote)
                .where(DBVote.user_id == user_id)
                .where(DBVote.article_id == click.article_id)
                .where(DBVote.period == period)
            )
            if result2.scalar_one_or_none():
                continue  # already voted

            result3 = await session.execute(
                select(DBArticle).where(DBArticle.id == click.article_id)
            )
            article = result3.scalar_one_or_none()
            if article:
                pending.append(
                    {
                        "article": _format_article(article),
                        "period": period,
                        "read_at": click.clicked_at.isoformat(),
                    }
                )

    return {"pending_votes": pending}


# =============================================================================
# Stats routes  –  require authentication
# =============================================================================

@router.get(
    "/users/me/stats",
    response_model=UserStatsResponse,
    tags=["users"],
    summary="Get my reading stats and signal score",
)
async def get_my_stats(
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's reading statistics and signal score."""
    from sqlalchemy import func

    await get_or_create_user(session, user_id)

    result = await session.execute(
        select(DBUserSignalScore).where(DBUserSignalScore.user_id == user_id)
    )
    signal_record = result.scalar_one_or_none()
    signal_score = signal_record.correlation_score if signal_record else 0.0

    result = await session.execute(
        select(func.count(DBClick.id)).where(DBClick.user_id == user_id)
    )
    total_reads = result.scalar() or 0

    result = await session.execute(
        select(func.count(DBVote.id)).where(DBVote.user_id == user_id)
    )
    total_votes = result.scalar() or 0

    votes_by_period: dict[str, int] = {}
    for period in VotePeriod:
        result = await session.execute(
            select(func.count(DBVote.id))
            .where(DBVote.user_id == user_id)
            .where(DBVote.period == period.value)
        )
        votes_by_period[period.value] = result.scalar() or 0

    result = await session.execute(
        select(DBClick, DBArticle)
        .join(DBArticle, DBClick.article_id == DBArticle.id)
        .where(DBClick.user_id == user_id)
        .order_by(DBClick.clicked_at.desc())
        .limit(10)
    )
    recent_reads = [
        {
            "article_id": click.article_id,
            "title": article.title,
            "topic_path": click.topic_path,
            "read_at": click.clicked_at.isoformat(),
        }
        for click, article in result.all()
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
# Admin routes  –  require X-Admin-API-Key header
# =============================================================================

@router.post(
    "/admin/run-ingestion",
    tags=["admin"],
    summary="Manually trigger article ingestion",
    dependencies=[Depends(require_admin)],
)
async def admin_run_ingestion():
    """
    Manually trigger the daily article ingestion job.

    This was previously a development-only endpoint in main.py.  It is now
    protected by ``X-Admin-API-Key`` so it can safely be exposed in production
    for on-demand refreshes.
    """
    import asyncio
    from app.jobs.daily_ingestion import DailyIngestionJob

    # Import lazily to avoid circular imports at module load time
    from app.api.routes import _database as db

    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")

    job = DailyIngestionJob(db)
    await job.initialize()
    asyncio.create_task(job.run())
    return {"message": "Ingestion job started"}


# =============================================================================
# Helper functions
# =============================================================================

def _format_article(db_article: DBArticle) -> ArticleResponse:
    """Convert a DBArticle row to an ArticleResponse."""
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
    """Return the earliest pending vote period for this article, or None."""
    from datetime import timedelta

    now = datetime.utcnow()

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
    periods = [
        ("1_week",  timedelta(days=7)),
        ("1_month", timedelta(days=30)),
        ("1_year",  timedelta(days=365)),
    ]

    for period_name, threshold in periods:
        if age >= threshold:
            result = await session.execute(
                select(DBVote)
                .where(DBVote.user_id == user_id)
                .where(DBVote.article_id == article_id)
                .where(DBVote.period == period_name)
            )
            if not result.scalar_one_or_none():
                return period_name

    return None
