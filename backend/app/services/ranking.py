"""
Ranking service - the core algorithm combining all signals.

This is where the magic happens:
1. Citation-based PageRank score (cold start)
2. Recency score (time decay)
3. Topic relevance (embedding similarity)
4. User votes weighted by time period (1 week < 1 month < 1 year)

The key insight: votes cast after longer periods carry more weight
because they reflect enduring relevance, not hype.
"""
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import DBArticle, DBArticleVoteStats, DBVote
from app.models.domain import Article, ArticleRanking, TopicPath, VotePeriod
from app.services.citation_graph import CitationGraphService
from app.services.embeddings import EmbeddingService, TopicEmbeddingService


class RankingService:
    """
    Main ranking service that combines all signals into a final score.

    The ranking formula:
        FinalScore = w_c * CitationScore
                   + w_r * RecencyScore
                   + w_t * TopicRelevanceScore
                   + w_v * UserVoteScore

    Where weights are configurable and sum to 1.0.

    As an article accumulates more votes, we gradually shift weight
    from citation-based scoring to vote-based scoring (λ transition).
    """

    def __init__(
        self,
        citation_service: CitationGraphService,
        embedding_service: EmbeddingService,
        topic_embedding_service: TopicEmbeddingService,
    ):
        self.settings = get_settings()
        self.citation_service = citation_service
        self.embedding_service = embedding_service
        self.topic_embedding_service = topic_embedding_service

    def compute_recency_score(
        self,
        published_at: datetime,
        half_life_days: float = 30.0,
    ) -> float:
        """
        Compute recency score with exponential decay.

        Uses half-life model: score halves every `half_life_days`.
        - Published today: score ≈ 1.0
        - Published 30 days ago: score ≈ 0.5
        - Published 60 days ago: score ≈ 0.25

        Args:
            published_at: Publication datetime
            half_life_days: Days until score halves

        Returns:
            Score in range (0, 1]
        """
        age_days = (datetime.utcnow() - published_at).total_seconds() / 86400
        decay_rate = math.log(2) / half_life_days
        return math.exp(-decay_rate * max(0, age_days))

    def compute_vote_score(
        self,
        votes_1_week: int,
        avg_score_1_week: float,
        votes_1_month: int,
        avg_score_1_month: float,
        votes_1_year: int,
        avg_score_1_year: float,
    ) -> float:
        """
        Compute weighted vote score across time periods.

        Later votes count more because they reflect lasting relevance.

        The weighting:
        - 1 week votes: 15% weight
        - 1 month votes: 35% weight
        - 1 year votes: 50% weight

        Returns:
            Score in range [0, 1] (normalized from 1-5 scale)
        """
        w_week = self.settings.vote_weight_1_week
        w_month = self.settings.vote_weight_1_month
        w_year = self.settings.vote_weight_1_year

        # Weighted average of scores (if we have votes)
        total_weight = 0
        weighted_sum = 0

        if votes_1_week > 0:
            weighted_sum += w_week * avg_score_1_week
            total_weight += w_week

        if votes_1_month > 0:
            weighted_sum += w_month * avg_score_1_month
            total_weight += w_month

        if votes_1_year > 0:
            weighted_sum += w_year * avg_score_1_year
            total_weight += w_year

        if total_weight == 0:
            return 0.0

        # Average score is on 1-5 scale, normalize to 0-1
        avg = weighted_sum / total_weight
        return (avg - 1) / 4  # Maps 1-5 to 0-1

    def compute_lambda(
        self,
        total_votes: int,
        threshold: int = 10,
    ) -> float:
        """
        Compute the transition factor λ between citation-based and vote-based scoring.

        As articles accumulate votes, we trust user feedback more than citations.

        Uses sigmoid function:
        - Few votes (< threshold): λ ≈ 0 (rely on citations)
        - Many votes (> threshold): λ ≈ 1 (rely on votes)

        Args:
            total_votes: Total number of votes on the article
            threshold: Vote count at which λ = 0.5

        Returns:
            λ in range [0, 1]
        """
        # Sigmoid centered at threshold
        return 1 / (1 + math.exp(-(total_votes - threshold) / 3))

    async def compute_topic_relevance(
        self,
        article_text: str,
        target_topic: TopicPath,
    ) -> float:
        """
        Compute how relevant an article is to a specific topic.

        Uses cosine similarity between article embedding and topic embedding.

        Returns:
            Score in range [-1, 1], typically [0, 1] for related content
        """
        topic_embedding = self.topic_embedding_service.get_topic_embedding(target_topic.path)
        if topic_embedding is None:
            return 0.0

        article_embedding = await self.embedding_service.embed_text(article_text)
        similarity = EmbeddingService.cosine_similarity(article_embedding, topic_embedding)

        # Normalize to [0, 1] range (similarity is typically positive for related content)
        return max(0, similarity)

    async def rank_article(
        self,
        article: Article,
        target_topic: TopicPath,
        vote_stats: Optional[dict] = None,
    ) -> ArticleRanking:
        """
        Compute the full ranking for a single article.

        Args:
            article: The article to rank
            target_topic: The topic the user is interested in
            vote_stats: Pre-computed vote statistics (optional)

        Returns:
            ArticleRanking with score breakdown
        """
        # 1. Citation score (PageRank)
        citation_score = self.citation_service.get_score(article.id)
        if citation_score == 0 and article.citation_count > 0:
            # Fallback: use log-normalized citation count
            citation_score = math.log1p(article.citation_count) / 10

        # 2. Recency score
        recency_score = self.compute_recency_score(article.published_at)

        # 3. Topic relevance
        article_text = f"{article.title} {article.abstract or ''}"
        topic_relevance = await self.compute_topic_relevance(article_text, target_topic)

        # 4. User vote score
        if vote_stats:
            user_vote_score = self.compute_vote_score(
                vote_stats.get("votes_1_week", 0),
                vote_stats.get("avg_score_1_week", 0),
                vote_stats.get("votes_1_month", 0),
                vote_stats.get("avg_score_1_month", 0),
                vote_stats.get("votes_1_year", 0),
                vote_stats.get("avg_score_1_year", 0),
            )
            total_votes = (
                vote_stats.get("votes_1_week", 0)
                + vote_stats.get("votes_1_month", 0)
                + vote_stats.get("votes_1_year", 0)
            )
        else:
            user_vote_score = 0.0
            total_votes = 0

        # Compute λ (transition from citations to votes)
        lambda_factor = self.compute_lambda(total_votes)

        # Base weights from config
        w_citation = self.settings.weight_citation_score
        w_recency = self.settings.weight_recency
        w_topic = self.settings.weight_topic_relevance
        w_votes = self.settings.weight_user_votes

        # Adjust weights based on λ
        # As λ increases, shift weight from citations to votes
        adjusted_w_citation = w_citation * (1 - lambda_factor * 0.5)
        adjusted_w_votes = w_votes + (w_citation * lambda_factor * 0.5)

        # Compute final score
        final_score = (
            adjusted_w_citation * citation_score
            + w_recency * recency_score
            + w_topic * topic_relevance
            + adjusted_w_votes * user_vote_score
        )

        # Build explanation
        explanation_parts = [
            f"Citation: {citation_score:.3f} (w={adjusted_w_citation:.2f})",
            f"Recency: {recency_score:.3f} (w={w_recency:.2f})",
            f"Topic: {topic_relevance:.3f} (w={w_topic:.2f})",
            f"Votes: {user_vote_score:.3f} (w={adjusted_w_votes:.2f}, λ={lambda_factor:.2f})",
        ]

        return ArticleRanking(
            article=article,
            rank=0,  # Set later when sorting
            citation_score=citation_score,
            recency_score=recency_score,
            topic_relevance_score=topic_relevance,
            user_vote_score=user_vote_score,
            final_score=final_score,
            scoring_explanation=" | ".join(explanation_parts),
        )

    async def rank_articles(
        self,
        articles: list[Article],
        target_topic: TopicPath,
        session: AsyncSession,
        limit: int = 10,
    ) -> list[ArticleRanking]:
        """
        Rank a list of articles for a specific topic.

        Args:
            articles: Articles to rank
            target_topic: The topic the user is interested in
            session: Database session for fetching vote stats
            limit: Maximum number of results to return

        Returns:
            List of ArticleRanking objects, sorted by score descending
        """
        # Fetch vote stats for all articles
        article_ids = [a.id for a in articles]
        vote_stats_map = await self._fetch_vote_stats(session, article_ids)

        # Rank each article
        rankings = []
        for article in articles:
            vote_stats = vote_stats_map.get(article.id)
            ranking = await self.rank_article(article, target_topic, vote_stats)
            rankings.append(ranking)

        # Sort by final score descending
        rankings.sort(key=lambda r: r.final_score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings[:limit]

    async def _fetch_vote_stats(
        self,
        session: AsyncSession,
        article_ids: list[str],
    ) -> dict[str, dict]:
        """Fetch pre-computed vote statistics from database."""
        if not article_ids:
            return {}

        result = await session.execute(
            select(DBArticleVoteStats).where(DBArticleVoteStats.article_id.in_(article_ids))
        )
        stats = result.scalars().all()

        return {
            s.article_id: {
                "votes_1_week": s.votes_1_week,
                "avg_score_1_week": s.avg_score_1_week,
                "votes_1_month": s.votes_1_month,
                "avg_score_1_month": s.avg_score_1_month,
                "votes_1_year": s.votes_1_year,
                "avg_score_1_year": s.avg_score_1_year,
            }
            for s in stats
        }

    async def get_daily_article(
        self,
        user_id: str,
        topic: TopicPath,
        session: AsyncSession,
        exclude_article_ids: Optional[set[str]] = None,
    ) -> Optional[ArticleRanking]:
        """
        Get the single best article for a user on a given topic today.

        This is the core function for "one article per day".

        Args:
            user_id: The user requesting the article
            topic: The topic they want to read about
            session: Database session
            exclude_article_ids: Articles to exclude (e.g., already read)

        Returns:
            The highest-ranked article, or None if no articles available
        """
        exclude_ids = exclude_article_ids or set()

        # Query articles for this topic
        query = (
            select(DBArticle)
            .where(DBArticle.topic_path == topic.path)
            .where(DBArticle.id.notin_(exclude_ids) if exclude_ids else True)
            .order_by(DBArticle.final_score.desc())
            .limit(50)  # Get top candidates for re-ranking
        )

        result = await session.execute(query)
        db_articles = result.scalars().all()

        if not db_articles:
            return None

        # Convert to domain objects
        articles = [self._db_to_domain(db_article) for db_article in db_articles]

        # Rank with full algorithm
        rankings = await self.rank_articles(articles, topic, session, limit=1)

        return rankings[0] if rankings else None

    def _db_to_domain(self, db_article: DBArticle) -> Article:
        """Convert database model to domain model."""
        import json

        authors = []
        if db_article.authors_json:
            try:
                authors_data = json.loads(db_article.authors_json) if isinstance(db_article.authors_json, str) else db_article.authors_json
                from app.models.domain import Author
                authors = [Author(**a) for a in authors_data]
            except (json.JSONDecodeError, TypeError):
                pass

        return Article(
            id=db_article.id,
            source=db_article.source,
            content_type=db_article.content_type,
            title=db_article.title,
            abstract=db_article.abstract,
            summary=db_article.summary,
            authors=authors,
            published_at=db_article.published_at,
            url=db_article.url,
            topic_path=db_article.topic_path,
            topic_relevance_score=db_article.topic_relevance_score,
            citation_count=db_article.citation_count,
            pagerank_score=db_article.pagerank_score,
            recency_score=db_article.recency_score,
            final_score=db_article.final_score,
        )


class VoteAggregationService:
    """
    Service for aggregating votes and computing vote statistics.

    Run periodically (e.g., hourly) to update the materialized view
    of vote statistics used by the ranking algorithm.
    """

    def __init__(self):
        self.settings = get_settings()

    async def aggregate_votes(self, session: AsyncSession) -> int:
        """
        Aggregate all votes and update the vote statistics table.

        Returns:
            Number of articles updated
        """
        # Get all articles with votes
        result = await session.execute(
            select(DBVote.article_id).distinct()
        )
        article_ids = [row[0] for row in result.all()]

        updated = 0
        for article_id in article_ids:
            stats = await self._compute_article_stats(session, article_id)
            await self._upsert_stats(session, article_id, stats)
            updated += 1

        await session.commit()
        return updated

    async def _compute_article_stats(
        self,
        session: AsyncSession,
        article_id: str,
    ) -> dict:
        """Compute vote statistics for a single article."""
        stats = {
            "votes_1_week": 0,
            "avg_score_1_week": 0.0,
            "votes_1_month": 0,
            "avg_score_1_month": 0.0,
            "votes_1_year": 0,
            "avg_score_1_year": 0.0,
        }

        for period in [VotePeriod.ONE_WEEK, VotePeriod.ONE_MONTH, VotePeriod.ONE_YEAR]:
            result = await session.execute(
                select(
                    func.count(DBVote.id),
                    func.avg(DBVote.score),
                )
                .where(DBVote.article_id == article_id)
                .where(DBVote.period == period.value)
            )
            row = result.one()
            count, avg = row

            period_key = period.value.replace("_", "_")
            stats[f"votes_{period_key}"] = count or 0
            stats[f"avg_score_{period_key}"] = float(avg) if avg else 0.0

        # Compute weighted score
        stats["weighted_vote_score"] = self._compute_weighted_score(stats)

        return stats

    def _compute_weighted_score(self, stats: dict) -> float:
        """Compute the time-weighted vote score."""
        w_week = self.settings.vote_weight_1_week
        w_month = self.settings.vote_weight_1_month
        w_year = self.settings.vote_weight_1_year

        total_weight = 0
        weighted_sum = 0

        if stats["votes_1_week"] > 0:
            weighted_sum += w_week * stats["avg_score_1_week"]
            total_weight += w_week

        if stats["votes_1_month"] > 0:
            weighted_sum += w_month * stats["avg_score_1_month"]
            total_weight += w_month

        if stats["votes_1_year"] > 0:
            weighted_sum += w_year * stats["avg_score_1_year"]
            total_weight += w_year

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    async def _upsert_stats(
        self,
        session: AsyncSession,
        article_id: str,
        stats: dict,
    ):
        """Insert or update vote statistics."""
        result = await session.execute(
            select(DBArticleVoteStats).where(DBArticleVoteStats.article_id == article_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.votes_1_week = stats["votes_1_week"]
            existing.avg_score_1_week = stats["avg_score_1_week"]
            existing.votes_1_month = stats["votes_1_month"]
            existing.avg_score_1_month = stats["avg_score_1_month"]
            existing.votes_1_year = stats["votes_1_year"]
            existing.avg_score_1_year = stats["avg_score_1_year"]
            existing.weighted_vote_score = stats["weighted_vote_score"]
            existing.computed_at = datetime.utcnow()
        else:
            new_stats = DBArticleVoteStats(
                article_id=article_id,
                **stats,
                computed_at=datetime.utcnow(),
            )
            session.add(new_stats)
