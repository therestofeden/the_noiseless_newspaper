"""
Core ranking service for articles.

Implements a hybrid ranking algorithm combining:
- Recency score (exponential decay)
- Vote score (time-weighted)
- PageRank score (citation importance)
- Topic match score (user preference alignment)

Uses a lambda parameter for smooth cold-start transition.
"""

import math
from datetime import datetime, timedelta
from typing import Sequence
from uuid import UUID

import structlog

from app.config import RankingWeights, get_settings
from app.models.domain import Article, ArticleRanking, TopicPath, UserPreferences

logger = structlog.get_logger(__name__)


class RankingService:
    """Service for computing article rankings."""

    def __init__(self, weights: RankingWeights | None = None):
        """Initialize the ranking service with optional custom weights."""
        self.weights = weights or get_settings().ranking

    def compute_recency_score(
        self,
        published_at: datetime,
        now: datetime | None = None,
    ) -> float:
        """
        Compute recency score using exponential decay.

        Score = exp(-lambda * t) where lambda = ln(2) / half_life
        This gives a score of 1.0 for new articles, 0.5 at half-life,
        and approaches 0 for very old articles.

        Args:
            published_at: Article publication timestamp
            now: Current time (defaults to utcnow)

        Returns:
            Score between 0 and 1
        """
        if now is None:
            now = datetime.utcnow()

        # Ensure we don't get negative age (future articles)
        age_hours = max(0, (now - published_at).total_seconds() / 3600)

        # Exponential decay: half-life formula
        decay_constant = math.log(2) / self.weights.recency_half_life_hours
        score = math.exp(-decay_constant * age_hours)

        return max(0.0, min(1.0, score))

    def compute_vote_score(
        self,
        upvotes: int,
        downvotes: int,
        vote_timestamps: Sequence[datetime] | None = None,
        now: datetime | None = None,
    ) -> float:
        """
        Compute time-weighted vote score.

        More recent votes count more heavily than older votes.
        Uses a time decay factor for each vote's contribution.

        The base score uses a Wilson score interval approximation
        for ranking with confidence.

        Args:
            upvotes: Total upvote count
            downvotes: Total downvote count
            vote_timestamps: Optional list of vote timestamps for time-weighting
            now: Current time (defaults to utcnow)

        Returns:
            Normalized score between 0 and 1
        """
        if now is None:
            now = datetime.utcnow()

        total_votes = upvotes + downvotes

        if total_votes == 0:
            return 0.5  # Neutral score for unvoted articles

        # Base Wilson score lower bound (simplified)
        # This gives us a confidence-adjusted positive ratio
        p = upvotes / total_votes
        z = 1.96  # 95% confidence
        denominator = 1 + z * z / total_votes

        wilson_center = (p + z * z / (2 * total_votes)) / denominator
        wilson_width = (z / denominator) * math.sqrt(
            p * (1 - p) / total_votes + z * z / (4 * total_votes * total_votes)
        )

        base_score = wilson_center - wilson_width  # Lower bound

        # Apply time decay if timestamps provided
        if vote_timestamps:
            decay_days = self.weights.vote_decay_days
            time_weight = 0.0
            total_weight = 0.0

            for ts in vote_timestamps:
                age_days = max(0, (now - ts).total_seconds() / 86400)
                weight = math.exp(-age_days / decay_days)
                time_weight += weight
                total_weight += 1.0

            if total_weight > 0:
                time_factor = time_weight / total_weight
                base_score *= (0.5 + 0.5 * time_factor)  # Scale by time relevance

        return max(0.0, min(1.0, base_score))

    def compute_lambda(self, total_votes: int) -> float:
        """
        Compute lambda parameter for cold-start transition.

        Lambda smoothly transitions from 0 (new article, rely on priors)
        to 1 (mature article, rely on actual votes) using a sigmoid.

        lambda = 1 / (1 + exp(-k * (votes - midpoint)))

        Args:
            total_votes: Total number of votes on the article

        Returns:
            Lambda value between 0 and 1
        """
        k = self.weights.lambda_steepness
        midpoint = self.weights.lambda_midpoint_votes

        # Sigmoid function
        exponent = -k * (total_votes - midpoint)
        # Clamp exponent to avoid overflow
        exponent = max(-20, min(20, exponent))

        return 1 / (1 + math.exp(exponent))

    def compute_topic_match_score(
        self,
        article_topics: Sequence[TopicPath],
        user_preferences: UserPreferences | None,
    ) -> float:
        """
        Compute how well article topics match user preferences.

        Args:
            article_topics: Topics assigned to the article
            user_preferences: User's topic preferences (if any)

        Returns:
            Score between 0 and 1
        """
        if not user_preferences or not user_preferences.topic_preferences:
            return 0.5  # Neutral score if no preferences

        if not article_topics:
            return 0.3  # Low score for unclassified articles

        max_score = 0.0
        pref_dict = {p.topic_path.path_string: p for p in user_preferences.topic_preferences}

        for article_topic in article_topics:
            # Check for exact match
            path = article_topic.path_string
            if path in pref_dict:
                pref = pref_dict[path]
                if pref.excluded:
                    return 0.0  # Excluded topic
                max_score = max(max_score, pref.weight)
                continue

            # Check for parent match (domain.subtopic)
            parent_path = f"{article_topic.domain}.{article_topic.subtopic}"
            if parent_path in pref_dict:
                pref = pref_dict[parent_path]
                if pref.excluded:
                    return 0.0
                max_score = max(max_score, pref.weight * 0.8)  # Slightly lower for parent match
                continue

            # Check for domain match
            if article_topic.domain in pref_dict:
                pref = pref_dict[article_topic.domain]
                if pref.excluded:
                    return 0.0
                max_score = max(max_score, pref.weight * 0.6)  # Lower for domain-only match

        return max_score if max_score > 0 else 0.4  # Default for non-matching topics

    def rank_article(
        self,
        article: Article,
        pagerank_score: float,
        upvotes: int = 0,
        downvotes: int = 0,
        vote_timestamps: Sequence[datetime] | None = None,
        user_preferences: UserPreferences | None = None,
        now: datetime | None = None,
    ) -> ArticleRanking:
        """
        Compute full ranking for a single article.

        Final score = lambda * (w1*vote + w2*pagerank + w3*recency + w4*topic)
                    + (1-lambda) * prior_score

        where prior_score is based on recency and topic match only.

        Args:
            article: The article to rank
            pagerank_score: Pre-computed PageRank score
            upvotes: Article upvote count
            downvotes: Article downvote count
            vote_timestamps: Optional vote timestamps for time-weighting
            user_preferences: User preferences for personalization
            now: Current time

        Returns:
            ArticleRanking with all scores
        """
        if now is None:
            now = datetime.utcnow()

        # Compute individual scores
        recency_score = self.compute_recency_score(article.published_at, now)
        vote_score = self.compute_vote_score(upvotes, downvotes, vote_timestamps, now)
        topic_match_score = self.compute_topic_match_score(article.topics, user_preferences)

        # Normalize PageRank score (assumed to be pre-normalized, but clamp anyway)
        pagerank_normalized = max(0.0, min(1.0, pagerank_score))

        # Compute lambda for cold-start handling
        total_votes = upvotes + downvotes
        lambda_value = self.compute_lambda(total_votes)

        # Compute weighted mature score (when we have enough votes)
        mature_score = (
            self.weights.weight_votes * vote_score
            + self.weights.weight_pagerank * pagerank_normalized
            + self.weights.weight_recency * recency_score
            + self.weights.weight_topic_match * topic_match_score
        )

        # Compute prior score (for cold-start, before we have votes)
        # Rely more on recency, topic match, and pagerank
        prior_score = (
            0.4 * recency_score
            + 0.3 * pagerank_normalized
            + 0.3 * topic_match_score
        )

        # Blend using lambda
        final_score = lambda_value * mature_score + (1 - lambda_value) * prior_score

        return ArticleRanking(
            article=article,
            final_score=final_score,
            recency_score=recency_score,
            vote_score=vote_score,
            pagerank_score=pagerank_normalized,
            topic_match_score=topic_match_score,
            lambda_value=lambda_value,
        )

    def rank_articles(
        self,
        articles: Sequence[Article],
        pagerank_scores: dict[UUID, float],
        vote_stats: dict[UUID, tuple[int, int, list[datetime] | None]],
        user_preferences: UserPreferences | None = None,
        now: datetime | None = None,
    ) -> list[ArticleRanking]:
        """
        Rank multiple articles and return sorted by score.

        Args:
            articles: Articles to rank
            pagerank_scores: Dict of article_id -> PageRank score
            vote_stats: Dict of article_id -> (upvotes, downvotes, timestamps)
            user_preferences: User preferences for personalization
            now: Current time

        Returns:
            List of ArticleRanking sorted by final_score descending
        """
        if now is None:
            now = datetime.utcnow()

        rankings = []
        for article in articles:
            pr_score = pagerank_scores.get(article.id, 0.0)
            upvotes, downvotes, timestamps = vote_stats.get(article.id, (0, 0, None))

            ranking = self.rank_article(
                article=article,
                pagerank_score=pr_score,
                upvotes=upvotes,
                downvotes=downvotes,
                vote_timestamps=timestamps,
                user_preferences=user_preferences,
                now=now,
            )
            rankings.append(ranking)

        # Sort by final score descending
        rankings.sort(key=lambda r: r.final_score, reverse=True)

        logger.info(
            "Ranked articles",
            count=len(rankings),
            top_score=rankings[0].final_score if rankings else None,
        )

        return rankings


# Global service instance
_ranking_service: RankingService | None = None


def get_ranking_service() -> RankingService:
    """Get or create the ranking service singleton."""
    global _ranking_service
    if _ranking_service is None:
        _ranking_service = RankingService()
    return _ranking_service
