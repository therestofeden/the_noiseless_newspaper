"""
Feature Extraction Service for Survival Prediction.

Extracts predictive features from articles that can indicate
whether content will have lasting value ("survive" over time).

Features are grouped into categories:
- Content features: What the article says and how
- Source features: Who published it and their track record
- Citation features: Academic impact signals
- Temporal features: When and in what context it was published
- Early engagement features: How users initially responded
"""
import re
import math
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DBArticle, DBVote, DBArticleVoteStats
from app.models.domain import Article, VotePeriod


@dataclass
class ArticleFeatures:
    """
    Complete feature vector for an article.

    All features are normalized to [0, 1] range or are categorical.
    Missing features default to 0.0 or domain-appropriate values.
    """
    # Identifiers (not used in model)
    article_id: str
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    # === CONTENT FEATURES ===
    # Title characteristics
    title_length: int = 0                    # Word count
    title_length_norm: float = 0.0           # Normalized (0-1, capped at 20 words)
    title_has_question: bool = False         # Contains "?"
    title_has_number: bool = False           # Contains digits
    title_has_colon: bool = False            # Contains ":" (often indicates structure)
    title_hedging_score: float = 0.0         # Presence of hedging words
    title_confidence_score: float = 0.0      # Presence of confident words

    # Abstract characteristics
    abstract_length: int = 0                 # Word count
    abstract_length_norm: float = 0.0        # Normalized (0-1, capped at 500 words)
    abstract_sentence_count: int = 0         # Number of sentences
    abstract_avg_sentence_length: float = 0.0  # Words per sentence
    abstract_complexity: float = 0.0         # Flesch-Kincaid proxy
    abstract_has_methodology: bool = False   # Mentions methods/approach
    abstract_has_results: bool = False       # Mentions results/findings
    abstract_has_limitations: bool = False   # Acknowledges limitations

    # Claim type indicators
    is_discovery: bool = False               # "we found", "we discovered"
    is_analysis: bool = False                # "we analyzed", "we examined"
    is_review: bool = False                  # "we review", "survey", "overview"
    is_opinion: bool = False                 # "we argue", "we believe"
    is_replication: bool = False             # "we replicate", "we reproduce"

    # === SOURCE FEATURES ===
    source_type: str = "unknown"             # "academic", "news", "preprint", etc.
    source_historical_survival: float = 0.5  # Avg survival score from this source
    source_article_count: int = 0            # How many articles we've seen from source
    source_survival_confidence: float = 0.0  # Confidence in source survival estimate

    author_count: int = 0                    # Number of authors
    author_count_norm: float = 0.0           # Normalized (0-1, capped at 10)
    author_historical_survival: float = 0.5  # Avg survival score from authors
    author_article_count: int = 0            # Total articles from these authors
    author_survival_confidence: float = 0.0  # Confidence in author survival estimate

    # === CITATION FEATURES ===
    citation_count_initial: int = 0          # Citations at ingestion time
    citation_count_norm: float = 0.0         # Log-normalized
    citation_velocity_7d: float = 0.0        # Citations per day in first week
    citation_velocity_norm: float = 0.0      # Normalized velocity
    reference_count: int = 0                 # How many papers it cites
    reference_count_norm: float = 0.0        # Normalized
    cross_domain_ratio: float = 0.0          # Citations from other fields (0-1)
    self_citation_ratio: float = 0.0         # Author self-citations (0-1)

    # PageRank-derived
    pagerank_score: float = 0.0              # From citation graph
    pagerank_percentile: float = 0.0         # Percentile among recent articles

    # === TEMPORAL FEATURES ===
    days_since_publication: int = 0          # Age in days
    day_of_week: int = 0                     # 0=Monday, 6=Sunday
    month_of_year: int = 0                   # 1-12
    is_weekend_publication: bool = False     # Published on Sat/Sun

    topic_trend_score: float = 0.5           # Is topic trending? (0=cold, 1=hot)
    topic_evergreen_score: float = 0.5       # Historical topic survival rate
    days_since_last_survivor: int = 0        # Days since last high-survival article in topic

    # === EARLY ENGAGEMENT FEATURES ===
    # (Only available after some time has passed)
    has_early_votes: bool = False            # Any votes within first week?
    early_vote_count: int = 0                # Number of 1-week votes
    early_vote_mean: float = 0.0             # Average 1-week vote (1-5 scale, normalized)
    early_vote_variance: float = 0.0         # Variance in 1-week votes
    early_vote_positive_ratio: float = 0.0   # % of votes >= 4

    # === EMBEDDING FEATURES ===
    # (Computed separately, stored as reference)
    topic_similarity: float = 0.0            # Similarity to assigned topic
    novelty_score: float = 0.5               # Distance from recent articles (0=similar, 1=novel)

    # === TARGET VARIABLE ===
    # (Only set for training data)
    actual_survival_score: Optional[float] = None  # Weighted vote score if available
    has_1year_votes: bool = False            # Do we have ground truth?

    def to_dict(self) -> dict:
        """Convert to dictionary for storage/serialization."""
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to numpy array for model input.

        Returns features in consistent order for model training/inference.
        Excludes identifiers and target variable.
        """
        return np.array([
            # Content features
            self.title_length_norm,
            float(self.title_has_question),
            float(self.title_has_number),
            float(self.title_has_colon),
            self.title_hedging_score,
            self.title_confidence_score,
            self.abstract_length_norm,
            self.abstract_avg_sentence_length / 30.0,  # Normalize to ~1
            self.abstract_complexity,
            float(self.abstract_has_methodology),
            float(self.abstract_has_results),
            float(self.abstract_has_limitations),
            float(self.is_discovery),
            float(self.is_analysis),
            float(self.is_review),
            float(self.is_opinion),
            float(self.is_replication),

            # Source features
            self.source_historical_survival,
            self.source_survival_confidence,
            self.author_count_norm,
            self.author_historical_survival,
            self.author_survival_confidence,

            # Citation features
            self.citation_count_norm,
            self.citation_velocity_norm,
            self.reference_count_norm,
            self.cross_domain_ratio,
            self.pagerank_score,
            self.pagerank_percentile,

            # Temporal features
            float(self.is_weekend_publication),
            self.topic_trend_score,
            self.topic_evergreen_score,
            min(self.days_since_last_survivor / 30.0, 1.0),  # Normalize

            # Early engagement (if available)
            float(self.has_early_votes),
            self.early_vote_mean / 5.0,  # Normalize from 1-5 to 0-1
            self.early_vote_variance / 2.0,  # Normalize
            self.early_vote_positive_ratio,

            # Embedding features
            self.topic_similarity,
            self.novelty_score,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered list of feature names matching to_feature_vector()."""
        return [
            "title_length_norm",
            "title_has_question",
            "title_has_number",
            "title_has_colon",
            "title_hedging_score",
            "title_confidence_score",
            "abstract_length_norm",
            "abstract_avg_sentence_length_norm",
            "abstract_complexity",
            "abstract_has_methodology",
            "abstract_has_results",
            "abstract_has_limitations",
            "is_discovery",
            "is_analysis",
            "is_review",
            "is_opinion",
            "is_replication",
            "source_historical_survival",
            "source_survival_confidence",
            "author_count_norm",
            "author_historical_survival",
            "author_survival_confidence",
            "citation_count_norm",
            "citation_velocity_norm",
            "reference_count_norm",
            "cross_domain_ratio",
            "pagerank_score",
            "pagerank_percentile",
            "is_weekend_publication",
            "topic_trend_score",
            "topic_evergreen_score",
            "days_since_last_survivor_norm",
            "has_early_votes",
            "early_vote_mean_norm",
            "early_vote_variance_norm",
            "early_vote_positive_ratio",
            "topic_similarity",
            "novelty_score",
        ]


class FeatureExtractionService:
    """
    Extracts predictive features from articles.

    This service is the foundation of the survival prediction model.
    It transforms raw article data into a feature vector that can
    predict whether the content will have lasting value.
    """

    # Words that indicate hedging/uncertainty
    HEDGING_WORDS = {
        "may", "might", "could", "possibly", "perhaps", "suggest",
        "appears", "seems", "likely", "potential", "preliminary",
        "exploratory", "tentative", "uncertain", "unclear"
    }

    # Words that indicate confidence/certainty
    CONFIDENCE_WORDS = {
        "demonstrate", "prove", "establish", "confirm", "show",
        "clearly", "definitely", "certainly", "undoubtedly",
        "conclusively", "significantly", "strongly"
    }

    # Claim type patterns
    DISCOVERY_PATTERNS = [
        r"\bwe (found|discovered|identified|detected|observed)\b",
        r"\b(novel|new|first) (finding|discovery|result)\b",
    ]
    ANALYSIS_PATTERNS = [
        r"\bwe (analyzed|examined|investigated|studied|explored)\b",
        r"\b(analysis|examination|investigation) of\b",
    ]
    REVIEW_PATTERNS = [
        r"\bwe (review|survey|summarize)\b",
        r"\b(comprehensive|systematic) (review|survey|overview)\b",
    ]
    OPINION_PATTERNS = [
        r"\bwe (argue|believe|propose|suggest|hypothesize)\b",
        r"\b(perspective|viewpoint|opinion|commentary)\b",
    ]
    REPLICATION_PATTERNS = [
        r"\bwe (replicate|reproduce|verify|validate)\b",
        r"\b(replication|reproduction) (study|of)\b",
    ]

    def __init__(self):
        # Compile regex patterns
        self._discovery_re = [re.compile(p, re.I) for p in self.DISCOVERY_PATTERNS]
        self._analysis_re = [re.compile(p, re.I) for p in self.ANALYSIS_PATTERNS]
        self._review_re = [re.compile(p, re.I) for p in self.REVIEW_PATTERNS]
        self._opinion_re = [re.compile(p, re.I) for p in self.OPINION_PATTERNS]
        self._replication_re = [re.compile(p, re.I) for p in self.REPLICATION_PATTERNS]

    async def extract_features(
        self,
        article: Article,
        session: AsyncSession,
    ) -> ArticleFeatures:
        """
        Extract all features for a single article.

        Args:
            article: The article to extract features from
            session: Database session for historical queries

        Returns:
            ArticleFeatures dataclass with all computed features
        """
        features = ArticleFeatures(article_id=article.id)

        # Extract content features
        self._extract_title_features(article, features)
        self._extract_abstract_features(article, features)
        self._extract_claim_type(article, features)

        # Extract source features (requires DB)
        await self._extract_source_features(article, session, features)
        await self._extract_author_features(article, session, features)

        # Extract citation features
        self._extract_citation_features(article, features)

        # Extract temporal features (requires DB)
        self._extract_temporal_features(article, features)
        await self._extract_topic_temporal_features(article, session, features)

        # Extract early engagement features (requires DB)
        await self._extract_early_engagement_features(article, session, features)

        # Store topic similarity if available
        if article.topic_relevance_score:
            features.topic_similarity = article.topic_relevance_score

        # Store PageRank if available
        if article.pagerank_score:
            features.pagerank_score = article.pagerank_score

        # Check for ground truth
        await self._extract_survival_label(article, session, features)

        return features

    def _extract_title_features(self, article: Article, features: ArticleFeatures):
        """Extract features from the article title."""
        title = article.title or ""
        words = title.split()

        features.title_length = len(words)
        features.title_length_norm = min(len(words) / 20.0, 1.0)
        features.title_has_question = "?" in title
        features.title_has_number = bool(re.search(r'\d', title))
        features.title_has_colon = ":" in title

        # Compute hedging and confidence scores
        title_lower = title.lower()
        title_words_set = set(title_lower.split())

        hedging_count = len(title_words_set & self.HEDGING_WORDS)
        confidence_count = len(title_words_set & self.CONFIDENCE_WORDS)

        features.title_hedging_score = min(hedging_count / 3.0, 1.0)
        features.title_confidence_score = min(confidence_count / 3.0, 1.0)

    def _extract_abstract_features(self, article: Article, features: ArticleFeatures):
        """Extract features from the article abstract."""
        abstract = article.abstract or ""
        words = abstract.split()
        sentences = re.split(r'[.!?]+', abstract)
        sentences = [s.strip() for s in sentences if s.strip()]

        features.abstract_length = len(words)
        features.abstract_length_norm = min(len(words) / 500.0, 1.0)
        features.abstract_sentence_count = len(sentences)

        if sentences:
            features.abstract_avg_sentence_length = len(words) / len(sentences)

        # Simple complexity proxy (average word length)
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            # Normalize: 4 chars = simple, 8+ chars = complex
            features.abstract_complexity = min((avg_word_length - 4) / 4.0, 1.0)
            features.abstract_complexity = max(0.0, features.abstract_complexity)

        # Check for structural elements
        abstract_lower = abstract.lower()
        features.abstract_has_methodology = any(
            word in abstract_lower
            for word in ["method", "approach", "procedure", "technique", "algorithm"]
        )
        features.abstract_has_results = any(
            word in abstract_lower
            for word in ["result", "finding", "outcome", "show", "demonstrate"]
        )
        features.abstract_has_limitations = any(
            word in abstract_lower
            for word in ["limitation", "caveat", "however", "although", "despite"]
        )

    def _extract_claim_type(self, article: Article, features: ArticleFeatures):
        """Identify the type of claim the article makes."""
        text = f"{article.title or ''} {article.abstract or ''}".lower()

        features.is_discovery = any(p.search(text) for p in self._discovery_re)
        features.is_analysis = any(p.search(text) for p in self._analysis_re)
        features.is_review = any(p.search(text) for p in self._review_re)
        features.is_opinion = any(p.search(text) for p in self._opinion_re)
        features.is_replication = any(p.search(text) for p in self._replication_re)

    async def _extract_source_features(
        self,
        article: Article,
        session: AsyncSession,
        features: ArticleFeatures,
    ):
        """Extract features about the article's source/publication."""
        features.source_type = article.source or "unknown"

        # Query historical survival rate for this source
        result = await session.execute(
            select(
                func.count(DBArticle.id),
                func.avg(DBArticleVoteStats.weighted_vote_score),
            )
            .join(DBArticleVoteStats, DBArticle.id == DBArticleVoteStats.article_id)
            .where(DBArticle.source == article.source)
            .where(DBArticleVoteStats.votes_1_year > 0)  # Has ground truth
        )
        row = result.one()
        count, avg_survival = row

        features.source_article_count = count or 0
        if count and count > 0 and avg_survival:
            features.source_historical_survival = float(avg_survival) / 5.0  # Normalize to 0-1
            # Confidence increases with more data
            features.source_survival_confidence = min(count / 50.0, 1.0)

    async def _extract_author_features(
        self,
        article: Article,
        session: AsyncSession,
        features: ArticleFeatures,
    ):
        """Extract features about the article's authors."""
        features.author_count = len(article.authors) if article.authors else 0
        features.author_count_norm = min(features.author_count / 10.0, 1.0)

        if not article.authors:
            return

        # Get author names for querying
        author_names = [a.name for a in article.authors if a.name]
        if not author_names:
            return

        # This is a simplified query - in production you'd have an author table
        # For now, we estimate based on articles with overlapping authors
        # (This would need proper author disambiguation in a real system)
        features.author_historical_survival = 0.5  # Default
        features.author_survival_confidence = 0.0  # Low confidence without proper author tracking

    def _extract_citation_features(self, article: Article, features: ArticleFeatures):
        """Extract citation-related features."""
        features.citation_count_initial = article.citation_count or 0

        # Log-normalize citation count
        if features.citation_count_initial > 0:
            features.citation_count_norm = math.log1p(features.citation_count_initial) / 10.0
            features.citation_count_norm = min(features.citation_count_norm, 1.0)

        # PageRank percentile would be computed in batch
        features.pagerank_score = article.pagerank_score or 0.0

        # Citation velocity requires tracking over time
        # For now, estimate from current count and age
        if article.published_at:
            age_days = (datetime.utcnow() - article.published_at).days
            if age_days > 0 and age_days <= 7:
                features.citation_velocity_7d = features.citation_count_initial / age_days
                features.citation_velocity_norm = min(features.citation_velocity_7d / 5.0, 1.0)

    def _extract_temporal_features(self, article: Article, features: ArticleFeatures):
        """Extract time-related features."""
        if not article.published_at:
            return

        pub_date = article.published_at
        features.days_since_publication = (datetime.utcnow() - pub_date).days
        features.day_of_week = pub_date.weekday()
        features.month_of_year = pub_date.month
        features.is_weekend_publication = pub_date.weekday() >= 5

    async def _extract_topic_temporal_features(
        self,
        article: Article,
        session: AsyncSession,
        features: ArticleFeatures,
    ):
        """Extract topic-specific temporal features."""
        if not article.topic_path:
            return

        # Topic evergreen score: historical survival rate in this topic
        result = await session.execute(
            select(func.avg(DBArticleVoteStats.weighted_vote_score))
            .join(DBArticle, DBArticle.id == DBArticleVoteStats.article_id)
            .where(DBArticle.topic_path == article.topic_path)
            .where(DBArticleVoteStats.votes_1_year > 0)
        )
        avg_survival = result.scalar()
        if avg_survival:
            features.topic_evergreen_score = float(avg_survival) / 5.0

        # Days since last "survivor" (high survival score) in topic
        result = await session.execute(
            select(func.max(DBArticle.published_at))
            .join(DBArticleVoteStats, DBArticle.id == DBArticleVoteStats.article_id)
            .where(DBArticle.topic_path == article.topic_path)
            .where(DBArticleVoteStats.weighted_vote_score >= 4.0)  # Threshold for "survivor"
        )
        last_survivor_date = result.scalar()
        if last_survivor_date:
            features.days_since_last_survivor = (datetime.utcnow() - last_survivor_date).days

        # Topic trend score would require external data (search trends, etc.)
        # For now, use publication volume as proxy
        week_ago = datetime.utcnow() - timedelta(days=7)
        result = await session.execute(
            select(func.count(DBArticle.id))
            .where(DBArticle.topic_path == article.topic_path)
            .where(DBArticle.published_at >= week_ago)
        )
        recent_count = result.scalar() or 0
        features.topic_trend_score = min(recent_count / 20.0, 1.0)  # Normalize

    async def _extract_early_engagement_features(
        self,
        article: Article,
        session: AsyncSession,
        features: ArticleFeatures,
    ):
        """Extract early user engagement features (1-week votes)."""
        # Get 1-week votes for this article
        result = await session.execute(
            select(DBVote.score)
            .where(DBVote.article_id == article.id)
            .where(DBVote.period == VotePeriod.ONE_WEEK.value)
        )
        votes = [row[0] for row in result.all()]

        if not votes:
            return

        features.has_early_votes = True
        features.early_vote_count = len(votes)
        features.early_vote_mean = sum(votes) / len(votes)

        if len(votes) > 1:
            mean = features.early_vote_mean
            features.early_vote_variance = sum((v - mean) ** 2 for v in votes) / len(votes)

        features.early_vote_positive_ratio = sum(1 for v in votes if v >= 4) / len(votes)

    async def _extract_survival_label(
        self,
        article: Article,
        session: AsyncSession,
        features: ArticleFeatures,
    ):
        """Extract the actual survival score if available (for training data)."""
        result = await session.execute(
            select(DBArticleVoteStats)
            .where(DBArticleVoteStats.article_id == article.id)
        )
        stats = result.scalar_one_or_none()

        if stats and stats.votes_1_year > 0:
            features.has_1year_votes = True
            features.actual_survival_score = stats.weighted_vote_score / 5.0  # Normalize to 0-1

    async def extract_features_batch(
        self,
        articles: list[Article],
        session: AsyncSession,
    ) -> list[ArticleFeatures]:
        """Extract features for multiple articles."""
        features_list = []
        for article in articles:
            features = await self.extract_features(article, session)
            features_list.append(features)
        return features_list


class NoveltyScorer:
    """
    Computes novelty scores for articles.

    Novelty is measured as the distance from recent articles in the same topic.
    High novelty = article covers something not recently discussed.
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self._recent_embeddings: dict[str, list[np.ndarray]] = {}  # topic -> embeddings

    async def compute_novelty(
        self,
        article: Article,
        recent_articles: list[Article],
    ) -> float:
        """
        Compute novelty score for an article.

        Args:
            article: The article to score
            recent_articles: Recent articles in the same topic

        Returns:
            Novelty score in [0, 1], where 1 = highly novel
        """
        if not recent_articles:
            return 0.5  # No comparison available

        # Get embedding for this article
        text = f"{article.title or ''} {article.abstract or ''}"
        article_embedding = await self.embedding_service.embed_text(text)

        # Get embeddings for recent articles
        similarities = []
        for recent in recent_articles:
            recent_text = f"{recent.title or ''} {recent.abstract or ''}"
            recent_embedding = await self.embedding_service.embed_text(recent_text)

            sim = self.embedding_service.cosine_similarity(
                article_embedding, recent_embedding
            )
            similarities.append(sim)

        # Novelty = 1 - max similarity to recent articles
        max_similarity = max(similarities)
        novelty = 1.0 - max(0, max_similarity)

        return novelty
