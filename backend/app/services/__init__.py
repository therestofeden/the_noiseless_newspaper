"""
Services layer - core business logic for The Noiseless Newspaper.

The services implement the key algorithms:

1. Citation Graph (citation_graph.py):
   - PageRank computation on citation network
   - Cold-start ranking for new articles

2. Embeddings (embeddings.py):
   - Semantic similarity for topic matching
   - Article-to-topic relevance scoring

3. Feature Extraction (feature_extraction.py):
   - Extract predictive features from articles
   - Content, source, citation, temporal features

4. Survival Model (survival_model.py):
   - ML models to predict article "survival"
   - Learns from time-delayed user votes
   - Evolves: heuristic → logistic → gradient boosting

5. Ranking (ranking.py):
   - Combines all signals into final score
   - Adaptive weighting based on vote availability
   - Exploration vs exploitation for learning

6. Summarization (summarization.py):
   - LLM-powered article summarization
"""

from app.services.citation_graph import CitationGraphService
from app.services.embeddings import EmbeddingService, TopicEmbeddingService
from app.services.feature_extraction import (
    ArticleFeatures,
    FeatureExtractionService,
    NoveltyScorer,
)
from app.services.survival_model import (
    BaseSurvivalModel,
    HeuristicSurvivalModel,
    LogisticSurvivalModel,
    GradientBoostingSurvivalModel,
    SurvivalModelManager,
    ExplorationStrategy,
    ModelMetrics,
)
from app.services.ranking import RankingService, VoteAggregationService
from app.services.summarization import SummarizationService

__all__ = [
    # Citation graph
    "CitationGraphService",
    # Embeddings
    "EmbeddingService",
    "TopicEmbeddingService",
    # Feature extraction
    "ArticleFeatures",
    "FeatureExtractionService",
    "NoveltyScorer",
    # Survival prediction
    "BaseSurvivalModel",
    "HeuristicSurvivalModel",
    "LogisticSurvivalModel",
    "GradientBoostingSurvivalModel",
    "SurvivalModelManager",
    "ExplorationStrategy",
    "ModelMetrics",
    # Ranking
    "RankingService",
    "VoteAggregationService",
    # Summarization
    "SummarizationService",
]
