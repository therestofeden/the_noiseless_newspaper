"""
Survival Prediction Model.

This module implements the machine learning model that predicts
whether an article will have lasting value ("survive") over time.

The model evolves through phases:
1. Heuristic: Simple rules based on citations and source
2. Logistic: Logistic regression on top features
3. Boosting: Gradient boosting with full feature set
4. Neural: Deep learning with text embeddings (future)

The model is trained on articles with 1-year votes (ground truth)
and predicts survival scores for new articles.
"""
import json
import pickle
import math
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import DBArticle, DBArticleVoteStats
from app.models.domain import Article
from app.services.feature_extraction import ArticleFeatures, FeatureExtractionService


@dataclass
class ModelMetrics:
    """Metrics from model training/evaluation."""
    trained_at: datetime
    training_samples: int
    validation_samples: int

    # Regression metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared

    # Classification metrics (survival > 0.6 threshold)
    accuracy: float
    precision: float
    recall: float
    auc_roc: float

    # Feature importance (top 10)
    feature_importance: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "trained_at": self.trained_at.isoformat(),
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "auc_roc": self.auc_roc,
            "feature_importance": self.feature_importance,
        }


class BaseSurvivalModel(ABC):
    """Abstract base class for survival prediction models."""

    @abstractmethod
    def predict(self, features: ArticleFeatures) -> float:
        """
        Predict survival score for an article.

        Args:
            features: Extracted features for the article

        Returns:
            Predicted survival score in [0, 1]
        """
        pass

    @abstractmethod
    def predict_batch(self, features_list: list[ArticleFeatures]) -> list[float]:
        """Predict survival scores for multiple articles."""
        pass

    @abstractmethod
    def train(
        self,
        features_list: list[ArticleFeatures],
        survival_scores: list[float],
    ) -> ModelMetrics:
        """
        Train the model on labeled data.

        Args:
            features_list: Features for training articles
            survival_scores: Actual survival scores (labels)

        Returns:
            Training metrics
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk."""
        pass


class HeuristicSurvivalModel(BaseSurvivalModel):
    """
    Phase 1: Heuristic-based survival prediction.

    Used when we have insufficient training data (<100 articles with votes).
    Combines simple rules based on domain knowledge:
    - Citation count (proxy for quality)
    - Source reputation (known good sources)
    - Content signals (methodology, limitations = good)
    """

    # Default source quality scores (0-1)
    SOURCE_QUALITY = {
        "arxiv": 0.7,
        "semantic_scholar": 0.75,
        "openalex": 0.7,
        "newsapi": 0.5,
        "nature": 0.9,
        "science": 0.9,
        "pnas": 0.85,
        "cell": 0.85,
        "unknown": 0.5,
    }

    def __init__(self):
        self.settings = get_settings()
        self._source_quality = self.SOURCE_QUALITY.copy()

    def predict(self, features: ArticleFeatures) -> float:
        """
        Predict survival using heuristics.

        Formula:
            score = 0.3 * citation_signal
                  + 0.3 * source_signal
                  + 0.2 * content_quality_signal
                  + 0.2 * author_signal
        """
        # Citation signal (log-normalized)
        citation_signal = features.citation_count_norm

        # Boost if we have PageRank
        if features.pagerank_score > 0:
            citation_signal = 0.5 * citation_signal + 0.5 * features.pagerank_score

        # Source signal
        source_key = features.source_type.lower()
        base_source_quality = self._source_quality.get(source_key, 0.5)

        # Adjust by historical data if available
        if features.source_survival_confidence > 0.3:
            source_signal = (
                0.3 * base_source_quality
                + 0.7 * features.source_historical_survival
            )
        else:
            source_signal = base_source_quality

        # Content quality signal
        content_signal = 0.5  # Base
        if features.abstract_has_methodology:
            content_signal += 0.15
        if features.abstract_has_results:
            content_signal += 0.1
        if features.abstract_has_limitations:
            content_signal += 0.15  # Acknowledging limitations is good!
        if features.is_review:
            content_signal += 0.1  # Reviews tend to have lasting value

        # Penalize pure opinion without evidence
        if features.is_opinion and not features.abstract_has_results:
            content_signal -= 0.1

        content_signal = max(0, min(1, content_signal))

        # Author signal
        if features.author_survival_confidence > 0.3:
            author_signal = features.author_historical_survival
        else:
            # Use author count as weak proxy (collaboration often = higher quality)
            author_signal = min(0.4 + features.author_count_norm * 0.3, 0.7)

        # Combine signals
        prediction = (
            0.30 * citation_signal
            + 0.30 * source_signal
            + 0.25 * content_signal
            + 0.15 * author_signal
        )

        # Boost if early votes are positive
        if features.has_early_votes and features.early_vote_positive_ratio > 0.5:
            prediction = prediction * 0.7 + features.early_vote_mean / 5.0 * 0.3

        return max(0.0, min(1.0, prediction))

    def predict_batch(self, features_list: list[ArticleFeatures]) -> list[float]:
        return [self.predict(f) for f in features_list]

    def train(
        self,
        features_list: list[ArticleFeatures],
        survival_scores: list[float],
    ) -> ModelMetrics:
        """
        "Train" heuristic model by calibrating source quality scores.

        We update source quality based on actual survival rates.
        """
        # Group by source
        source_scores: dict[str, list[float]] = {}
        for features, score in zip(features_list, survival_scores):
            source = features.source_type.lower()
            if source not in source_scores:
                source_scores[source] = []
            source_scores[source].append(score)

        # Update source quality estimates
        for source, scores in source_scores.items():
            if len(scores) >= 5:  # Need enough samples
                self._source_quality[source] = sum(scores) / len(scores)

        # Compute metrics
        predictions = self.predict_batch(features_list)
        metrics = self._compute_metrics(predictions, survival_scores, features_list)

        return metrics

    def _compute_metrics(
        self,
        predictions: list[float],
        actuals: list[float],
        features_list: list[ArticleFeatures],
    ) -> ModelMetrics:
        """Compute evaluation metrics."""
        n = len(predictions)

        # Regression metrics
        errors = [p - a for p, a in zip(predictions, actuals)]
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

        # R-squared
        mean_actual = sum(actuals) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum(e ** 2 for e in errors)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Classification metrics (threshold = 0.6)
        threshold = 0.6
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        actual_binary = [1 if a >= threshold else 0 for a in actuals]

        tp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 1)
        tn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 0)

        accuracy = (tp + tn) / n if n > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Simplified AUC (would need proper implementation)
        auc_roc = 0.5 + (accuracy - 0.5) * 0.8  # Rough estimate

        return ModelMetrics(
            trained_at=datetime.utcnow(),
            training_samples=n,
            validation_samples=0,
            mae=mae,
            rmse=rmse,
            r2=r2,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc_roc=auc_roc,
            feature_importance={
                "citation_count_norm": 0.30,
                "source_historical_survival": 0.30,
                "abstract_has_methodology": 0.15,
                "abstract_has_limitations": 0.15,
                "author_count_norm": 0.10,
            },
        )

    def save(self, path: Path):
        """Save source quality calibration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "model_type": "heuristic",
                "source_quality": self._source_quality,
                "saved_at": datetime.utcnow().isoformat(),
            }, f, indent=2)

    def load(self, path: Path):
        """Load source quality calibration."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self._source_quality.update(data.get("source_quality", {}))


class LogisticSurvivalModel(BaseSurvivalModel):
    """
    Phase 2: Logistic regression survival prediction.

    Used when we have 100-1000 articles with votes.
    Learns feature weights from data.
    """

    def __init__(self):
        self.settings = get_settings()
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feature_names = ArticleFeatures.feature_names()
        self._metrics: Optional[ModelMetrics] = None

    def predict(self, features: ArticleFeatures) -> float:
        if self._weights is None:
            # Fall back to heuristic
            return HeuristicSurvivalModel().predict(features)

        x = features.to_feature_vector()
        logit = np.dot(x, self._weights) + self._bias
        return self._sigmoid(logit)

    def predict_batch(self, features_list: list[ArticleFeatures]) -> list[float]:
        if self._weights is None:
            return HeuristicSurvivalModel().predict_batch(features_list)

        X = np.array([f.to_feature_vector() for f in features_list])
        logits = np.dot(X, self._weights) + self._bias
        return [self._sigmoid(l) for l in logits]

    def train(
        self,
        features_list: list[ArticleFeatures],
        survival_scores: list[float],
    ) -> ModelMetrics:
        """
        Train logistic regression using gradient descent.

        This is a simplified implementation. In production, use sklearn.
        """
        X = np.array([f.to_feature_vector() for f in features_list])
        y = np.array(survival_scores)

        # Add small regularization to avoid overfitting
        n_features = X.shape[1]
        self._weights = np.zeros(n_features)
        self._bias = 0.0

        learning_rate = 0.1
        n_iterations = 1000
        lambda_reg = 0.01

        for _ in range(n_iterations):
            # Forward pass
            logits = np.dot(X, self._weights) + self._bias
            predictions = np.array([self._sigmoid(l) for l in logits])

            # Compute gradients
            errors = predictions - y
            grad_weights = np.dot(X.T, errors) / len(y) + lambda_reg * self._weights
            grad_bias = np.mean(errors)

            # Update
            self._weights -= learning_rate * grad_weights
            self._bias -= learning_rate * grad_bias

        # Compute metrics
        predictions = self.predict_batch(features_list)
        self._metrics = self._compute_metrics(
            predictions, survival_scores, features_list
        )

        return self._metrics

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)

    def _compute_metrics(
        self,
        predictions: list[float],
        actuals: list[float],
        features_list: list[ArticleFeatures],
    ) -> ModelMetrics:
        """Compute evaluation metrics."""
        n = len(predictions)

        # Regression metrics
        errors = [p - a for p, a in zip(predictions, actuals)]
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

        mean_actual = sum(actuals) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum(e ** 2 for e in errors)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Classification metrics
        threshold = 0.6
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        actual_binary = [1 if a >= threshold else 0 for a in actuals]

        tp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 1)
        tn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 0)

        accuracy = (tp + tn) / n if n > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        auc_roc = 0.5 + (accuracy - 0.5) * 0.8

        # Feature importance from weights
        feature_importance = {}
        if self._weights is not None:
            abs_weights = np.abs(self._weights)
            sorted_indices = np.argsort(abs_weights)[::-1][:10]
            for idx in sorted_indices:
                feature_importance[self._feature_names[idx]] = float(abs_weights[idx])

        return ModelMetrics(
            trained_at=datetime.utcnow(),
            training_samples=n,
            validation_samples=0,
            mae=mae,
            rmse=rmse,
            r2=r2,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            auc_roc=auc_roc,
            feature_importance=feature_importance,
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_type": "logistic",
                "weights": self._weights,
                "bias": self._bias,
                "metrics": self._metrics.to_dict() if self._metrics else None,
            }, f)

    def load(self, path: Path):
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self._weights = data.get("weights")
                self._bias = data.get("bias", 0.0)


class GradientBoostingSurvivalModel(BaseSurvivalModel):
    """
    Phase 3: Gradient boosting survival prediction.

    Used when we have 1000+ articles with votes.
    Uses XGBoost or LightGBM for better performance.

    This is a placeholder - in production, import and use actual libraries.
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._feature_names = ArticleFeatures.feature_names()
        self._fallback = LogisticSurvivalModel()

    def predict(self, features: ArticleFeatures) -> float:
        if self._model is None:
            return self._fallback.predict(features)

        # Would use: self._model.predict(features.to_feature_vector().reshape(1, -1))[0]
        return self._fallback.predict(features)

    def predict_batch(self, features_list: list[ArticleFeatures]) -> list[float]:
        if self._model is None:
            return self._fallback.predict_batch(features_list)

        # Would use: self._model.predict(X)
        return self._fallback.predict_batch(features_list)

    def train(
        self,
        features_list: list[ArticleFeatures],
        survival_scores: list[float],
    ) -> ModelMetrics:
        """
        Train gradient boosting model.

        In production:
            import xgboost as xgb
            X = np.array([f.to_feature_vector() for f in features_list])
            y = np.array(survival_scores)
            self._model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
            )
            self._model.fit(X, y)
        """
        # For now, fall back to logistic
        return self._fallback.train(features_list, survival_scores)

    def save(self, path: Path):
        # Would save XGBoost model
        self._fallback.save(path)

    def load(self, path: Path):
        self._fallback.load(path)


class SurvivalModelManager:
    """
    Manages survival prediction models.

    Handles:
    - Model selection based on data availability
    - Training and retraining
    - Model persistence
    - A/B testing between models
    """

    MODEL_PATH = Path("data/models")

    def __init__(self, feature_service: FeatureExtractionService):
        self.settings = get_settings()
        self.feature_service = feature_service

        # Initialize models
        self._heuristic = HeuristicSurvivalModel()
        self._logistic = LogisticSurvivalModel()
        self._boosting = GradientBoostingSurvivalModel()

        # Currently active model
        self._active_model: BaseSurvivalModel = self._heuristic

        # Try to load saved models
        self._load_models()

    def _load_models(self):
        """Load saved models from disk."""
        heuristic_path = self.MODEL_PATH / "heuristic.json"
        logistic_path = self.MODEL_PATH / "logistic.pkl"
        boosting_path = self.MODEL_PATH / "boosting.pkl"

        self._heuristic.load(heuristic_path)
        self._logistic.load(logistic_path)
        self._boosting.load(boosting_path)

    def _save_models(self):
        """Save all models to disk."""
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)

        self._heuristic.save(self.MODEL_PATH / "heuristic.json")
        self._logistic.save(self.MODEL_PATH / "logistic.pkl")
        self._boosting.save(self.MODEL_PATH / "boosting.pkl")

    async def predict(
        self,
        article: Article,
        session: AsyncSession,
    ) -> float:
        """
        Predict survival score for an article.

        Uses the best available model based on training data.
        """
        features = await self.feature_service.extract_features(article, session)
        return self._active_model.predict(features)

    async def predict_batch(
        self,
        articles: list[Article],
        session: AsyncSession,
    ) -> list[float]:
        """Predict survival scores for multiple articles."""
        features_list = await self.feature_service.extract_features_batch(
            articles, session
        )
        return self._active_model.predict_batch(features_list)

    async def train(self, session: AsyncSession) -> ModelMetrics:
        """
        Train models on all available labeled data.

        Selects the appropriate model based on data size:
        - <100 samples: Calibrate heuristic only
        - 100-1000 samples: Train logistic regression
        - 1000+ samples: Train gradient boosting
        """
        # Get all articles with 1-year votes (ground truth)
        result = await session.execute(
            select(DBArticle, DBArticleVoteStats)
            .join(DBArticleVoteStats, DBArticle.id == DBArticleVoteStats.article_id)
            .where(DBArticleVoteStats.votes_1_year > 0)
        )
        rows = result.all()

        if not rows:
            return ModelMetrics(
                trained_at=datetime.utcnow(),
                training_samples=0,
                validation_samples=0,
                mae=0, rmse=0, r2=0,
                accuracy=0, precision=0, recall=0, auc_roc=0.5,
                feature_importance={},
            )

        # Convert to domain objects and extract features
        articles = []
        survival_scores = []

        for db_article, vote_stats in rows:
            article = self._db_to_domain(db_article)
            articles.append(article)
            # Normalize survival score to 0-1
            survival_scores.append(vote_stats.weighted_vote_score / 5.0)

        features_list = await self.feature_service.extract_features_batch(
            articles, session
        )

        n_samples = len(features_list)

        # Train appropriate model
        if n_samples < 100:
            metrics = self._heuristic.train(features_list, survival_scores)
            self._active_model = self._heuristic
        elif n_samples < 1000:
            metrics = self._logistic.train(features_list, survival_scores)
            self._active_model = self._logistic
        else:
            metrics = self._boosting.train(features_list, survival_scores)
            self._active_model = self._boosting

        # Save models
        self._save_models()

        return metrics

    def get_active_model_type(self) -> str:
        """Return the type of the currently active model."""
        if isinstance(self._active_model, HeuristicSurvivalModel):
            return "heuristic"
        elif isinstance(self._active_model, LogisticSurvivalModel):
            return "logistic"
        elif isinstance(self._active_model, GradientBoostingSurvivalModel):
            return "gradient_boosting"
        return "unknown"

    def _db_to_domain(self, db_article: DBArticle) -> Article:
        """Convert database model to domain model."""
        import json
        from app.models.domain import Author

        authors = []
        if db_article.authors_json:
            try:
                authors_data = (
                    json.loads(db_article.authors_json)
                    if isinstance(db_article.authors_json, str)
                    else db_article.authors_json
                )
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


class ExplorationStrategy:
    """
    Exploration vs. Exploitation for article selection.

    Balances showing high-predicted-survival articles (exploitation)
    with gathering data on uncertain articles (exploration).
    """

    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Probability of exploration (showing uncertain article)
        """
        self.epsilon = epsilon

    def select_article(
        self,
        articles: list[Article],
        predictions: list[float],
        uncertainties: Optional[list[float]] = None,
    ) -> int:
        """
        Select which article to show.

        With probability epsilon, explores (picks uncertain article).
        Otherwise, exploits (picks highest predicted survival).

        Args:
            articles: List of candidate articles
            predictions: Predicted survival scores
            uncertainties: Optional uncertainty estimates

        Returns:
            Index of selected article
        """
        import random

        if random.random() < self.epsilon:
            # Exploration: pick article with high uncertainty
            if uncertainties:
                # Weight by uncertainty
                weights = [u + 0.01 for u in uncertainties]  # Avoid zero
                total = sum(weights)
                probs = [w / total for w in weights]
                return random.choices(range(len(articles)), weights=probs)[0]
            else:
                # Random selection
                return random.randint(0, len(articles) - 1)
        else:
            # Exploitation: pick highest predicted survival
            return max(range(len(predictions)), key=lambda i: predictions[i])

    def thompson_sampling(
        self,
        articles: list[Article],
        alpha: list[float],
        beta: list[float],
    ) -> int:
        """
        Thompson Sampling for article selection.

        Models each article's survival probability as Beta(alpha, beta).
        Samples from each distribution and picks the highest.

        Args:
            articles: List of candidate articles
            alpha: Alpha parameters (successes + 1)
            beta: Beta parameters (failures + 1)

        Returns:
            Index of selected article
        """
        import random

        samples = []
        for a, b in zip(alpha, beta):
            # Sample from Beta distribution
            # Simplified: use mean + noise instead of proper Beta
            mean = a / (a + b)
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            sample = mean + random.gauss(0, math.sqrt(variance))
            samples.append(sample)

        return max(range(len(samples)), key=lambda i: samples[i])
