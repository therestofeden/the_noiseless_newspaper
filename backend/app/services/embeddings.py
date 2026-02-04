"""
Embedding service for semantic similarity matching.
Used to match articles to topics and find similar content.
"""
import asyncio
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.config import get_settings


class EmbeddingService:
    """
    Service for computing and comparing text embeddings.

    Supports multiple backends:
    1. Local sentence-transformers (free, runs on CPU)
    2. OpenAI text-embedding-3 (paid, higher quality)
    3. Mock embeddings (for testing)

    Embeddings are normalized vectors that capture semantic meaning.
    Similar texts have similar embeddings (high cosine similarity).
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._openai_client = None
        self._dimension = self.settings.embedding_dimension

    async def initialize(self):
        """Initialize the embedding model (lazy loading)."""
        if self.settings.use_openai_embeddings and self.settings.openai_api_key:
            await self._init_openai()
        else:
            await self._init_local()

    async def _init_openai(self):
        """Initialize OpenAI embeddings client."""
        try:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self._dimension = 1536  # text-embedding-3-small dimension

        except ImportError:
            # Fall back to local if openai not installed
            await self._init_local()

    async def _init_local(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.settings.embedding_model)
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

        except ImportError:
            # Fall back to mock embeddings if sentence-transformers not installed
            self._model = None

    async def embed_text(self, text: str) -> NDArray[np.float32]:
        """
        Compute embedding for a single text.

        Returns:
            Normalized embedding vector
        """
        if self._openai_client:
            return await self._embed_openai(text)
        elif self._model:
            return await self._embed_local(text)
        else:
            return self._embed_mock(text)

    async def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Compute embeddings for multiple texts (batched for efficiency).

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        if self._openai_client:
            return await self._embed_openai_batch(texts)
        elif self._model:
            return await self._embed_local_batch(texts)
        else:
            return np.array([self._embed_mock(t) for t in texts], dtype=np.float32)

    async def _embed_openai(self, text: str) -> NDArray[np.float32]:
        """Embed using OpenAI API."""
        response = await self._openai_client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return self._normalize(embedding)

    async def _embed_openai_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Batch embed using OpenAI API."""
        # OpenAI supports up to 2048 texts per request
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self._openai_client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
            all_embeddings.extend(embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        return self._normalize_batch(result)

    async def _embed_local(self, text: str) -> NDArray[np.float32]:
        """Embed using local sentence-transformers."""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return self._normalize(embedding.astype(np.float32))

    async def _embed_local_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Batch embed using local sentence-transformers."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        return self._normalize_batch(embeddings.astype(np.float32))

    def _embed_mock(self, text: str) -> NDArray[np.float32]:
        """Generate mock embedding for testing."""
        # Use hash of text to generate deterministic "embedding"
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert hash to float array and pad/truncate to dimension
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        if len(embedding) < self._dimension:
            embedding = np.pad(embedding, (0, self._dimension - len(embedding)))
        else:
            embedding = embedding[:self._dimension]

        return self._normalize(embedding)

    def _normalize(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _normalize_batch(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize batch of embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        return embeddings / norms

    @staticmethod
    def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """
        Compute cosine similarity between two embeddings.

        For normalized vectors, this is just the dot product.
        Returns value in [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite.
        """
        return float(np.dot(a, b))

    @staticmethod
    def cosine_similarity_matrix(
        query_embeddings: NDArray[np.float32],
        corpus_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Compute similarity matrix between queries and corpus.

        Returns:
            Matrix of shape (n_queries, n_corpus) with similarity scores
        """
        return np.matmul(query_embeddings, corpus_embeddings.T)

    def find_most_similar(
        self,
        query_embedding: NDArray[np.float32],
        corpus_embeddings: NDArray[np.float32],
        corpus_ids: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find most similar items from corpus.

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity
        """
        similarities = np.dot(corpus_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            (corpus_ids[i], float(similarities[i]))
            for i in top_indices
        ]


class TopicEmbeddingService:
    """
    Manages embeddings for the topic taxonomy.

    Pre-computes embeddings for all topics so we can quickly
    match articles to the most relevant topics.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._topic_embeddings: dict[str, NDArray[np.float32]] = {}
        self._topic_texts: dict[str, str] = {}

    async def initialize_topic_embeddings(self, taxonomy: dict):
        """
        Pre-compute embeddings for all topics in the taxonomy.

        Each topic is embedded using its name, description, and keywords.
        """
        topic_texts = {}

        for cat_id, category in taxonomy.items():
            for sub_id, subtopic in category.get("subtopics", {}).items():
                for niche_id, niche in subtopic.get("niches", {}).items():
                    path = f"{cat_id}/{sub_id}/{niche_id}"

                    # Combine topic information for richer embedding
                    text_parts = [
                        category.get("name", ""),
                        subtopic.get("name", ""),
                        niche.get("name", ""),
                    ]

                    # Add keywords if available
                    keywords = niche.get("keywords", [])
                    text_parts.extend(keywords)

                    topic_texts[path] = " ".join(text_parts)

        # Batch embed all topics
        if topic_texts:
            paths = list(topic_texts.keys())
            texts = list(topic_texts.values())

            embeddings = await self.embedding_service.embed_texts(texts)

            for path, embedding in zip(paths, embeddings):
                self._topic_embeddings[path] = embedding
                self._topic_texts[path] = topic_texts[path]

    async def match_article_to_topics(
        self,
        article_text: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Find the most relevant topics for an article.

        Args:
            article_text: Article title + abstract for matching
            top_k: Number of top topics to return

        Returns:
            List of (topic_path, similarity_score) tuples
        """
        if not self._topic_embeddings:
            return []

        article_embedding = await self.embedding_service.embed_text(article_text)

        paths = list(self._topic_embeddings.keys())
        embeddings = np.array(list(self._topic_embeddings.values()))

        return self.embedding_service.find_most_similar(
            article_embedding,
            embeddings,
            paths,
            top_k=top_k,
        )

    def get_topic_embedding(self, topic_path: str) -> Optional[NDArray[np.float32]]:
        """Get the pre-computed embedding for a topic."""
        return self._topic_embeddings.get(topic_path)


class ArticleIndexService:
    """
    In-memory index for article embeddings.
    Enables fast similarity search across all articles.

    For production with >100k articles, this should be replaced
    with a proper vector database (Pinecone, Qdrant, pgvector).
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._embeddings: NDArray[np.float32] = np.array([])
        self._article_ids: list[str] = []

    async def add_article(self, article_id: str, text: str):
        """Add an article to the index."""
        embedding = await self.embedding_service.embed_text(text)

        if len(self._embeddings) == 0:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

        self._article_ids.append(article_id)

    async def add_articles_batch(self, articles: list[tuple[str, str]]):
        """Add multiple articles to the index."""
        if not articles:
            return

        ids, texts = zip(*articles)
        embeddings = await self.embedding_service.embed_texts(list(texts))

        if len(self._embeddings) == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._article_ids.extend(ids)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Find most similar articles to a query embedding.

        Returns:
            List of (article_id, similarity_score) tuples
        """
        if len(self._embeddings) == 0:
            return []

        similarities = np.dot(self._embeddings, query_embedding)

        # Apply exclusion filter
        if exclude_ids:
            for i, article_id in enumerate(self._article_ids):
                if article_id in exclude_ids:
                    similarities[i] = -np.inf

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            (self._article_ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > -np.inf
        ]

    def clear(self):
        """Clear the index."""
        self._embeddings = np.array([])
        self._article_ids = []

    @property
    def size(self) -> int:
        """Number of articles in the index."""
        return len(self._article_ids)
