"""
Summarization service using LLMs (Claude or GPT).
Generates concise summaries for articles to display in the daily digest.
"""
import asyncio
from typing import Optional

from app.config import get_settings
from app.models.domain import Article


class SummarizationService:
    """
    Service for generating article summaries using LLMs.

    Summaries are generated once when articles are ingested,
    then cached in the database.
    """

    def __init__(self):
        self.settings = get_settings()
        self._anthropic_client = None
        self._openai_client = None

    async def initialize(self):
        """Initialize the LLM client (lazy loading)."""
        if self.settings.anthropic_api_key:
            await self._init_anthropic()
        elif self.settings.openai_api_key:
            await self._init_openai()

    async def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(
                api_key=self.settings.anthropic_api_key
            )
        except ImportError:
            pass

    async def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key
            )
        except ImportError:
            pass

    async def summarize(
        self,
        article: Article,
        max_length: int = 150,
    ) -> str:
        """
        Generate a concise summary for an article.

        The summary should:
        - Be 2-3 sentences
        - Capture the key finding/insight
        - Be accessible to educated non-experts
        - Avoid jargon where possible

        Args:
            article: Article to summarize
            max_length: Maximum words in summary

        Returns:
            Generated summary string
        """
        # If article already has an abstract, we can use that as fallback
        if not article.abstract:
            return article.title

        if self._anthropic_client:
            return await self._summarize_anthropic(article, max_length)
        elif self._openai_client:
            return await self._summarize_openai(article, max_length)
        else:
            return self._summarize_fallback(article, max_length)

    async def _summarize_anthropic(
        self,
        article: Article,
        max_length: int,
    ) -> str:
        """Generate summary using Claude."""
        prompt = self._build_prompt(article, max_length)

        try:
            response = await self._anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cheap for summaries
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except Exception as e:
            print(f"Anthropic summarization error: {e}")
            return self._summarize_fallback(article, max_length)

    async def _summarize_openai(
        self,
        article: Article,
        max_length: int,
    ) -> str:
        """Generate summary using GPT."""
        prompt = self._build_prompt(article, max_length)

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Fast and cheap for summaries
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI summarization error: {e}")
            return self._summarize_fallback(article, max_length)

    def _build_prompt(self, article: Article, max_length: int) -> str:
        """Build the summarization prompt."""
        return f"""Summarize this academic/news article in 2-3 sentences ({max_length} words max).
Focus on the key finding or insight. Make it accessible to educated non-experts.
Avoid jargon. Be concrete and specific about what was discovered or reported.

Title: {article.title}

Abstract/Content:
{article.abstract or 'No abstract available.'}

Summary:"""

    def _summarize_fallback(self, article: Article, max_length: int) -> str:
        """
        Fallback summarization without LLM.
        Uses first sentences of abstract.
        """
        if not article.abstract:
            return article.title

        # Take first 2-3 sentences
        sentences = article.abstract.replace("\n", " ").split(". ")
        summary_sentences = []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > max_length:
                break
            summary_sentences.append(sentence)
            word_count += len(words)

            if len(summary_sentences) >= 3:
                break

        summary = ". ".join(summary_sentences)
        if not summary.endswith("."):
            summary += "."

        return summary

    async def summarize_batch(
        self,
        articles: list[Article],
        max_length: int = 150,
    ) -> dict[str, str]:
        """
        Summarize multiple articles (with rate limiting).

        Returns:
            Dictionary mapping article_id to summary
        """
        results = {}

        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]

            # Process batch concurrently
            tasks = [self.summarize(article, max_length) for article in batch]
            summaries = await asyncio.gather(*tasks, return_exceptions=True)

            for article, summary in zip(batch, summaries):
                if isinstance(summary, Exception):
                    results[article.id] = self._summarize_fallback(article, max_length)
                else:
                    results[article.id] = summary

            # Small delay between batches
            if i + batch_size < len(articles):
                await asyncio.sleep(1)

        return results


class ContentClassificationService:
    """
    Service for classifying articles into topics using LLMs.

    This is used when an article doesn't clearly fit into our taxonomy
    based on its source category alone.
    """

    def __init__(self, summarization_service: SummarizationService):
        self.summarization = summarization_service

    async def classify_topic(
        self,
        article: Article,
        taxonomy: dict,
    ) -> Optional[str]:
        """
        Classify an article into the most appropriate topic path.

        Uses LLM to understand the article content and match it
        to our taxonomy.

        Returns:
            Topic path string (e.g., "ai-ml/llms/interpretability") or None
        """
        # Build taxonomy description for prompt
        taxonomy_desc = self._build_taxonomy_description(taxonomy)

        prompt = f"""Given this article, classify it into the most specific matching topic from the taxonomy below.
Return ONLY the topic path (e.g., "ai-ml/llms/interpretability"), nothing else.
If no topic fits well, return "none".

Article Title: {article.title}

Article Abstract:
{article.abstract or 'No abstract available.'}

Taxonomy:
{taxonomy_desc}

Topic path:"""

        # Use the summarization service's LLM client
        if self.summarization._anthropic_client:
            try:
                response = await self.summarization._anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text.strip().lower()
                if result != "none" and "/" in result:
                    return result
            except Exception:
                pass

        elif self.summarization._openai_client:
            try:
                response = await self.summarization._openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.choices[0].message.content.strip().lower()
                if result != "none" and "/" in result:
                    return result
            except Exception:
                pass

        return None

    def _build_taxonomy_description(self, taxonomy: dict) -> str:
        """Build a text description of the taxonomy for the prompt."""
        lines = []
        for cat_id, category in taxonomy.items():
            cat_name = category.get("name", cat_id)
            for sub_id, subtopic in category.get("subtopics", {}).items():
                sub_name = subtopic.get("name", sub_id)
                for niche_id, niche in subtopic.get("niches", {}).items():
                    niche_name = niche.get("name", niche_id)
                    path = f"{cat_id}/{sub_id}/{niche_id}"
                    lines.append(f"- {path}: {cat_name} > {sub_name} > {niche_name}")

        return "\n".join(lines)
