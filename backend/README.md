# The Noiseless Newspaper - Backend

> Less (noise) is More. Signal survives time.

A retrieval and ranking system that surfaces one high-signal article per day, chosen by what matters over time rather than what trends right now.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│  /api/v1/taxonomy          - Topic hierarchy                    │
│  /api/v1/users/.../preferences - User topic selections          │
│  /api/v1/users/.../daily-article - THE daily article            │
│  /api/v1/users/.../suggestions - Smart topic suggestions        │
│  /api/v1/users/.../votes   - Time-delayed relevance voting      │
│  /api/v1/users/.../stats   - User signal score & history        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Ranking Service                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  PageRank    │ │   Recency    │ │   Topic      │            │
│  │  Citation    │ │   Decay      │ │   Embedding  │            │
│  │  Score       │ │   Score      │ │   Similarity │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Final Score = λ * VoteScore + (1-λ) * CitationScore     │  │
│  │  λ increases as article accumulates more votes           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Article Source Adapters                       │
│  ┌─────────┐ ┌─────────────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ arXiv   │ │ Semantic Scholar│ │ OpenAlex │ │   NewsAPI   │  │
│  │         │ │ (+ citations)   │ │          │ │             │  │
│  └─────────┘ └─────────────────┘ └──────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## The Algorithm

### Cold Start (New Articles)
Articles without user votes are ranked by **PageRank on the citation graph**:

```python
InitialScore = α × CitationCount + β × CitationVelocity + γ × SourceAuthority
# α=0.4, β=0.35, γ=0.25

PageRankScore = networkx.pagerank(citation_graph, alpha=0.85)
```

### Time-Weighted Voting
Users vote on relevance at three time intervals:

| Period | Weight | Rationale |
|--------|--------|-----------|
| 1 week | 15% | Initial impression, may be hype |
| 1 month | 35% | Some perspective gained |
| 1 year | 50% | True long-term signal |

### Lambda Transition
As articles accumulate votes, we shift from citation-based to vote-based scoring:

```python
λ = sigmoid(total_votes / threshold)  # Smooth transition

FinalScore = (1-λ) × CitationScore + λ × WeightedVoteScore
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip or uv

### Installation

```bash
# Clone and enter directory
cd noiseless-backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file:

```env
# Required for production, optional for development (uses mocks)
DATABASE_URL=sqlite+aiosqlite:///./data/noiseless.db

# API Keys (all optional - uses mock data if not provided)
SEMANTIC_SCHOLAR_API_KEY=your_key  # For citation data
NEWSAPI_KEY=your_key               # For news articles
OPENAI_API_KEY=your_key            # For embeddings
ANTHROPIC_API_KEY=your_key         # For summaries

# Environment
ENVIRONMENT=development
DEBUG=true
```

### Running

```bash
# Start the server
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
noiseless-backend/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── core/
│   │   └── taxonomy.py        # Topic hierarchy definition
│   ├── jobs/
│   │   └── daily_ingestion.py # Batch job for fetching articles
│   ├── models/
│   │   ├── database.py        # SQLAlchemy models
│   │   └── domain.py          # Domain/business models
│   ├── services/
│   │   ├── citation_graph.py  # PageRank computation
│   │   ├── embeddings.py      # Semantic similarity
│   │   ├── ranking.py         # Main ranking algorithm
│   │   └── summarization.py   # LLM summaries
│   ├── sources/
│   │   ├── arxiv.py           # arXiv API adapter
│   │   ├── semantic_scholar.py # Semantic Scholar adapter
│   │   ├── openalex.py        # OpenAlex adapter
│   │   ├── newsapi.py         # NewsAPI adapter
│   │   └── mock.py            # Mock data for development
│   ├── config.py              # Configuration management
│   └── main.py                # FastAPI application
├── tests/                     # Test suite
├── data/                      # SQLite database (gitignored)
├── pyproject.toml             # Dependencies
└── README.md
```

## Key Endpoints

### Get Daily Article
```http
GET /api/v1/users/{user_id}/daily-article?topic_path=ai-ml/llms/interpretability
```

Returns THE one article for today on the selected topic.

### Submit Vote
```http
POST /api/v1/users/{user_id}/votes
{
  "article_id": "arxiv:2401.12345",
  "period": "1_month",
  "score": 4
}
```

### Get Smart Suggestions
```http
GET /api/v1/users/{user_id}/suggestions
```

Returns personalized topic suggestions based on reading history.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Running the Ingestion Job Manually

```bash
# Via API (development mode only)
curl -X POST http://localhost:8000/api/v1/admin/run-ingestion

# Or directly
python -c "import asyncio; from app.jobs.daily_ingestion import run_daily_job; asyncio.run(run_daily_job())"
```

### Database Migrations

Using Alembic (when needed):

```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/noiseless

# Set all API keys
SEMANTIC_SCHOLAR_API_KEY=...
NEWSAPI_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## The Philosophy

> "What feels important today often isn't. We optimize for what you'll still care about in a year."

The Noiseless Newspaper is built on one core insight: **signal survives time**.

Most content platforms optimize for engagement, which selects for novelty and outrage. We optimize for **retrospective relevance** - content that users rate as important long after they first encountered it.

The longer you wait to vote, the more your vote counts. This inverts the typical engagement metric and creates a natural filter for lasting value.

---

Built with 🤫 by The Noiseless Team
