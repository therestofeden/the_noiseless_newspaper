# Noiseless Newspaper Backend

The backend API for The Noiseless Newspaper - a curated news aggregator with intelligent article ranking.

## Overview

This FastAPI backend provides:

- **Article Aggregation**: Fetches articles from multiple sources (arXiv, NewsAPI, etc.)
- **Intelligent Ranking**: Combines recency, votes, PageRank, and personalization
- **3-Level Taxonomy**: Organizes content into domains, subtopics, and niches
- **User Preferences**: Personalized article recommendations
- **Citation Graph**: NetworkX-based PageRank for academic importance

## Quick Start

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Installation

```bash
# Clone and navigate to backend
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (API keys are optional for development)
```

### Running the Server

```bash
# Development mode with auto-reload
python -m app.main

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Pydantic settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── taxonomy.py      # 3-level topic taxonomy
│   ├── models/
│   │   ├── __init__.py
│   │   ├── domain.py        # Pydantic models
│   │   └── database.py      # SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ranking.py       # Article ranking algorithm
│   │   └── citation_graph.py # PageRank computation
│   └── sources/
│       ├── __init__.py
│       └── mock.py          # Mock data adapter
├── pyproject.toml
├── .env.example
└── README.md
```

## API Endpoints

### Taxonomy

- `GET /api/v1/taxonomy` - Get full taxonomy structure

### User Preferences

- `GET /api/v1/users/{user_id}/preferences` - Get user preferences
- `PUT /api/v1/users/{user_id}/preferences` - Update preferences

### Daily Articles

- `GET /api/v1/users/{user_id}/daily-article` - Get personalized recommendations

### Votes

- `POST /api/v1/users/{user_id}/votes` - Create a vote
- `GET /api/v1/users/{user_id}/votes` - Get user's votes

### Stats

- `GET /api/v1/users/{user_id}/stats` - Get user engagement stats

### Health

- `GET /api/v1/health` - Health check endpoint

## Ranking Algorithm

The ranking system combines multiple signals:

### Score Components

1. **Recency Score** (20%): Exponential decay based on publication time
   ```
   score = exp(-ln(2) * hours / half_life)
   ```

2. **Vote Score** (40%): Wilson score interval with time-weighted votes
   ```
   score = wilson_lower_bound * time_weight_factor
   ```

3. **PageRank Score** (30%): Citation-based importance
   ```
   score = normalized_pagerank(citation_graph)
   ```

4. **Topic Match Score** (10%): User preference alignment

### Cold-Start Handling

Uses a sigmoid lambda function to transition from prior-based ranking (new articles) to vote-based ranking (mature articles):

```
lambda = 1 / (1 + exp(-k * (votes - midpoint)))
final_score = lambda * mature_score + (1 - lambda) * prior_score
```

## Taxonomy

The system uses a 3-level taxonomy:

### Domains
- **ai-ml**: AI & Machine Learning
- **physics**: Physics
- **biotech**: Biotechnology
- **economics**: Economics
- **politics**: Politics & Policy

### Example Path
```
ai-ml.llms.architectures
│     │    └── Niche: Model Architectures
│     └── Subtopic: Large Language Models
└── Domain: AI & Machine Learning
```

## Database

Uses SQLAlchemy with async support. Default is SQLite for development; PostgreSQL recommended for production.

### Models

- `DBArticle`: Stored articles with metadata
- `DBUser`: User accounts
- `DBVote`: User votes (upvote/downvote/bookmark)
- `DBClick`: Article read tracking
- `DBUserPreferences`: User topic preferences
- `DBCitation`: Article citation relationships
- `DBArticleVoteStats`: Aggregated vote statistics

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy app/

# Format
ruff format .
```

## Environment Variables

See `.env.example` for all configuration options. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | SQLite |
| `DEBUG` | Enable debug mode | `false` |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `RANKING_RECENCY_HALF_LIFE_HOURS` | Recency decay half-life | 24 |
| `RANKING_WEIGHT_VOTES` | Vote component weight | 0.4 |

## License

MIT
