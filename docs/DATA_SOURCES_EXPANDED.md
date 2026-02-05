# Data Sources - Expanded Topics

This document catalogs primary sources for the 6 new general news topics.

> **Reminder**: We prioritize sources with free APIs or RSS feeds. For news topics, RSS feeds are often the primary access method.

---

## 6. Sports ⚽

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **ESPN** | Sports News | RSS Feeds | Comprehensive US/global sports coverage |
| 2 | **BBC Sport** | Sports News | RSS Feeds | UK/global sports, football focus |
| 3 | **The Athletic** | In-depth Analysis | RSS (limited) | Premium sports journalism |
| 4 | **Sky Sports** | Sports News | RSS Feeds | UK sports, Premier League |
| 5 | **Bleacher Report** | Sports News | RSS Feeds | US sports, viral content |
| 6 | **Sports Illustrated** | Sports Journalism | RSS Feeds | Long-form sports writing |
| 7 | **FIFA** | Official | RSS/Press | Football/soccer official news |
| 8 | **Olympics.com** | Official | RSS Feeds | Olympic sports, events |
| 9 | **ESPN FC** | Football | RSS Feeds | Soccer-specific coverage |
| 10 | **The Ringer** | Analysis/Culture | RSS Feeds | Sports + pop culture intersection |

### RSS Feeds

```python
SPORTS_FEEDS = {
    "espn": [
        "https://www.espn.com/espn/rss/news",
        "https://www.espn.com/espn/rss/nfl/news",
        "https://www.espn.com/espn/rss/nba/news",
        "https://www.espn.com/espn/rss/soccer/news",
    ],
    "bbc_sport": [
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://feeds.bbci.co.uk/sport/tennis/rss.xml",
    ],
    "sky_sports": [
        "https://www.skysports.com/rss/12040",  # Football
        "https://www.skysports.com/rss/12111",  # News
    ],
    "si": "https://www.si.com/rss/si_topstories.rss",
    "bleacher_report": "https://bleacherreport.com/articles/feed",
    "the_ringer": "https://www.theringer.com/rss/index.xml",
}
```

---

## 7. Entertainment & Culture 🎬

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **Variety** | Entertainment Trade | RSS Feeds | Film, TV, music industry |
| 2 | **The Hollywood Reporter** | Entertainment Trade | RSS Feeds | Movies, TV, entertainment business |
| 3 | **Deadline** | Breaking Entertainment | RSS Feeds | Industry news, deals |
| 4 | **Pitchfork** | Music | RSS Feeds | Album reviews, music news |
| 5 | **Rolling Stone** | Music/Culture | RSS Feeds | Music, pop culture |
| 6 | **The Guardian - Culture** | Arts/Culture | RSS Feeds | Books, art, film, music |
| 7 | **The A.V. Club** | Pop Culture | RSS Feeds | TV, film, games reviews |
| 8 | **Vulture** | Entertainment | RSS Feeds | TV, movies, comedy |
| 9 | **IndieWire** | Independent Film | RSS Feeds | Indies, festivals, art cinema |
| 10 | **Literary Hub** | Books/Literature | RSS Feeds | Book news, essays, reviews |

### RSS Feeds

```python
ENTERTAINMENT_FEEDS = {
    "variety": [
        "https://variety.com/feed/",
        "https://variety.com/v/film/feed/",
        "https://variety.com/v/tv/feed/",
    ],
    "hollywood_reporter": [
        "https://www.hollywoodreporter.com/feed/",
    ],
    "deadline": "https://deadline.com/feed/",
    "pitchfork": [
        "https://pitchfork.com/feed/feed-news/rss",
        "https://pitchfork.com/feed/feed-album-reviews/rss",
    ],
    "rolling_stone": "https://www.rollingstone.com/feed/",
    "guardian_culture": [
        "https://www.theguardian.com/culture/rss",
        "https://www.theguardian.com/books/rss",
        "https://www.theguardian.com/film/rss",
    ],
    "av_club": "https://www.avclub.com/rss",
    "vulture": "https://www.vulture.com/feed/rss/index.xml",
    "indiewire": "https://www.indiewire.com/feed/",
    "lithub": "https://lithub.com/feed/",
}
```

---

## 8. Technology 📱

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **TechCrunch** | Tech News | RSS Feeds | Startups, products, funding |
| 2 | **The Verge** | Consumer Tech | RSS Feeds | Gadgets, platforms, reviews |
| 3 | **Wired** | Tech/Culture | RSS Feeds | Tech trends, long-form |
| 4 | **Ars Technica** | Tech Deep Dives | RSS Feeds | Technical analysis, science |
| 5 | **Engadget** | Consumer Tech | RSS Feeds | Product news, reviews |
| 6 | **CNET** | Consumer Tech | RSS Feeds | Reviews, how-tos |
| 7 | **Hacker News** | Tech Community | API (free) | Developer/startup community |
| 8 | **Krebs on Security** | Cybersecurity | RSS Feeds | Security news, analysis |
| 9 | **9to5Mac/Google** | Apple/Google | RSS Feeds | Platform-specific news |
| 10 | **Protocol** | Tech Policy | RSS Feeds | Tech industry, policy |

### API & RSS Details

```python
TECHNOLOGY_FEEDS = {
    "techcrunch": "https://techcrunch.com/feed/",
    "verge": "https://www.theverge.com/rss/index.xml",
    "wired": "https://www.wired.com/feed/rss",
    "ars_technica": "https://feeds.arstechnica.com/arstechnica/index",
    "engadget": "https://www.engadget.com/rss.xml",
    "cnet": "https://www.cnet.com/rss/news/",
    "krebs": "https://krebsonsecurity.com/feed/",
    "9to5mac": "https://9to5mac.com/feed/",
    "9to5google": "https://9to5google.com/feed/",
}

# Hacker News API
HACKER_NEWS_API = {
    "base_url": "https://hacker-news.firebaseio.com/v0",
    "top_stories": "/topstories.json",
    "item": "/item/{id}.json",
    "rate_limit": None,  # No official limit
}
```

---

## 9. Business & Markets 📈

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **Reuters Business** | Wire Service | RSS Feeds | Global business news |
| 2 | **Bloomberg** | Financial News | RSS (limited) | Markets, companies, economy |
| 3 | **Financial Times** | Business News | RSS (limited) | Global business, markets |
| 4 | **Wall Street Journal** | Business News | RSS (limited) | US business, markets |
| 5 | **CNBC** | Financial News | RSS Feeds | Markets, investing |
| 6 | **Fortune** | Business Magazine | RSS Feeds | Corporate, leadership |
| 7 | **Harvard Business Review** | Management | RSS Feeds | Strategy, leadership |
| 8 | **Forbes** | Business | RSS Feeds | Billionaires, companies |
| 9 | **Crunchbase News** | Startups | RSS + API | Funding, acquisitions |
| 10 | **Axios** | Business/Policy | RSS Feeds | Concise business news |

### RSS Feeds

```python
BUSINESS_FEEDS = {
    "reuters_business": [
        "https://www.reuters.com/business/feed/",
        "https://www.reuters.com/markets/feed/",
    ],
    "cnbc": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",  # Markets
    ],
    "fortune": "https://fortune.com/feed/",
    "hbr": "https://hbr.org/feed",
    "forbes": "https://www.forbes.com/innovation/feed/",
    "axios": [
        "https://api.axios.com/feed/",
        "https://api.axios.com/feed/technology",
    ],
}

# Crunchbase API (requires API key for full access)
CRUNCHBASE_API = {
    "base_url": "https://api.crunchbase.com/v4",
    "news_endpoint": "/data/news",
    "requires_key": True,
}
```

---

## 10. World News 🌍

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **Reuters World** | Wire Service | RSS Feeds | Global news, breaking |
| 2 | **AP News** | Wire Service | RSS Feeds | Global news, breaking |
| 3 | **BBC World** | News | RSS Feeds | Global coverage |
| 4 | **Al Jazeera** | News | RSS Feeds | Middle East, global |
| 5 | **The Guardian World** | News | RSS Feeds | International coverage |
| 6 | **Foreign Policy** | Analysis | RSS Feeds | Geopolitics, diplomacy |
| 7 | **The Economist** | Analysis | RSS (limited) | Global analysis |
| 8 | **Council on Foreign Relations** | Think Tank | RSS Feeds | Foreign policy |
| 9 | **War on the Rocks** | Defense/Security | RSS Feeds | Military, strategy |
| 10 | **Stratfor** | Intelligence | RSS (limited) | Geopolitical analysis |

### RSS Feeds

```python
WORLD_NEWS_FEEDS = {
    "reuters_world": "https://www.reuters.com/world/feed/",
    "ap_world": "https://feeds.apnews.com/rss/world",
    "bbc_world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
        "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
        "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    ],
    "al_jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "foreign_policy": "https://foreignpolicy.com/feed/",
    "cfr": "https://www.cfr.org/rss.xml",
    "war_on_rocks": "https://warontherocks.com/feed/",
}

# GDELT Project (free global events database)
GDELT_API = {
    "base_url": "https://api.gdeltproject.org/api/v2",
    "doc_endpoint": "/doc/doc",
    "rate_limit": "Reasonable use",
}
```

---

## 11. Environment & Climate 🌱

| # | Source | Type | API Access | Content Coverage |
|---|--------|------|------------|------------------|
| 1 | **Carbon Brief** | Climate Journalism | RSS Feeds | Climate science, policy |
| 2 | **The Guardian Environment** | News | RSS Feeds | Environment news |
| 3 | **Yale Environment 360** | Analysis | RSS Feeds | Environmental journalism |
| 4 | **Grist** | Climate/Solutions | RSS Feeds | Climate, sustainability |
| 5 | **Inside Climate News** | Investigative | RSS Feeds | Climate investigations |
| 6 | **Nature Climate Change** | Academic | RSS Feeds | Climate research |
| 7 | **NOAA Climate.gov** | Official | RSS Feeds | Climate data, news |
| 8 | **NASA Climate** | Official | RSS Feeds | Climate science |
| 9 | **Mongabay** | Conservation | RSS Feeds | Biodiversity, forests |
| 10 | **Canary Media** | Clean Energy | RSS Feeds | Energy transition |

### RSS Feeds

```python
ENVIRONMENT_FEEDS = {
    "carbon_brief": "https://www.carbonbrief.org/feed/",
    "guardian_env": "https://www.theguardian.com/environment/rss",
    "yale_e360": "https://e360.yale.edu/feed",
    "grist": "https://grist.org/feed/",
    "inside_climate": "https://insideclimatenews.org/feed/",
    "nature_climate": "https://www.nature.com/nclimate.rss",
    "noaa_climate": "https://www.climate.gov/feeds/all",
    "nasa_climate": "https://climate.nasa.gov/feed/news",
    "mongabay": "https://news.mongabay.com/feed/",
    "canary_media": "https://www.canarymedia.com/feed/",
}

# OpenAQ API (air quality data)
OPENAQ_API = {
    "base_url": "https://api.openaq.org/v2",
    "measurements": "/measurements",
    "rate_limit": "Reasonable use",
}
```

---

## Source Quality Scoring

For news sources, authority scores are based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Editorial standards | 30% | Fact-checking, corrections policy |
| Original reporting | 25% | Primary sources vs aggregation |
| Longevity/reputation | 20% | Years in operation, awards |
| Transparency | 15% | Ownership, funding disclosure |
| Citation by others | 10% | Referenced by other outlets |

### Suggested Authority Scores

```python
SOURCE_AUTHORITY = {
    # Wire services (highest)
    "reuters": 0.95,
    "ap": 0.95,

    # Major outlets
    "bbc": 0.90,
    "guardian": 0.85,
    "nyt": 0.85,
    "ft": 0.90,
    "economist": 0.90,

    # Trade publications
    "variety": 0.85,
    "techcrunch": 0.80,
    "hbr": 0.90,

    # Specialized
    "carbon_brief": 0.90,
    "foreign_policy": 0.85,
    "ars_technica": 0.85,

    # Community/aggregation
    "hacker_news": 0.60,
    "bleacher_report": 0.65,
}
```

---

## Implementation Notes

### Combined Feed Configuration

```python
from app.services.data_ingestion.base import TopicDomain

ALL_RSS_FEEDS = {
    # Original topics (unchanged)
    TopicDomain.AI_ML: [...],
    TopicDomain.PHYSICS: [...],
    TopicDomain.ECONOMICS: [...],
    TopicDomain.BIOTECH: [...],
    TopicDomain.POLITICS: [...],

    # New topics
    TopicDomain.SPORTS: SPORTS_FEEDS,
    TopicDomain.ENTERTAINMENT: ENTERTAINMENT_FEEDS,
    TopicDomain.TECHNOLOGY: TECHNOLOGY_FEEDS,
    TopicDomain.BUSINESS: BUSINESS_FEEDS,
    TopicDomain.WORLD: WORLD_NEWS_FEEDS,
    TopicDomain.ENVIRONMENT: ENVIRONMENT_FEEDS,
}
```

### Fetch Frequency

| Topic Type | Fetch Interval | Rationale |
|------------|----------------|-----------|
| Academic (AI, Physics, etc.) | Every 6 hours | Research doesn't change rapidly |
| News (Sports, World, etc.) | Every 1-2 hours | Breaking news matters |
| Business/Markets | Every 30-60 min | Markets move fast |

---

## Total Source Count

| Category | Topics | Sources | RSS Feeds |
|----------|--------|---------|-----------|
| Original (Academic) | 5 | 50 | ~25 |
| New (News) | 6 | 60 | ~60 |
| **Total** | **11** | **110** | **~85** |
