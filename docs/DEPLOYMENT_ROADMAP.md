# The Noiseless Newspaper - Deployment Roadmap

## Goal: Validate the Concept

Get a working product in front of real users to test if "one article per day that survives time" resonates.

---

## What You Have vs. What You Need

### ✅ Already Built
- Topic taxonomy (11 topics, 44 subtopics)
- Data ingestion system (110+ sources, RSS + APIs)
- Ranking algorithm (recency + quality + survival prediction)
- Survival prediction model architecture
- Frontend design (beautiful, distinctive)

### 🔨 Need to Build for MVP
1. **Database** - Store articles, users, votes
2. **API Server** - Serve articles to frontend
3. **Scheduled Ingestion** - Fetch articles automatically
4. **User Accounts** - Track interests and votes
5. **Email Delivery** - Send daily article (optional but powerful)

### 🚫 Skip for Now (Post-Validation)
- Payment/subscription system
- Advanced ML model training
- Mobile app
- Social features
- Admin dashboard

---

## Recommended Stack (Simple + Affordable)

| Component | Service | Cost | Why |
|-----------|---------|------|-----|
| **Frontend** | Vercel | Free | Auto-deploy from GitHub, great DX |
| **Backend API** | Railway | ~$5/mo | Simple Python deployment, scales |
| **Database** | Supabase | Free tier | Postgres + Auth built-in |
| **Scheduled Jobs** | Railway Cron | Included | Run ingestion every 2 hours |
| **Email** | Resend | Free (3k/mo) | Simple API, great deliverability |
| **Domain** | Any registrar | ~$12/yr | thenoislelessnewspaper.com? |

**Total: ~$5-10/month** (well under budget)

---

## MVP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│                   (Vercel - Static)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Landing  │  │  Topics  │  │  Daily   │  │  Voting  │    │
│  │   Page   │  │ Selector │  │ Article  │  │   UI     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND API                             │
│                    (Railway - FastAPI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  /auth   │  │/articles │  │  /vote   │  │  /user   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│       SUPABASE          │    │    SCHEDULED JOBS           │
│  ┌─────────────────┐    │    │    (Railway Cron)           │
│  │    Postgres     │    │    │  ┌───────────────────────┐  │
│  │  - articles     │    │    │  │ Ingest (every 2h)     │  │
│  │  - users        │    │    │  │ Rank (daily)          │  │
│  │  - votes        │    │    │  │ Email (daily 7am)     │  │
│  │  - interests    │    │    │  └───────────────────────┘  │
│  └─────────────────┘    │    └─────────────────────────────┘
│  ┌─────────────────┐    │
│  │  Auth (built-in)│    │
│  └─────────────────┘    │
└─────────────────────────┘
```

---

## Step-by-Step Deployment Plan

### Phase 1: Database Setup (Day 1)
1. Create Supabase project
2. Run database migrations (create tables)
3. Set up Row Level Security (RLS) policies
4. Get connection string

### Phase 2: Backend Deployment (Day 1-2)
1. Create Railway project
2. Connect GitHub repo
3. Configure environment variables
4. Deploy FastAPI backend
5. Test API endpoints

### Phase 3: Frontend Deployment (Day 2)
1. Create Vercel project
2. Connect GitHub repo
3. Configure API URL environment variable
4. Deploy frontend
5. Set up custom domain (optional)

### Phase 4: Scheduled Jobs (Day 2-3)
1. Set up Railway cron job for ingestion
2. Test ingestion runs successfully
3. Verify articles appear in database

### Phase 5: User Testing (Day 3+)
1. Create test account
2. Select topics
3. Verify daily article selection works
4. Test voting flow
5. Invite 5-10 beta users

---

## Database Schema (Simplified for MVP)

```sql
-- Users (Supabase Auth handles most of this)
CREATE TABLE user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  email TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- User topic interests
CREATE TABLE user_interests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES user_profiles(id),
  topic TEXT NOT NULL,  -- e.g., 'ai-ml', 'sports'
  subtopic TEXT,        -- e.g., 'deep-learning'
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, topic, subtopic)
);

-- Articles (ingested from sources)
CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id TEXT NOT NULL,
  source_name TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  abstract TEXT,
  authors TEXT[],
  published_at TIMESTAMP,
  fetched_at TIMESTAMP DEFAULT NOW(),
  topics TEXT[],           -- e.g., ['ai-ml', 'physics']
  doi TEXT,
  arxiv_id TEXT,
  authority_score FLOAT DEFAULT 0.5,
  ranking_score FLOAT DEFAULT 0,
  UNIQUE(source_name, external_id)
);

-- User votes (time-delayed)
CREATE TABLE votes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES user_profiles(id),
  article_id UUID REFERENCES articles(id),
  shown_at TIMESTAMP NOT NULL,      -- When user saw the article
  vote_window TEXT NOT NULL,        -- '1_week', '1_month', '1_year'
  vote_due_at TIMESTAMP NOT NULL,   -- When to ask for vote
  vote_value INT,                   -- NULL until voted, then -1, 0, 1
  voted_at TIMESTAMP,
  UNIQUE(user_id, article_id, vote_window)
);

-- Daily article selections (what was shown to each user)
CREATE TABLE daily_selections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES user_profiles(id),
  article_id UUID REFERENCES articles(id),
  topic TEXT NOT NULL,
  selected_at DATE DEFAULT CURRENT_DATE,
  UNIQUE(user_id, topic, selected_at)
);

-- Indexes for performance
CREATE INDEX idx_articles_topics ON articles USING GIN(topics);
CREATE INDEX idx_articles_published ON articles(published_at DESC);
CREATE INDEX idx_votes_due ON votes(vote_due_at) WHERE vote_value IS NULL;
CREATE INDEX idx_daily_user_date ON daily_selections(user_id, selected_at);
```

---

## API Endpoints (MVP)

```
POST   /auth/signup          - Create account (via Supabase)
POST   /auth/login           - Login
POST   /auth/logout          - Logout

GET    /user/profile         - Get user profile
PUT    /user/interests       - Update topic interests

GET    /articles/today       - Get today's article for user's topics
GET    /articles/topic/:id   - Get recent articles for a topic

GET    /votes/pending        - Get votes due (1-week, 1-month, 1-year)
POST   /votes/:article_id    - Submit a vote

GET    /topics               - List all topics and subtopics
```

---

## Environment Variables Needed

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...  # For backend only

# Optional: API keys for higher rate limits
SEMANTIC_SCHOLAR_API_KEY=
PUBMED_API_KEY=

# Email (Resend)
RESEND_API_KEY=re_xxx

# App config
ENVIRONMENT=production
FRONTEND_URL=https://thenoiselessnewspaper.com
```

---

## What Success Looks Like

### Week 1 Metrics
- [ ] 10+ users signed up
- [ ] Users selecting topics
- [ ] Daily articles being delivered
- [ ] Some votes coming in

### Week 2-4 Metrics
- [ ] Retention: users coming back daily
- [ ] Vote completion rate > 50%
- [ ] Qualitative feedback: "this is useful"

### Validation Questions to Answer
1. Do people actually want ONE article per day? Or is it too restrictive?
2. Do time-delayed votes feel meaningful or annoying?
3. Which topics get the most engagement?
4. What's the "aha moment" for users?

---

## Immediate Next Steps

1. **Create Supabase project** (10 min)
   - Go to supabase.com
   - Create new project
   - Run the SQL schema above

2. **Set up Railway** (15 min)
   - Go to railway.app
   - Connect GitHub repo
   - Deploy backend

3. **Deploy frontend to Vercel** (10 min)
   - Go to vercel.com
   - Import GitHub repo
   - Deploy

4. **Test end-to-end** (30 min)
   - Sign up as user
   - Select topics
   - See daily article
   - Vote

Total time to live: **~1-2 days** of focused work.

---

## Questions for You

Before we start building, I want to confirm:

1. Do you want email delivery of daily articles, or just web-only for MVP?
2. Do you have a domain name in mind, or should we use the free Vercel/Railway URLs first?
3. Should we start with Supabase setup now?
