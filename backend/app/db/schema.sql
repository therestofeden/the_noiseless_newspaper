-- The Noiseless Newspaper - Database Schema
-- Run this in Supabase SQL Editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- USERS & AUTHENTICATION
-- ============================================

-- User profiles (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  display_name TEXT,
  timezone TEXT DEFAULT 'UTC',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User topic interests
CREATE TABLE IF NOT EXISTS user_interests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
  topic TEXT NOT NULL,           -- e.g., 'ai-ml', 'sports'
  subtopic TEXT,                 -- e.g., 'deep-learning', 'nfl'
  niche TEXT,                    -- e.g., 'transformers', 'playoffs'
  priority INT DEFAULT 1,        -- User's ranking of this interest
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, topic, subtopic, niche)
);

CREATE INDEX idx_user_interests_user ON user_interests(user_id);
CREATE INDEX idx_user_interests_topic ON user_interests(topic);

-- ============================================
-- ARTICLES
-- ============================================

-- Articles ingested from sources
CREATE TABLE IF NOT EXISTS articles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  -- Source identification
  external_id TEXT NOT NULL,           -- Source-specific ID (DOI, arXiv ID, URL hash)
  source_name TEXT NOT NULL,           -- e.g., 'arxiv', 'bbc', 'techcrunch'

  -- Content
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  abstract TEXT,
  content TEXT,                        -- Full content if available

  -- Metadata
  authors TEXT[] DEFAULT '{}',
  published_at TIMESTAMPTZ,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),

  -- Classification
  topics TEXT[] DEFAULT '{}',          -- e.g., ['ai-ml', 'physics']
  subtopics TEXT[] DEFAULT '{}',

  -- Academic identifiers
  doi TEXT,
  arxiv_id TEXT,
  pmid TEXT,

  -- Quality signals
  authority_score FLOAT DEFAULT 0.5,   -- Source authority (0-1)
  citation_count INT DEFAULT 0,
  peer_reviewed BOOLEAN DEFAULT FALSE,

  -- Computed scores (updated by ranking job)
  recency_score FLOAT DEFAULT 0,
  predicted_survival FLOAT DEFAULT 0.5,
  ranking_score FLOAT DEFAULT 0,

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(source_name, external_id)
);

CREATE INDEX idx_articles_topics ON articles USING GIN(topics);
CREATE INDEX idx_articles_published ON articles(published_at DESC);
CREATE INDEX idx_articles_ranking ON articles(ranking_score DESC);
CREATE INDEX idx_articles_source ON articles(source_name);

-- ============================================
-- DAILY SELECTIONS
-- ============================================

-- What article was shown to each user each day
CREATE TABLE IF NOT EXISTS daily_selections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
  article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  topic TEXT NOT NULL,
  selected_date DATE DEFAULT CURRENT_DATE,
  ranking_score_at_selection FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(user_id, topic, selected_date)
);

CREATE INDEX idx_daily_user ON daily_selections(user_id);
CREATE INDEX idx_daily_date ON daily_selections(selected_date DESC);
CREATE INDEX idx_daily_article ON daily_selections(article_id);

-- ============================================
-- VOTES (Time-Delayed Feedback)
-- ============================================

-- User votes on articles (asked at different time intervals)
CREATE TABLE IF NOT EXISTS votes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
  article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  selection_id UUID REFERENCES daily_selections(id) ON DELETE SET NULL,

  -- When the article was shown
  shown_at TIMESTAMPTZ NOT NULL,

  -- Vote timing
  vote_window TEXT NOT NULL,           -- '1_week', '1_month', '1_year'
  vote_due_at TIMESTAMPTZ NOT NULL,    -- When to ask for this vote

  -- The actual vote
  vote_value INT,                      -- NULL = not yet voted, -1/0/1 = bad/neutral/good
  voted_at TIMESTAMPTZ,

  -- For analysis
  time_to_vote_seconds INT,            -- How long after prompt did they vote

  created_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(user_id, article_id, vote_window)
);

CREATE INDEX idx_votes_pending ON votes(vote_due_at) WHERE vote_value IS NULL;
CREATE INDEX idx_votes_user ON votes(user_id);
CREATE INDEX idx_votes_article ON votes(article_id);

-- ============================================
-- AGGREGATED VOTE STATISTICS
-- ============================================

-- Pre-computed vote statistics per article (updated by background job)
CREATE TABLE IF NOT EXISTS article_vote_stats (
  article_id UUID PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,

  -- Vote counts by window
  votes_1_week_positive INT DEFAULT 0,
  votes_1_week_neutral INT DEFAULT 0,
  votes_1_week_negative INT DEFAULT 0,

  votes_1_month_positive INT DEFAULT 0,
  votes_1_month_neutral INT DEFAULT 0,
  votes_1_month_negative INT DEFAULT 0,

  votes_1_year_positive INT DEFAULT 0,
  votes_1_year_neutral INT DEFAULT 0,
  votes_1_year_negative INT DEFAULT 0,

  -- Computed scores
  total_votes INT DEFAULT 0,
  survival_score FLOAT DEFAULT 0,      -- Weighted combination

  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- INGESTION TRACKING
-- ============================================

-- Track ingestion runs for monitoring
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  started_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  status TEXT DEFAULT 'running',       -- 'running', 'completed', 'failed'

  articles_fetched INT DEFAULT 0,
  articles_new INT DEFAULT 0,
  articles_updated INT DEFAULT 0,
  articles_skipped INT DEFAULT 0,

  errors JSONB DEFAULT '[]',

  source_stats JSONB DEFAULT '{}'      -- Per-source statistics
);

CREATE INDEX idx_ingestion_runs_status ON ingestion_runs(status, started_at DESC);

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on user-specific tables
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_interests ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_selections ENABLE ROW LEVEL SECURITY;
ALTER TABLE votes ENABLE ROW LEVEL SECURITY;

-- User profiles: users can only see/edit their own
CREATE POLICY "Users can view own profile" ON user_profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON user_profiles
  FOR UPDATE USING (auth.uid() = id);

-- User interests: users can only manage their own
CREATE POLICY "Users can view own interests" ON user_interests
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own interests" ON user_interests
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own interests" ON user_interests
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own interests" ON user_interests
  FOR DELETE USING (auth.uid() = user_id);

-- Daily selections: users can view their own
CREATE POLICY "Users can view own selections" ON daily_selections
  FOR SELECT USING (auth.uid() = user_id);

-- Votes: users can view and manage their own
CREATE POLICY "Users can view own votes" ON votes
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own votes" ON votes
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own votes" ON votes
  FOR UPDATE USING (auth.uid() = user_id);

-- Articles: everyone can view (public read)
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Articles are publicly readable" ON articles
  FOR SELECT USING (true);

-- Article vote stats: public read
ALTER TABLE article_vote_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Vote stats are publicly readable" ON article_vote_stats
  FOR SELECT USING (true);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO user_profiles (id, email)
  VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to auto-create profile on signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- Function to update article vote stats
CREATE OR REPLACE FUNCTION update_article_vote_stats(p_article_id UUID)
RETURNS VOID AS $$
BEGIN
  INSERT INTO article_vote_stats (article_id)
  VALUES (p_article_id)
  ON CONFLICT (article_id) DO NOTHING;

  UPDATE article_vote_stats
  SET
    votes_1_week_positive = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_week' AND vote_value = 1),
    votes_1_week_neutral = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_week' AND vote_value = 0),
    votes_1_week_negative = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_week' AND vote_value = -1),

    votes_1_month_positive = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_month' AND vote_value = 1),
    votes_1_month_neutral = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_month' AND vote_value = 0),
    votes_1_month_negative = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_month' AND vote_value = -1),

    votes_1_year_positive = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_year' AND vote_value = 1),
    votes_1_year_neutral = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_year' AND vote_value = 0),
    votes_1_year_negative = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_window = '1_year' AND vote_value = -1),

    total_votes = (SELECT COUNT(*) FROM votes WHERE article_id = p_article_id AND vote_value IS NOT NULL),

    -- Survival score: weighted average (1-week: 15%, 1-month: 35%, 1-year: 50%)
    survival_score = (
      SELECT
        CASE WHEN COUNT(*) FILTER (WHERE vote_value IS NOT NULL) = 0 THEN 0.5
        ELSE (
          0.15 * COALESCE(AVG(vote_value) FILTER (WHERE vote_window = '1_week'), 0) +
          0.35 * COALESCE(AVG(vote_value) FILTER (WHERE vote_window = '1_month'), 0) +
          0.50 * COALESCE(AVG(vote_value) FILTER (WHERE vote_window = '1_year'), 0) +
          0.5  -- Center at 0.5 since vote_value is -1 to 1
        ) / 2 + 0.5
        END
      FROM votes
      WHERE article_id = p_article_id
    ),

    updated_at = NOW()
  WHERE article_id = p_article_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update stats when votes change
CREATE OR REPLACE FUNCTION trigger_update_vote_stats()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM update_article_vote_stats(COALESCE(NEW.article_id, OLD.article_id));
  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS votes_stats_trigger ON votes;
CREATE TRIGGER votes_stats_trigger
  AFTER INSERT OR UPDATE OR DELETE ON votes
  FOR EACH ROW EXECUTE FUNCTION trigger_update_vote_stats();

-- ============================================
-- INITIAL DATA (Topics Reference)
-- ============================================

-- Topics reference table (for validation and UI)
CREATE TABLE IF NOT EXISTS topics (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  icon TEXT,
  color TEXT,
  type TEXT DEFAULT 'deep',  -- 'deep' or 'news'
  display_order INT DEFAULT 0
);

INSERT INTO topics (id, name, icon, color, type, display_order) VALUES
  ('ai-ml', 'Artificial Intelligence', '◈', 'terra', 'deep', 1),
  ('physics', 'Physics', '◉', 'forest', 'deep', 2),
  ('economics', 'Economics & Finance', '◇', 'terra', 'deep', 3),
  ('biotech', 'Biotechnology', '◎', 'forest', 'deep', 4),
  ('politics', 'Politics & Policy', '◆', 'terra', 'deep', 5),
  ('sports', 'Sports', '⚽', 'sky', 'news', 6),
  ('entertainment', 'Entertainment & Culture', '🎬', 'violet', 'news', 7),
  ('technology', 'Technology', '📱', 'cyan', 'news', 8),
  ('business', 'Business & Markets', '📈', 'amber', 'news', 9),
  ('world', 'World News', '🌍', 'indigo', 'news', 10),
  ('environment', 'Environment & Climate', '🌱', 'emerald', 'news', 11)
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name,
  icon = EXCLUDED.icon,
  color = EXCLUDED.color,
  type = EXCLUDED.type,
  display_order = EXCLUDED.display_order;

-- Topics are public read
ALTER TABLE topics ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Topics are publicly readable" ON topics
  FOR SELECT USING (true);
