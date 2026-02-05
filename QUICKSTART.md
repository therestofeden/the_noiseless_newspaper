# The Noiseless Newspaper - Quick Start Guide

Get your newspaper live in ~30 minutes.

---

## Step 1: Create Supabase Project (5 min)

1. Go to [supabase.com](https://supabase.com) and sign in
2. Click "New Project"
3. Choose a name (e.g., `noiseless-newspaper`)
4. Set a strong database password (save it!)
5. Select a region close to you
6. Wait for project to be created (~2 min)

### Run Database Schema

1. In Supabase dashboard, go to **SQL Editor**
2. Copy contents of `backend/app/db/schema.sql`
3. Paste and click **Run**
4. You should see "Success. No rows returned"

### Get Connection Info

Go to **Settings → Database** and note:
- **Connection string (URI)** - use "Transaction" pooler
- Copy and replace `[YOUR-PASSWORD]` with your actual password

Go to **Settings → API** and note:
- **Project URL** (e.g., `https://xxx.supabase.co`)
- **anon/public** key
- **service_role** key (keep secret!)

---

## Step 2: Deploy Backend to Railway (10 min)

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your `the_noiseless_newspaper` repo
4. Choose the `backend` folder as root

### Add Environment Variables

In Railway project → **Variables**, add:

```
DATABASE_URL=postgresql+asyncpg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
SUPABASE_URL=https://[PROJECT-REF].supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
ENVIRONMENT=production
DEBUG=false
```

### Generate Domain

1. Go to **Settings → Networking**
2. Click **"Generate Domain"**
3. Note your URL (e.g., `https://noiseless-backend-production.up.railway.app`)

### Verify Deployment

Visit `https://your-railway-url.railway.app/health`

Should return:
```json
{"status": "healthy", "service": "noiseless-newspaper", "version": "0.1.0"}
```

---

## Step 3: Deploy Frontend to Vercel (5 min)

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **"Add New Project"**
3. Import your `the_noiseless_newspaper` repo
4. Set **Root Directory** to `frontend`
5. Click **Deploy**

### Update API URL

After Railway is deployed, edit `frontend/index.html`:

Find this line:
```javascript
const API_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000/api/v1'
  : 'https://your-railway-app.railway.app/api/v1';  // TODO: Replace with actual Railway URL
```

Replace with your actual Railway URL:
```javascript
const API_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000/api/v1'
  : 'https://noiseless-backend-production.up.railway.app/api/v1';
```

Commit and push - Vercel will auto-redeploy.

---

## Step 4: Run Initial Ingestion (5 min)

The backend has a scheduler that runs every few hours, but for the first run:

### Option A: Trigger via API (if in development mode)
```bash
curl -X POST https://your-railway-url/api/v1/admin/run-ingestion
```

### Option B: Run manually on Railway
1. Go to Railway dashboard
2. Open your service
3. Click **"New"** → **"One-off command"**
4. Run: `python -c "from app.jobs.daily_ingestion import DailyIngestionJob; import asyncio; asyncio.run(DailyIngestionJob().run())"`

---

## Step 5: Test It! (5 min)

1. Visit your Vercel frontend URL
2. Click "Begin" on the landing page
3. Select some topics
4. You should see a daily article!

---

## Troubleshooting

### "No articles available"
- Ingestion hasn't run yet
- Wait for scheduled job or trigger manually

### Frontend shows loading forever
- Check browser console for errors
- Verify API URL is correct
- Check Railway logs for backend errors

### Database connection failed
- Verify DATABASE_URL is correct
- Make sure you used the pooler connection string
- Check password doesn't have special characters that need escaping

### CORS errors
- Backend allows all origins by default
- Check Railway logs for actual error

---

## Next Steps

1. **Buy a domain** - Point it to Vercel
2. **Invite beta users** - Share the URL
3. **Monitor** - Check Railway/Supabase dashboards
4. **Iterate** - Gather feedback, improve

---

## Useful Commands

### Local Development

```bash
# Backend
cd backend
pip install -e .
cp .env.example .env  # Edit with your values
uvicorn app.main:app --reload

# Frontend (just open in browser)
open frontend/index.html
```

### Database

```bash
# Connect to Supabase via psql
psql "postgresql://postgres.[PROJECT-REF]:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"

# View articles
SELECT title, source, published_at FROM articles ORDER BY published_at DESC LIMIT 10;

# View users
SELECT * FROM user_profiles;
```

---

## Estimated Costs

| Service | Free Tier | Paid |
|---------|-----------|------|
| Supabase | 500MB database, 1GB bandwidth | $25/mo for Pro |
| Railway | $5 free credit/month | ~$5-10/mo |
| Vercel | Unlimited for hobby | $20/mo for Pro |

**Total: $0-10/month** for MVP validation
