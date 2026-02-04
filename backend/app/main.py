"""
Main FastAPI application for The Noiseless Newspaper.
"""
import asyncio
from contextlib import asynccontextmanager

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_database
from app.config import get_settings
from app.jobs.daily_ingestion import DailyIngestionJob
from app.models.database import Database

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Global instances
database: Database = None
scheduler: AsyncIOScheduler = None
ingestion_job: DailyIngestionJob = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    global database, scheduler, ingestion_job

    settings = get_settings()

    # Initialize database
    logger.info("Initializing database", url=settings.database_url)
    database = Database(settings.database_url)
    await database.create_tables()
    set_database(database)

    # Initialize ingestion job
    logger.info("Initializing ingestion job")
    ingestion_job = DailyIngestionJob(database)
    await ingestion_job.initialize()

    # Initialize scheduler for batch jobs
    scheduler = AsyncIOScheduler()

    # Schedule daily ingestion job
    scheduler.add_job(
        run_daily_ingestion,
        CronTrigger(hour=settings.batch_job_hour, minute=settings.batch_job_minute),
        id="daily_ingestion",
        name="Daily Article Ingestion",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started",
        batch_job_time=f"{settings.batch_job_hour:02d}:{settings.batch_job_minute:02d} UTC",
    )

    # Run initial ingestion if database is empty
    if settings.environment == "development":
        async with database.async_session() as session:
            from sqlalchemy import select, func
            from app.models.database import DBArticle
            result = await session.execute(select(func.count(DBArticle.id)))
            count = result.scalar()
            if count == 0:
                logger.info("Database empty, running initial ingestion")
                asyncio.create_task(run_daily_ingestion())

    yield

    # Shutdown
    logger.info("Shutting down")
    if scheduler:
        scheduler.shutdown()


async def run_daily_ingestion():
    """Run the daily ingestion job."""
    global ingestion_job
    try:
        stats = await ingestion_job.run()
        logger.info("Daily ingestion completed", stats=stats)
    except Exception as e:
        logger.error("Daily ingestion failed", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="The Noiseless Newspaper",
    description="One article per day. Chosen by what matters over time.",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "noiseless-newspaper",
        "version": "0.1.0",
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "The Noiseless Newspaper API",
        "tagline": "Less (noise) is More.",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "taxonomy": "/api/v1/taxonomy",
            "preferences": "/api/v1/users/{user_id}/preferences",
            "daily_article": "/api/v1/users/{user_id}/daily-article",
            "suggestions": "/api/v1/users/{user_id}/suggestions",
            "votes": "/api/v1/users/{user_id}/votes",
            "stats": "/api/v1/users/{user_id}/stats",
        },
    }


# Manual trigger for ingestion (development only)
@app.post("/api/v1/admin/run-ingestion")
async def trigger_ingestion():
    """Manually trigger the ingestion job (for development/testing)."""
    settings = get_settings()
    if settings.environment != "development":
        return {"error": "Only available in development mode"}

    asyncio.create_task(run_daily_ingestion())
    return {"message": "Ingestion job started"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
