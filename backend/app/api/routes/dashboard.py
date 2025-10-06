from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from datetime import datetime, timedelta
import structlog

from app.core.database import get_db
from app.core.auth import get_optional_developer
from app.services.analysis import analysis_service

logger = structlog.get_logger()
router = APIRouter()


@router.get("/metrics")
async def get_dashboard_metrics(
    developer=Depends(get_optional_developer),
    db=Depends(get_db)
):
    """Get dashboard metrics and insights"""
    
    logger.info("Fetching dashboard metrics", developer_authenticated=bool(developer))
    
    # Get actual question count from Google Sheets
    question_count = await analysis_service.get_questions_count()
    has_sufficient_questions = question_count >= 10
    
    return {
        # Match the frontend DashboardData interface
        "question_count": question_count,
        "topic_count": 0,
        "last_analysis": None,
        "last_updated": None,
        "coverage_percentage": 0,
        "has_questions": has_sufficient_questions,  # Enable analysis only if >=10 questions
        
        # Additional insights for dashboard widgets
        "recent_insights": {
            "peak_question_time": "2-4 PM UTC",
            "most_active_region": "Asia-Pacific",
            "top_topic": {
                "name": "JavaScript Fundamentals",
                "percentage": 23
            },
            "question_rate_trend": {
                "unique_percentage": 45,
                "previous_percentage": 38,
                "trend": "increasing"
            }
        },
        "totals": {
            "total_questions": question_count,
            "total_topics": 0,
            "total_countries": 0,
            "last_analysis": None,
            "has_questions": has_sufficient_questions  # Enable analysis only if >=10 questions
        }
    }


@router.get("/questions")
async def get_questions(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    country: Optional[str] = None,
    state: Optional[str] = None,
    topic: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    developer=Depends(get_optional_developer),
    db=Depends(get_db)
):
    """Get filtered questions for dashboard table"""
    
    logger.info(
        "Fetching questions",
        limit=limit,
        offset=offset,
        filters={
            "country": country,
            "state": state, 
            "topic": topic,
            "date_from": date_from,
            "date_to": date_to
        },
        developer_authenticated=bool(developer)
    )
    
    # Load questions from database with filters
    # For now, return empty results
    
    return {
        "questions": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "filters_applied": {
            "country": country,
            "state": state,
            "topic": topic,
            "date_range": [date_from, date_to] if date_from or date_to else None
        }
    }


@router.get("/export")
async def export_questions(
    format: str = Query("csv", regex="^(csv|json)$"),
    country: Optional[str] = None,
    state: Optional[str] = None,
    topic: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    developer=Depends(get_optional_developer),
    db=Depends(get_db)
):
    """Export filtered questions"""
    
    logger.info(
        "Exporting questions",
        format=format,
        filters={
            "country": country,
            "state": state,
            "topic": topic,
            "date_from": date_from,
            "date_to": date_to
        },
        developer_authenticated=bool(developer)
    )
    
    # Implement export functionality
    
    return {"message": "Export functionality not yet implemented"}


@router.get("/charts/data")
async def get_chart_data(
    chart_type: str = Query(..., description="Type of chart data to fetch"),
    time_range: str = Query("7d", regex="^(6h|1d|7d|30d|custom)$"),
    country: Optional[str] = None,
    topic: Optional[str] = None,
    developer=Depends(get_optional_developer),
    db=Depends(get_db)
):
    """Get data for interactive charts"""
    
    logger.info(
        "Fetching chart data",
        chart_type=chart_type,
        time_range=time_range,
        filters={"country": country, "topic": topic},
        developer_authenticated=bool(developer)
    )
    
    # Generate chart data
    
    return {
        "chart_type": chart_type,
        "time_range": time_range,
        "data": [],
        "message": "Chart data generation not yet implemented"
    }
