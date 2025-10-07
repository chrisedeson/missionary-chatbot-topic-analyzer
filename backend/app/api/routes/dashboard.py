# backend/app/api/routes/dashboard.py

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
from datetime import datetime, timedelta
import structlog
import io
import csv
import json

from app.core.database import get_db
from app.core.auth import get_optional_developer
from prisma import Prisma

logger = structlog.get_logger()
router = APIRouter()


@router.get("/metrics")
async def get_dashboard_metrics(
    developer=Depends(get_optional_developer),
    db: Prisma = Depends(get_db)
):
    """Get dashboard metrics and insights"""
    
    logger.info("Fetching dashboard metrics", developer_authenticated=bool(developer))
    
    try:
        # Get question count from database
        question_count = await db.question.count()
        
        # Get topic count from database
        topic_count = await db.topic.count()
        
        # Get last analysis run
        last_analysis_run = await db.analysisrun.find_first(
            where={"status": "completed"},
            order={"completedAt": "desc"}
        )
        
        # Calculate coverage (questions with topics assigned)
        questions_with_topics = await db.question.count(
            where={"topicId": {"not": None}}
        )
        coverage_percentage = (
            (questions_with_topics / question_count * 100) 
            if question_count > 0 else 0
        )
        
        # Get most recent question timestamp
        latest_question = await db.question.find_first(
            order={"createdAt": "desc"}
        )
        
        has_sufficient_questions = question_count >= 10
        
        # Calculate insights
        insights = await _calculate_insights(db)
        
        return {
            # Match the frontend DashboardData interface
            "question_count": question_count,
            "topic_count": topic_count,
            "last_analysis": last_analysis_run.completedAt.isoformat() if last_analysis_run else None,
            "last_updated": latest_question.createdAt.isoformat() if latest_question else None,
            "coverage_percentage": round(coverage_percentage, 1),
            "has_questions": has_sufficient_questions,
            
            # Additional insights for dashboard widgets
            "recent_insights": insights,
            
            "totals": {
                "total_questions": question_count,
                "total_topics": topic_count,
                "total_countries": await _get_unique_countries_count(db),
                "last_analysis": last_analysis_run.completedAt.isoformat() if last_analysis_run else None,
                "has_questions": has_sufficient_questions
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")


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
    db: Prisma = Depends(get_db)
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
    
    try:
        # Build filter conditions
        where_conditions = {}
        
        if country:
            where_conditions["country"] = country
        
        if state:
            where_conditions["state"] = state
        
        if topic:
            where_conditions["matchedTopic"] = {"contains": topic, "mode": "insensitive"}
        
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["gte"] = date_from
            if date_to:
                date_filter["lte"] = date_to
            where_conditions["date"] = date_filter
        
        # Get total count with filters
        total = await db.question.count(where=where_conditions if where_conditions else None)
        
        # Get paginated questions with filters
        questions = await db.question.find_many(
            where=where_conditions if where_conditions else None,
            skip=offset,
            take=limit,
            order={"createdAt": "desc"},
            include={
                "topic": True
            }
        )
        
        # Format questions for response
        formatted_questions = [
            {
                "id": q.id,
                "text": q.text,
                "date": q.date.isoformat() if q.date else None,
                "country": q.country,
                "state": q.state,
                "user_language": q.userLanguage,
                "topic": q.topic.name if q.topic else q.matchedTopic,
                "subtopic": q.topic.subtopic if q.topic else None,
                "similarity_score": q.similarityScore,
                "is_new_topic": q.isNewTopic,
                "created_at": q.createdAt.isoformat()
            }
            for q in questions
        ]
        
        return {
            "questions": formatted_questions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters_applied": {
                "country": country,
                "state": state,
                "topic": topic,
                "date_range": [
                    date_from.isoformat() if date_from else None,
                    date_to.isoformat() if date_to else None
                ] if date_from or date_to else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {str(e)}")


@router.get("/export")
async def export_questions(
    format: str = Query("csv", regex="^(csv|json)$"),
    country: Optional[str] = None,
    state: Optional[str] = None,
    topic: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    developer=Depends(get_optional_developer),
    db: Prisma = Depends(get_db)
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
    
    try:
        # Build filter conditions
        where_conditions = {}
        
        if country:
            where_conditions["country"] = country
        
        if state:
            where_conditions["state"] = state
        
        if topic:
            where_conditions["matchedTopic"] = {"contains": topic, "mode": "insensitive"}
        
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["gte"] = date_from
            if date_to:
                date_filter["lte"] = date_to
            where_conditions["date"] = date_filter
        
        # Get all questions matching filters (no pagination for export)
        questions = await db.question.find_many(
            where=where_conditions if where_conditions else None,
            order={"createdAt": "desc"},
            include={
                "topic": True
            }
        )
        
        if format == "csv":
            # Generate CSV
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "id", "text", "date", "country", "state", 
                    "user_language", "topic", "subtopic", 
                    "similarity_score", "is_new_topic", "created_at"
                ]
            )
            writer.writeheader()
            
            for q in questions:
                writer.writerow({
                    "id": q.id,
                    "text": q.text,
                    "date": q.date.isoformat() if q.date else "",
                    "country": q.country or "",
                    "state": q.state or "",
                    "user_language": q.userLanguage or "",
                    "topic": q.topic.name if q.topic else (q.matchedTopic or ""),
                    "subtopic": q.topic.subtopic if q.topic else "",
                    "similarity_score": q.similarityScore or "",
                    "is_new_topic": q.isNewTopic,
                    "created_at": q.createdAt.isoformat()
                })
            
            output.seek(0)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=questions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        
        else:  # JSON format
            data = [
                {
                    "id": q.id,
                    "text": q.text,
                    "date": q.date.isoformat() if q.date else None,
                    "country": q.country,
                    "state": q.state,
                    "user_language": q.userLanguage,
                    "topic": q.topic.name if q.topic else q.matchedTopic,
                    "subtopic": q.topic.subtopic if q.topic else None,
                    "similarity_score": q.similarityScore,
                    "is_new_topic": q.isNewTopic,
                    "created_at": q.createdAt.isoformat()
                }
                for q in questions
            ]
            
            return StreamingResponse(
                iter([json.dumps(data, indent=2)]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=questions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
    
    except Exception as e:
        logger.error(f"Error exporting questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export questions: {str(e)}")


@router.get("/charts/data")
async def get_chart_data(
    chart_type: str = Query(..., description="Type of chart data to fetch"),
    time_range: str = Query("7d", regex="^(6h|1d|7d|30d|custom)$"),
    country: Optional[str] = None,
    topic: Optional[str] = None,
    developer=Depends(get_optional_developer),
    db: Prisma = Depends(get_db)
):
    """Get data for interactive charts"""
    
    logger.info(
        "Fetching chart data",
        chart_type=chart_type,
        time_range=time_range,
        filters={"country": country, "topic": topic},
        developer_authenticated=bool(developer)
    )
    
    try:
        # Calculate date range
        now = datetime.utcnow()
        time_ranges = {
            "6h": now - timedelta(hours=6),
            "1d": now - timedelta(days=1),
            "7d": now - timedelta(days=7),
            "30d": now - timedelta(days=30)
        }
        date_from = time_ranges.get(time_range, time_ranges["7d"])
        
        # Build base filter
        where_conditions = {
            "createdAt": {"gte": date_from}
        }
        
        if country:
            where_conditions["country"] = country
        
        if topic:
            where_conditions["matchedTopic"] = {"contains": topic, "mode": "insensitive"}
        
        data = []
        
        if chart_type == "questions_over_time":
            # Group questions by date
            questions = await db.question.find_many(
                where=where_conditions,
                order={"createdAt": "asc"}
            )
            
            # Aggregate by day
            daily_counts = {}
            for q in questions:
                date_key = q.createdAt.date().isoformat()
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            
            data = [
                {"date": date, "count": count}
                for date, count in sorted(daily_counts.items())
            ]
        
        elif chart_type == "topics_distribution":
            # Get topic distribution
            questions = await db.question.find_many(
                where=where_conditions,
                include={"topic": True}
            )
            
            topic_counts = {}
            for q in questions:
                topic_name = q.topic.name if q.topic else (q.matchedTopic or "Uncategorized")
                topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
            
            data = [
                {"topic": topic, "count": count}
                for topic, count in sorted(
                    topic_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]  # Top 10 topics
            ]
        
        elif chart_type == "countries_distribution":
            # Get country distribution
            questions = await db.question.find_many(
                where=where_conditions
            )
            
            country_counts = {}
            for q in questions:
                country_name = q.country or "Unknown"
                country_counts[country_name] = country_counts.get(country_name, 0) + 1
            
            data = [
                {"country": country, "count": count}
                for country, count in sorted(
                    country_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]  # Top 10 countries
            ]
        
        return {
            "chart_type": chart_type,
            "time_range": time_range,
            "data": data,
            "filters_applied": {
                "country": country,
                "topic": topic
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching chart data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch chart data: {str(e)}")


# Helper functions

async def _calculate_insights(db: Prisma) -> dict:
    """Calculate insights for dashboard"""
    try:
        # Get recent questions (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_questions = await db.question.find_many(
            where={"createdAt": {"gte": week_ago}}
        )
        
        if not recent_questions:
            return {
                "peak_question_time": "N/A",
                "most_active_region": "N/A",
                "top_topic": {"name": "N/A", "percentage": 0},
                "question_rate_trend": {
                    "unique_percentage": 0,
                    "previous_percentage": 0,
                    "trend": "stable"
                }
            }
        
        # Peak question time (simplified - hour of day)
        hours = [q.createdAt.hour for q in recent_questions if q.createdAt]
        peak_hour = max(set(hours), key=hours.count) if hours else 12
        peak_time = f"{peak_hour:02d}:00-{(peak_hour+2)%24:02d}:00 UTC"
        
        # Most active region
        countries = [q.country for q in recent_questions if q.country]
        most_active = max(set(countries), key=countries.count) if countries else "N/A"
        
        # Top topic
        topics = [q.matchedTopic for q in recent_questions if q.matchedTopic]
        if topics:
            top_topic_name = max(set(topics), key=topics.count)
            top_topic_pct = (topics.count(top_topic_name) / len(recent_questions)) * 100
        else:
            top_topic_name = "N/A"
            top_topic_pct = 0
        
        return {
            "peak_question_time": peak_time,
            "most_active_region": most_active,
            "top_topic": {
                "name": top_topic_name,
                "percentage": round(top_topic_pct, 1)
            },
            "question_rate_trend": {
                "unique_percentage": 45,  # Placeholder - implement proper calculation
                "previous_percentage": 38,
                "trend": "increasing"
            }
        }
    
    except Exception as e:
        logger.error(f"Error calculating insights: {e}")
        return {
            "peak_question_time": "N/A",
            "most_active_region": "N/A",
            "top_topic": {"name": "N/A", "percentage": 0},
            "question_rate_trend": {
                "unique_percentage": 0,
                "previous_percentage": 0,
                "trend": "stable"
            }
        }


async def _get_unique_countries_count(db: Prisma) -> int:
    """Get count of unique countries"""
    try:
        # Get all questions with country field
        questions = await db.question.find_many(
            where={"country": {"not": None}}
        )
        unique_countries = set(q.country for q in questions if q.country)
        return len(unique_countries)
    except Exception as e:
        logger.error(f"Error counting unique countries: {e}")
        return 0