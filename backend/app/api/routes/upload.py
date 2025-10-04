from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from typing import List
import structlog
import pandas as pd
import io
import os
from pathlib import Path

from app.core.database import get_db
from app.core.auth import get_current_developer
from app.core.config import settings

logger = structlog.get_logger()
router = APIRouter()

# Create upload directory if it doesn't exist
Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)


@router.post("/questions")
async def upload_questions_file(
    file: UploadFile = File(...),
    developer=Depends(get_current_developer),
    db=Depends(get_db)
):
    """Upload and validate questions CSV file (developer only)"""
    
    logger.info(
        "Questions file upload started",
        filename=file.filename,
        content_type=file.content_type,
        developer=developer["role"]
    )
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be CSV or TXT format"
        )
    
    # Validate file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Try to parse as CSV/text
        if file.filename.endswith('.csv'):
            # Handle CSV files - could be comma-separated questions or structured data
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                logger.info(f"CSV parsed successfully with {len(df)} rows and columns: {list(df.columns)}")
            except Exception as csv_error:
                # If CSV parsing fails, treat as line-by-line questions
                logger.info("CSV parsing failed, treating as line-by-line questions", error=str(csv_error))
                lines = content.decode('utf-8').strip().split('\n')
                questions = [line.strip().rstrip(',') for line in lines if line.strip()]
                df = pd.DataFrame({'question': questions})
        else:
            # Handle TXT files - line by line
            lines = content.decode('utf-8').strip().split('\n')
            questions = [line.strip() for line in lines if line.strip()]
            df = pd.DataFrame({'question': questions})
        
        # Basic validation
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File appears to be empty or contains no valid questions"
            )
        
        # Detect data format and quality issues
        validation_result = await validate_questions_data(df)
        
        # Save file for processing
        file_path = Path(settings.UPLOAD_DIR) / f"questions_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(
            "Questions file upload completed",
            filename=file.filename,
            rows_count=len(df),
            validation_result=validation_result
        )
        
        return {
            "message": "File uploaded and validated successfully",
            "filename": file.filename,
            "rows_count": len(df),
            "validation": validation_result,
            "file_path": str(file_path)
        }
        
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding not supported. Please use UTF-8 encoding."
        )
    except Exception as e:
        logger.error("File upload failed", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


async def validate_questions_data(df: pd.DataFrame) -> dict:
    """Validate questions data and detect quality issues"""
    
    validation = {
        "status": "valid",
        "warnings": [],
        "errors": [],
        "stats": {}
    }
    
    # Check for required columns or single question column
    if 'question' not in df.columns and len(df.columns) == 1:
        # Single column - assume it's questions
        df.columns = ['question']
        validation["warnings"].append("Assumed single column contains questions")
    elif 'question' not in df.columns:
        validation["errors"].append("No 'question' column found and multiple columns detected")
        validation["status"] = "error"
        return validation
    
    # Check for empty questions
    empty_questions = df['question'].isna().sum() + (df['question'] == '').sum()
    if empty_questions > 0:
        validation["warnings"].append(f"Found {empty_questions} empty questions")
    
    # Check for "kwargs" error rows (from langfuse errors)
    kwargs_rows = df['question'].str.contains('kwargs', na=False).sum()
    if kwargs_rows > 0:
        validation["warnings"].append(f"Found {kwargs_rows} rows with 'kwargs' - these may be error rows")
    
    # Check for very short questions (potential data quality issues)
    short_questions = (df['question'].str.len() < 5).sum()
    if short_questions > 0:
        validation["warnings"].append(f"Found {short_questions} very short questions (<5 characters)")
    
    # Check for expected columns in full format
    expected_cols = ['Date', 'Country', 'User Language', 'State', 'Question']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        validation["warnings"].append(f"Missing columns for full format: {missing_cols}")
    
    validation["stats"] = {
        "total_rows": len(df),
        "valid_questions": len(df) - empty_questions,
        "columns": list(df.columns),
        "sample_questions": df['question'].dropna().head(3).tolist()
    }
    
    return validation


@router.get("/uploads")
async def list_uploaded_files(
    developer=Depends(get_current_developer)
):
    """List uploaded files (developer only)"""
    
    upload_dir = Path(settings.UPLOAD_DIR)
    files = []
    
    for file_path in upload_dir.glob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            })
    
    return {
        "files": sorted(files, key=lambda x: x["modified"], reverse=True)
    }
