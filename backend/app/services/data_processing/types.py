"""Type definitions for data processing"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

class ProcessingResult:
    """Result of CSV processing operation"""
    
    def __init__(
        self,
        processing_id: str,
        filename: str,
        status: str,
        statistics: Optional[Dict[str, Any]] = None,
        structure_analysis: Optional[Dict[str, Any]] = None,
        sample_questions: Optional[List[Dict]] = None,
        errors: Optional[List[str]] = None,
        database: Optional[Dict[str, Any]] = None,
        google_sheets: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        completed_at: Optional[str] = None
    ):
        self.processing_id = processing_id
        self.filename = filename
        self.status = status
        self.statistics = statistics or {}
        self.structure_analysis = structure_analysis or {}
        self.sample_questions = sample_questions or []
        self.errors = errors or []
        self.database = database or {}
        self.google_sheets = google_sheets or {}
        self.error = error
        self.completed_at = completed_at or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "processing_id": self.processing_id,
            "filename": self.filename,
            "status": self.status,
            "completed_at": self.completed_at
        }
        
        if self.status == "completed":
            result.update({
                "statistics": self.statistics,
                "structure_analysis": self.structure_analysis,
                "sample_questions": self.sample_questions,
                "errors": self.errors,
                "database": self.database,
                "google_sheets": self.google_sheets
            })
        elif self.status == "failed":
            result["error"] = self.error
        
        return result

class CSVStructure:
    """Information about CSV file structure"""
    
    def __init__(
        self,
        has_headers: bool = False,
        columns: Optional[Dict[str, Any]] = None,
        total_rows: int = 0,
        question_column: Optional[int] = None,
        kwargs_rows: int = 0
    ):
        self.has_headers = has_headers
        self.columns = columns or {}
        self.total_rows = total_rows
        self.question_column = question_column
        self.kwargs_rows = kwargs_rows
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "has_headers": self.has_headers,
            "columns": self.columns,
            "total_rows": self.total_rows,
            "question_column": self.question_column,
            "kwargs_rows": self.kwargs_rows
        }

class ColumnValidation:
    """Column validation result"""
    
    def __init__(self, is_valid: bool, errors: List[str], column_info: Dict[str, Any]):
        self.is_valid = is_valid
        self.errors = errors
        self.column_info = column_info