"""Main data processing orchestrator"""

import pandas as pd
import io
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .types import ProcessingResult, CSVStructure
from .csv_analyzer import CSVAnalyzer
from .question_extractor import QuestionExtractor

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing service orchestrator"""
    
    def __init__(self):
        self.active_processes: Dict[str, Dict] = {}
        self.csv_analyzer = CSVAnalyzer()
        self.question_extractor = QuestionExtractor()
    
    async def process_questions_file(
        self, 
        file_content: bytes,
        filename: str,
        processing_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded questions CSV file through the complete pipeline.
        
        Returns processing results with statistics and any errors.
        """
        
        if progress_callback:
            await progress_callback(processing_id, "validation", 10, "Validating file format...")
        
        try:
            # Parse CSV with error handling for malformed data
            content_str = file_content.decode('utf-8')
            
            # Try reading with different parsing options to handle malformed CSV
            try:
                # First, try standard parsing
                df = pd.read_csv(io.StringIO(content_str), header=None)
            except pd.errors.ParserError as e:
                # If that fails, try with error handling for bad lines
                logger.warning(f"CSV parsing error, trying with error handling: {e}")
                df = pd.read_csv(
                    io.StringIO(content_str), 
                    header=None, 
                    on_bad_lines='skip',  # Skip bad lines
                    engine='python'  # Use python engine for more flexibility
                )
            
            if len(df) == 0:
                raise ValueError("CSV file is empty after parsing")
            
            if progress_callback:
                await progress_callback(processing_id, "validation", 20, f"Loaded {len(df)} rows")
            
            # Analyze structure
            structure, df = self.csv_analyzer.detect_csv_structure(df)
            
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 30, "Analyzing data structure...")
            
            if structure.question_column is None:
                raise ValueError("Could not identify question column in CSV")
            
            # Extract and clean questions
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 40, "Extracting questions...")
            
            questions, kwargs_processed, cleaning_errors = self.question_extractor.extract_questions_from_dataframe(
                df, structure.question_column
            )
            
            # Progress update
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 70, f"Extracted {len(questions)} valid questions")
            
            # Remove duplicates based on timestamp + question
            if progress_callback:
                await progress_callback(processing_id, "deduplication", 75, "Removing duplicates...")
            
            original_count = len(questions)
            questions = self.question_extractor.remove_duplicates(questions)
            duplicates_removed = original_count - len(questions)
            
            if progress_callback:
                await progress_callback(processing_id, "deduplication", 78, f"Removed {duplicates_removed} duplicates, {len(questions)} unique questions remain")
            
            # Write to Database (with deduplication)
            database_result = None
            database_error = None
            
            if progress_callback:
                await progress_callback(processing_id, "database_write", 80, "Writing to database...")
            
            try:
                # Import questions service
                from app.services.questions import get_questions_service
                
                # Write questions to database with deduplication
                questions_service = await get_questions_service()
                if questions_service is not None:
                    database_result = await questions_service.create_questions_with_deduplication(
                        questions=questions,
                        processing_id=processing_id
                    )
                    
                    if progress_callback:
                        await progress_callback(processing_id, "database_write", 90, f"Successfully wrote {database_result['rows_written']} questions to database")
                else:
                    raise Exception("Database service not available")
                
            except Exception as e:
                database_error = {
                    "type": "database_error",
                    "message": str(e)
                }
                logger.error(f"Database write failed: {e}")
                
                if progress_callback:
                    await progress_callback(processing_id, "database_error", 85, f"Database write failed: {str(e)}")
            
            # Sync to Google Sheets (for reporting)
            sheets_result = None
            sheets_error = None
            
            if database_result and database_result["rows_written"] > 0:
                if progress_callback:
                    await progress_callback(processing_id, "sheets_sync", 95, "Syncing to Google Sheets...")
                
                try:
                    # Sync database to Google Sheets
                    questions_service = await get_questions_service()
                    if questions_service is not None:
                        sync_result = await questions_service.sync_to_google_sheets()
                        
                        if sync_result["status"] == "success":
                            sheets_result = {
                                "rows_written": database_result["rows_written"],
                                "duplicates_skipped": database_result["duplicates_skipped"],
                                "total_processed": database_result["total_processed"],
                                "sync_method": "database_to_sheets"
                            }
                            
                            if progress_callback:
                                await progress_callback(processing_id, "sheets_sync", 98, f"Successfully synced {sync_result['questions_synced']} questions to Google Sheets")
                        else:
                            sheets_error = {
                                "type": "sync_error",
                                "message": sync_result["error"]
                            }
                    else:
                        sheets_error = {
                            "type": "sync_error",
                            "message": "Questions service not available"
                        }
                            
                except Exception as e:
                    sheets_error = {
                        "type": "sync_error", 
                        "message": str(e)
                    }
                    logger.warning(f"Google Sheets sync failed (non-critical): {e}")
                    
                    if progress_callback:
                        await progress_callback(processing_id, "sheets_error", 96, f"Google Sheets sync failed: {str(e)}")
            else:
                # No new data to sync
                sheets_result = {
                    "rows_written": 0,
                    "duplicates_skipped": database_result["duplicates_skipped"] if database_result else 0,
                    "total_processed": len(questions),
                    "sync_method": "no_sync_needed"
                }
            
            # Handle case where database failed
            if database_error:
                logger.error(f"Database write failed: {database_error['message']}")
                
                if progress_callback:
                    await progress_callback(processing_id, "error", 85, f"Processing failed: {database_error['message']}")
                
                # Return error response
                results = {
                    "processing_id": processing_id,
                    "filename": filename,
                    "status": "failed",
                    "error": database_error,
                    "database": {
                        "write_attempted": True,
                        "write_successful": False,
                        "write_result": None,
                        "write_error": database_error
                    },
                    "google_sheets": {
                        "sync_attempted": False,
                        "write_successful": False,
                        "write_result": None,
                        "write_error": None
                    },
                    "completed_at": datetime.now().isoformat()
                }
                
                self.active_processes[processing_id] = results
                return results
            
            # Always complete processing, even if Google Sheets failed
            if progress_callback:
                completion_message = "Processing completed successfully!"
                if sheets_error:
                    completion_message = "Processing completed (Google Sheets unavailable)"
                await progress_callback(processing_id, "completion", 100, completion_message)
            
            # Store results
            results = {
                "processing_id": processing_id,
                "filename": filename,
                "status": "completed",
                "statistics": {
                    "total_rows_processed": int(len(df)),  # Convert to Python int
                    "questions_before_deduplication": int(original_count),
                    "duplicates_removed": int(duplicates_removed),
                    "valid_questions_extracted": int(len(questions)),
                    "kwargs_rows_processed": int(kwargs_processed),
                    "cleaning_errors": int(len(cleaning_errors)),
                    "success_rate": float(len(questions) / len(df)) if len(df) > 0 else 0.0
                },
                "structure_analysis": self._convert_numpy_types(structure.to_dict()),
                "sample_questions": questions[:5],  # First 5 for preview
                "errors": cleaning_errors[:10],  # First 10 errors
                "database": {
                    "write_attempted": True,
                    "write_successful": database_result is not None,
                    "write_result": database_result,
                    "write_error": database_error
                },
                "google_sheets": {
                    "sync_attempted": database_result is not None and database_result.get("rows_written", 0) > 0,
                    "write_successful": sheets_result is not None,
                    "write_result": sheets_result,
                    "write_error": sheets_error
                },
                "completed_at": datetime.now().isoformat()
            }
            
            # Store in active processes
            self.active_processes[processing_id] = results
            
            return results
            
        except Exception as e:
            error_result = {
                "processing_id": processing_id,
                "filename": filename,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
            
            self.active_processes[processing_id] = error_result
            
            if progress_callback:
                await progress_callback(processing_id, "error", 0, f"Processing failed: {str(e)}")
            
            raise
    
    def get_processing_status(self, processing_id: str) -> Optional[Dict]:
        """Get status of a processing job"""
        return self.active_processes.get(processing_id)
    
    def get_all_processing_history(self) -> List[Dict]:
        """Get history of all processing jobs"""
        return list(self.active_processes.values())
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

# Legacy alias for backward compatibility
DataProcessingService = DataProcessor