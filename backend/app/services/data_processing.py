"""
Data Processing Service for Questions Upload and Cleaning

Handles the complete pipeline:
1. CSV validation and parsing
2. Data cleaning and extraction from kwargs
3. Google Sheets integration
4. Progress tracking for UI updates
"""

import pandas as pd
import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import io

from app.core.config import settings
from app.services.google_sheets import google_sheets_service, ColumnMismatchError

logger = logging.getLogger(__name__)

class DataProcessingService:
    """Service for processing uploaded questions files"""
    
    def __init__(self):
        self.active_processes: Dict[str, Dict] = {}
    
    def clean_acm_prefix(self, question: str) -> str:
        """
        Remove ACM question prefix from questions before processing.
        
        This function removes prefixes like "(ACMs Question):" or "(ACM question):"
        that identify questions from ACM missionaries. These prefixes should be removed
        before processing to prevent clustering based on source rather than content.
        
        Args:
            question (str): The original question text
        
        Returns:
            str: The cleaned question text with ACM prefix removed
        """
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        # Pattern to match ACM prefixes (case-insensitive)
        # Patterns to match:
        # - (ACMs Question):
        # - (ACM question):
        # - (ACMs Question)
        # - (ACM question)
        # Add colon as optional to handle both formats
        pattern = r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*'
        
        # Remove the prefix and strip any remaining whitespace
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        # Return original question if nothing was removed and it's empty after cleaning
        return cleaned if cleaned else question
    
    def remove_duplicates(self, questions: List[Dict]) -> List[Dict]:
        """
        Remove duplicates based on timestamp and question text.
        
        Duplication is defined as having the same timestamp AND the same question text. This prevents 
        re-processing of questions that already exist in the system from past uploads, ensuring that 
        re-uploading the same data (e.g., from a file or backup) does not result in duplication.
        
        Args:
            questions: List of question dictionaries with metadata
        
        Returns:
            List of unique questions with duplicates removed
        """
        seen = set()
        unique_questions = []
        duplicates_removed = 0
        
        for question in questions:
            # Create a key from timestamp and cleaned question text
            timestamp = question.get('metadata', {}).get('date', '')
            question_text = question.get('extracted_question', '')
            
            # Clean the question text for comparison (remove ACM prefix, normalize whitespace)
            clean_text = self.clean_acm_prefix(question_text).strip().lower()
            
            # Create composite key for duplicate detection
            duplicate_key = (str(timestamp), clean_text)
            
            if duplicate_key not in seen:
                seen.add(duplicate_key)
                unique_questions.append(question)
            else:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate question: timestamp={timestamp}, question='{question_text[:50]}...'")
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate questions (same timestamp + question)")
        
        return unique_questions
    
    def extract_question_from_kwargs(self, kwargs_content: str) -> Optional[str]:
        """
        Extract the actual question from kwargs JSON structure.
        
        The real question is the first "content:" value in the kwargs.
        Other content is chatbot responses which we ignore.
        """
        try:
            # Try to parse as JSON
            if kwargs_content.startswith('{') or kwargs_content.startswith('['):
                data = json.loads(kwargs_content)
                
                # Look for the first "content" field
                if isinstance(data, dict):
                    if "content" in data:
                        return str(data["content"]).strip()
                    
                    # Sometimes it's nested in messages
                    if "messages" in data and isinstance(data["messages"], list):
                        for message in data["messages"]:
                            if isinstance(message, dict) and "content" in message:
                                return str(message["content"]).strip()
                
                elif isinstance(data, list) and len(data) > 0:
                    # If it's a list, take the first item's content
                    first_item = data[0]
                    if isinstance(first_item, dict) and "content" in first_item:
                        return str(first_item["content"]).strip()
            
            # If JSON parsing fails, try regex to extract content
            content_match = re.search(r'"content":\s*"([^"]*)"', kwargs_content)
            if content_match:
                return content_match.group(1).strip()
            
            # Last resort: return the original if it looks like a question
            if len(kwargs_content.strip()) > 5 and "?" in kwargs_content:
                return kwargs_content.strip()
                
        except Exception as e:
            logger.warning(f"Failed to extract question from kwargs: {e}")
        
        return None
    
    def clean_question_text(self, text: str) -> Optional[str]:
        """Clean and validate question text"""
        if not text or not isinstance(text, str):
            logger.debug(f"Skipping non-string or empty text: {repr(text)}")
            return None
        
        # First, remove ACM prefix if present
        text = self.clean_acm_prefix(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Skip empty strings after cleaning
        if not text:
            logger.debug(f"Skipping empty text after whitespace cleaning")
            return None
        
        try:
            # Remove common noise
            original_text = text
            text = re.sub(r'^(Question:\s*|Q:\s*|\d+\.\s*)', '', text, flags=re.IGNORECASE)
            
            # Must be at least 3 characters and not just numbers/symbols
            if len(text) < 3:
                logger.debug(f"Skipping text too short (<3 chars): '{original_text}' -> '{text}'")
                return None
            if re.match(r'^[\d\s\-_.,;:!?]*$', text):
                logger.debug(f"Skipping text with only numbers/symbols: '{original_text}' -> '{text}'")
                return None
        except Exception as e:
            logger.warning(f"Regex error in clean_question_text: {e}, text: '{text}'")
            return None
        
        logger.debug(f"Successfully cleaned text: '{original_text}' -> '{text}'")
        return text
    
    def detect_csv_structure(self, df: pd.DataFrame) -> tuple[Dict[str, Any], pd.DataFrame]:
        """
        Analyze CSV structure and detect columns.
        Handle cases with/without headers, different column orders.
        
        Returns:
            Tuple of (structure_info, processed_dataframe)
        """
        structure = {
            "has_headers": False,
            "columns": {},
            "total_rows": len(df),
            "question_column": None,
            "kwargs_rows": 0
        }
        
        # Check if first row looks like headers
        first_row = df.iloc[0] if len(df) > 0 else None
        if first_row is not None:
            # Common header patterns
            header_patterns = ['date', 'country', 'language', 'state', 'question', 'time']
            matches = sum(1 for col in first_row.astype(str).str.lower() 
                         if any(pattern in col.lower() for pattern in header_patterns))
            
            if matches >= 2:  # If 2+ columns match header patterns
                structure["has_headers"] = True
                df.columns = first_row.astype(str)
                df = df.drop(df.index[0]).reset_index(drop=True)
                structure["total_rows"] = len(df)  # Update count after header removal
        
        # Find question column
        for i, col in enumerate(df.columns):
            col_name = str(col).lower()
            sample_data = df[col].astype(str).str[:100]  # First 100 chars of each cell
            
            # Count kwargs-like content with error handling
            try:
                kwargs_count = sample_data.str.contains(r'[{\[].*content.*[}\]]', regex=True, na=False).sum()
            except Exception as e:
                logger.warning(f"Error counting kwargs in column {i}: {e}")
                kwargs_count = 0
            
            structure["kwargs_rows"] += kwargs_count
            
            # Look for question-like content
            question_indicators = [
                'question', 'what', 'how', 'why', 'when', 'where', 
                'can', 'should', 'would', 'could', r'\?'  # Escape the question mark
            ]
            
            question_score = 0
            try:
                for indicator in question_indicators:
                    # Use regex=False for literal string matching except for question mark
                    is_regex = indicator.startswith('r')
                    if is_regex:
                        indicator = indicator[1:]  # Remove 'r' prefix
                    question_score += sample_data.str.contains(
                        indicator, case=False, na=False, regex=is_regex
                    ).sum()
            except Exception as e:
                logger.warning(f"Error scoring questions in column {i}: {e}")
                question_score = 0
            
            if (col_name in ['question', 'questions'] or 
                question_score > len(df) * 0.1 or  # 10% contain question words
                kwargs_count > 0):  # Contains kwargs
                structure["question_column"] = i
                structure["columns"][f"col_{i}"] = {
                    "name": col,
                    "type": "question",
                    "kwargs_count": kwargs_count,
                    "question_score": question_score
                }
                break
        
        return structure, df
    
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
            structure, df = self.detect_csv_structure(df)
            
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 30, "Analyzing data structure...")
            
            if structure["question_column"] is None:
                raise ValueError("Could not identify question column in CSV")
            
            # Extract and clean questions
            question_col_idx = structure["question_column"]
            questions = []
            kwargs_processed = 0
            cleaning_errors = []
            
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 40, "Extracting questions...")
            
            for idx, row in df.iterrows():
                try:
                    raw_question = str(row.iloc[question_col_idx])
                    
                    # Check if this looks like kwargs
                    if ('{' in raw_question and '"content"' in raw_question) or \
                       ('[' in raw_question and '"content"' in raw_question):
                        # Extract from kwargs
                        extracted = self.extract_question_from_kwargs(raw_question)
                        if extracted:
                            clean_question = self.clean_question_text(extracted)
                            if clean_question:
                                questions.append({
                                    "original_row": idx,
                                    "raw_content": raw_question[:200] + "..." if len(raw_question) > 200 else raw_question,
                                    "extracted_question": clean_question,
                                    "source": "kwargs",
                                    "metadata": {
                                        "date": row.iloc[0] if len(row) > 0 else None,
                                        "country": row.iloc[1] if len(row) > 1 else None,
                                        "language": row.iloc[2] if len(row) > 2 else None,
                                        "state": row.iloc[3] if len(row) > 3 else None,
                                    }
                                })
                                kwargs_processed += 1
                    else:
                        # Regular question
                        clean_question = self.clean_question_text(raw_question)
                        if clean_question:
                            questions.append({
                                "original_row": idx,
                                "raw_content": raw_question,
                                "extracted_question": clean_question,
                                "source": "direct",
                                "metadata": {
                                    "date": row.iloc[0] if len(row) > 0 else None,
                                    "country": row.iloc[1] if len(row) > 1 else None,
                                    "language": row.iloc[2] if len(row) > 2 else None,
                                    "state": row.iloc[3] if len(row) > 3 else None,
                                }
                            })
                
                except Exception as e:
                    cleaning_errors.append(f"Row {idx}: {str(e)}")
                    continue
                
                # Progress update every 100 rows
                if idx % 100 == 0 and progress_callback:
                    progress = 40 + int((idx / len(df)) * 30)  # 40-70% for processing
                    await progress_callback(processing_id, "cleaning", progress, f"Processed {idx}/{len(df)} rows")
            
            if progress_callback:
                await progress_callback(processing_id, "cleaning", 70, f"Extracted {len(questions)} valid questions")
            
            # Remove duplicates based on timestamp + question
            if progress_callback:
                await progress_callback(processing_id, "deduplication", 75, "Removing duplicates...")
            
            original_count = len(questions)
            questions = self.remove_duplicates(questions)
            duplicates_removed = original_count - len(questions)
            
            if progress_callback:
                await progress_callback(processing_id, "deduplication", 78, f"Removed {duplicates_removed} duplicates, {len(questions)} unique questions remain")
            
            # Remove duplicates based on timestamp + question
            if progress_callback:
                await progress_callback(processing_id, "deduplication", 75, "Removing duplicates...")
            
            original_count = len(questions)
            questions = self.remove_duplicates(questions)
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
                database_result = await questions_service.create_questions_with_deduplication(
                    questions=questions,
                    processing_id=processing_id
                )
                
                if progress_callback:
                    await progress_callback(processing_id, "database_write", 90, f"Successfully wrote {database_result['rows_written']} questions to database")
                
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
                "structure_analysis": self._convert_numpy_types(structure),
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
        import numpy as np
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

# Global service instance
data_processing_service = DataProcessingService()