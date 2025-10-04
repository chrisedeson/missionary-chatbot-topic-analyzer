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

logger = logging.getLogger(__name__)

class DataProcessingService:
    """Service for processing uploaded questions files"""
    
    def __init__(self):
        self.active_processes: Dict[str, Dict] = {}
    
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
            return None
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common noise
        text = re.sub(r'^(Question:\s*|Q:\s*|\d+\.\s*)', '', text, flags=re.IGNORECASE)
        
        # Must be at least 3 characters and not just numbers/symbols
        if len(text) < 3 or re.match(r'^[\d\s\-_.,;:!?]*$', text):
            return None
        
        return text
    
    def detect_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze CSV structure and detect columns.
        Handle cases with/without headers, different column orders.
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
        
        # Find question column
        for i, col in enumerate(df.columns):
            col_name = str(col).lower()
            sample_data = df[col].astype(str).str[:100]  # First 100 chars of each cell
            
            # Count kwargs-like content
            kwargs_count = sample_data.str.contains(r'[{\[].*content.*[}\]]', regex=True, na=False).sum()
            structure["kwargs_rows"] += kwargs_count
            
            # Look for question-like content
            question_indicators = [
                'question', 'what', 'how', 'why', 'when', 'where', 
                'can', 'should', 'would', 'could', '?'
            ]
            
            question_score = 0
            for indicator in question_indicators:
                question_score += sample_data.str.contains(indicator, case=False, na=False).sum()
            
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
        
        return structure
    
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
            structure = self.detect_csv_structure(df)
            
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
            
            # TODO: Update Google Sheets
            if progress_callback:
                await progress_callback(processing_id, "sheets_update", 80, "Updating Google Sheets...")
            
            # Simulate sheets update for now
            await asyncio.sleep(2)
            
            if progress_callback:
                await progress_callback(processing_id, "completion", 100, "Processing completed successfully!")
            
            # Store results
            results = {
                "processing_id": processing_id,
                "filename": filename,
                "status": "completed",
                "statistics": {
                    "total_rows_processed": len(df),
                    "valid_questions_extracted": len(questions),
                    "kwargs_rows_processed": kwargs_processed,
                    "cleaning_errors": len(cleaning_errors),
                    "success_rate": len(questions) / len(df) if len(df) > 0 else 0
                },
                "structure_analysis": structure,
                "sample_questions": questions[:5],  # First 5 for preview
                "errors": cleaning_errors[:10],  # First 10 errors
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

# Global service instance
data_processing_service = DataProcessingService()