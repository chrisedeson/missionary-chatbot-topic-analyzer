"""
Questions Database Service

Handles CRUD operations for questions with automatic deduplication.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from prisma import Prisma
from prisma.models import Question

from app.core.database import get_db

logger = logging.getLogger(__name__)

class QuestionsService:
    """Service for managing questions in the database with deduplication"""
    
    def __init__(self, db: Prisma):
        self.db = db
    
    async def create_questions_with_deduplication(
        self, 
        questions: List[Dict], 
        processing_id: str
    ) -> Dict:
        """
        Insert questions into database with automatic deduplication.
        
        Args:
            questions: List of question dictionaries with metadata
            processing_id: Processing ID for tracking
        
        Returns:
            Dict with insertion results and statistics
        """
        try:
            total_processed = len(questions)
            new_questions = []
            duplicates_skipped = 0
            data_errors_skipped = 0  # Track data validation errors
            
            logger.info(f"Processing {total_processed} questions for deduplication")
            
            # Process each question
            for i, question_data in enumerate(questions):
                try:
                    question_text = question_data.get('extracted_question', '').strip()
                    metadata = question_data.get('metadata', {})
                    
                    # Skip if no question text
                    if not question_text:
                        logger.warning(f"Skipping empty question at index {i}")
                        data_errors_skipped += 1
                        continue
                    
                    # Parse metadata with error handling
                    date_obj = None
                    date_str = metadata.get('date', '')
                    if date_str and date_str.strip():
                        try:
                            # Try to parse the date string
                            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError) as e:
                            logger.warning(f"Could not parse date '{date_str}' for question {i}: {e}")
                            # Continue processing without date instead of skipping entire row
                            date_obj = None
                    
                    # Check for duplicate (same question text)
                    existing_question = await self.db.question.find_first(
                        where={'text': question_text}
                    )
                    
                    if existing_question:
                        duplicates_skipped += 1
                        logger.debug(f"Duplicate question found: {question_text[:50]}...")
                        continue
                    
                    # Create new question record with safe defaults
                    question_record = {
                        'text': question_text,
                        'date': date_obj,  # Can be None
                        'country': metadata.get('country') or None,  # Convert empty strings to None
                        'state': metadata.get('state') or None,
                        'userLanguage': metadata.get('language') or None,
                        # embedding will be populated later during analysis
                        'embedding': [],
                    }
                    
                    new_questions.append(question_record)
                    logger.debug(f"New question added: {question_text[:50]}...")
                    
                except Exception as e:
                    # Handle any other data processing errors for individual questions
                    logger.error(f"Error processing question at index {i}: {e}")
                    data_errors_skipped += 1
                    continue
            
            # Batch insert new questions
            inserted_questions = []
            insertion_errors = 0
            if new_questions:
                logger.info(f"Inserting {len(new_questions)} new questions into database")
                
                # Insert questions one by one to handle any individual errors
                for question_data in new_questions:
                    try:
                        question = await self.db.question.create(data=question_data)
                        inserted_questions.append(question)
                    except Exception as e:
                        logger.error(f"Failed to insert question: {e}")
                        insertion_errors += 1
            
            rows_written = len(inserted_questions)
            
            logger.info(f"Successfully inserted {rows_written} questions, skipped {duplicates_skipped} duplicates, {data_errors_skipped} data errors, {insertion_errors} insertion errors")
            
            return {
                "status": "success",
                "rows_written": rows_written,
                "duplicates_skipped": duplicates_skipped,
                "data_errors_skipped": data_errors_skipped,
                "insertion_errors": insertion_errors,
                "total_processed": total_processed,
                "processing_id": processing_id,
                "inserted_questions": inserted_questions
            }
            
        except Exception as e:
            logger.error(f"Error inserting questions: {e}")
            raise Exception(f"Failed to insert questions into database: {str(e)}")
    
    async def get_all_questions(self, limit: Optional[int] = None) -> List[Question]:
        """Get all questions from database"""
        try:
            if limit:
                questions = await self.db.question.find_many(
                    take=limit,
                    order={'createdAt': 'desc'}
                )
            else:
                questions = await self.db.question.find_many(
                    order={'createdAt': 'desc'}
                )
            return questions
        except Exception as e:
            logger.error(f"Error fetching questions: {e}")
            raise
    
    async def get_questions_count(self) -> int:
        """Get total count of questions in database"""
        try:
            count = await self.db.question.count()
            return count
        except Exception as e:
            logger.error(f"Error counting questions: {e}")
            return 0
    
    async def get_questions_for_analysis(self, limit: Optional[int] = None) -> List[Question]:
        """Get questions that need analysis (no embedding yet)"""
        try:
            if limit:
                questions = await self.db.question.find_many(
                    where={
                        'embedding': {'equals': []}  # Empty embedding array
                    },
                    take=limit,
                    order={'createdAt': 'asc'}
                )
            else:
                questions = await self.db.question.find_many(
                    where={
                        'embedding': {'equals': []}  # Empty embedding array
                    },
                    order={'createdAt': 'asc'}
                )
            return questions
        except Exception as e:
            logger.error(f"Error fetching questions for analysis: {e}")
            raise
    
    async def sync_to_google_sheets(self) -> Dict:
        """
        Sync database questions to Google Sheets for reporting.
        Clears the sheet and writes all questions fresh.
        """
        try:
            # Get all questions from database
            questions = await self.get_all_questions()
            
            # Convert to simple Google Sheets format (no deduplication complexity)
            sheets_data = []
            for question in questions:
                sheets_data.append({
                    'extracted_question': question.text,
                    'date': question.date.isoformat() if question.date else '',
                    'country': question.country or '',
                    'state': question.state or '',
                    'language': question.userLanguage or ''
                })
            
            # Import here to avoid circular imports
            from app.services.google_sheets import GoogleSheetsService
            from app.core.config import settings
            
            sheets_service = GoogleSheetsService()
            
            # Clear the sheet completely and write all data fresh
            result = sheets_service.clear_and_write_questions(
                questions=sheets_data,
                sheet_id=settings.QUESTIONS_SHEET_ID
            )
            
            return {
                "status": "success",
                "questions_synced": len(sheets_data)
            }
            
        except Exception as e:
            logger.error(f"Error syncing to Google Sheets: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _write_all_to_sheets(self, sheets_service, questions_data):
        """Helper method to write all questions to Google Sheets"""
        # This is a simplified version - in production you might want to:
        # 1. Clear the sheet first
        # 2. Write in batches
        # 3. Handle partial failures
        
        from app.core.config import settings
        
        try:
            result = sheets_service.write_questions_to_sheet(
                questions=questions_data,
                sheet_id=settings.QUESTIONS_SHEET_ID
            )
            return result
        except Exception as e:
            logger.error(f"Failed to write to Google Sheets: {e}")
            raise


# Global service instance
questions_service = None


async def init_questions_service():
    """Initialize the global questions service"""
    global questions_service
    try:
        db = await get_db()
        if db:
            questions_service = QuestionsService(db)
            logger.info("Questions service initialized successfully")
        else:
            logger.warning("Database not available, questions service not initialized")
    except Exception as e:
        logger.error(f"Failed to initialize questions service: {e}")


async def get_questions_service() -> QuestionsService:
    """Dependency to get questions service"""
    global questions_service
    if questions_service is None:
        db = await get_db()
        questions_service = QuestionsService(db)
    return questions_service