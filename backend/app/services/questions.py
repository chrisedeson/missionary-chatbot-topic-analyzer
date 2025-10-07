"""
Questions Database Service

Handles CRUD operations for questions with automatic deduplication and
integration with the embedding cache system.
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
            data_errors_skipped = 0
            
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
                            # Handle various date formats
                            if isinstance(date_str, datetime):
                                date_obj = date_str
                            else:
                                # Try ISO format first
                                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        except (ValueError, AttributeError) as e:
                            logger.warning(f"Could not parse date '{date_str}' at index {i}: {e}")
                            date_obj = None
                    
                    # Check for duplicate (same question text)
                    existing_question = await self.db.question.find_first(
                        where={'text': question_text}
                    )
                    
                    if existing_question:
                        duplicates_skipped += 1
                        logger.debug(f"Duplicate question found: {question_text[:50]}...")
                        continue
                    
                    # Prepare question record with safe defaults
                    question_record = {
                        'text': question_text,
                        'date': date_obj,
                        'country': metadata.get('country') or None,
                        'state': metadata.get('state') or None,
                        'userLanguage': metadata.get('language') or None,
                        # Start with empty embedding - will be populated during analysis
                        'embedding': [],
                    }
                    
                    new_questions.append(question_record)
                    
                except Exception as e:
                    logger.error(f"Error processing question at index {i}: {e}")
                    data_errors_skipped += 1
                    continue
            
            # Batch insert new questions
            inserted_questions = []
            insertion_errors = 0
            
            if new_questions:
                logger.info(f"Inserting {len(new_questions)} new questions into database")
                
                # Insert questions one by one for better error handling
                for question_data in new_questions:
                    try:
                        question = await self.db.question.create(data=question_data)
                        inserted_questions.append(question)
                    except Exception as e:
                        logger.error(f"Failed to insert question '{question_data.get('text', '')[:50]}...': {e}")
                        insertion_errors += 1
            
            rows_written = len(inserted_questions)
            
            logger.info(
                f"Question insertion complete: "
                f"{rows_written} new, "
                f"{duplicates_skipped} duplicates, "
                f"{data_errors_skipped} data errors, "
                f"{insertion_errors} insertion errors"
            )
            
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
            logger.error(f"Error inserting questions: {e}", exc_info=True)
            raise Exception(f"Failed to insert questions into database: {str(e)}")
    
    async def get_all_questions(self, limit: Optional[int] = None) -> List[Question]:
        """Get all questions from database"""
        try:
            query_params = {"order": {"createdAt": "desc"}}
            
            if limit:
                query_params["take"] = limit
            
            questions = await self.db.question.find_many(**query_params)
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
    
    async def get_questions_for_analysis(
        self, 
        limit: Optional[int] = None,
        include_unprocessed_only: bool = True
    ) -> List[Question]:
        """
        Get questions that need analysis.
        
        Args:
            limit: Maximum number of questions to return
            include_unprocessed_only: If True, only return questions without embeddings
            
        Returns:
            List of Question objects
        """
        try:
            query_params = {"order": {"createdAt": "asc"}}
            
            if include_unprocessed_only:
                # Only get questions that haven't been embedded yet
                query_params["where"] = {
                    "embedding": {"isEmpty": True}
                }
            
            if limit:
                query_params["take"] = limit
            
            questions = await self.db.question.find_many(**query_params)
            return questions
            
        except Exception as e:
            logger.error(f"Error fetching questions for analysis: {e}")
            raise
    
    async def update_question_embedding(
        self,
        question_id: str,
        embedding: List[float]
    ) -> Question:
        """
        Update a question's embedding vector.
        
        Args:
            question_id: Question ID
            embedding: Embedding vector
            
        Returns:
            Updated Question object
        """
        try:
            question = await self.db.question.update(
                where={"id": question_id},
                data={"embedding": embedding}
            )
            return question
        except Exception as e:
            logger.error(f"Error updating question embedding: {e}")
            raise
    
    async def bulk_update_embeddings(
        self,
        question_embeddings: Dict[str, List[float]]
    ) -> int:
        """
        Bulk update embeddings for multiple questions.
        
        Args:
            question_embeddings: Dict mapping question IDs to embeddings
            
        Returns:
            Number of questions updated
        """
        updated_count = 0
        
        try:
            for question_id, embedding in question_embeddings.items():
                try:
                    await self.db.question.update(
                        where={"id": question_id},
                        data={"embedding": embedding}
                    )
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Failed to update embedding for question {question_id}: {e}")
            
            logger.info(f"Bulk updated {updated_count} question embeddings")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error in bulk embedding update: {e}")
            raise
    
    async def get_questions_without_topics(
        self,
        limit: Optional[int] = None
    ) -> List[Question]:
        """Get questions that haven't been assigned to topics yet"""
        try:
            query_params = {
                "where": {"topicId": None},
                "order": {"createdAt": "desc"}
            }
            
            if limit:
                query_params["take"] = limit
            
            questions = await self.db.question.find_many(**query_params)
            return questions
            
        except Exception as e:
            logger.error(f"Error fetching unassigned questions: {e}")
            raise
    
    async def assign_question_to_topic(
        self,
        question_id: str,
        topic_id: str,
        similarity_score: Optional[float] = None
    ) -> Question:
        """
        Assign a question to a topic.
        
        Args:
            question_id: Question ID
            topic_id: Topic ID to assign
            similarity_score: Optional similarity score
            
        Returns:
            Updated Question object
        """
        try:
            update_data = {
                "topicId": topic_id,
                "isNewTopic": False
            }
            
            if similarity_score is not None:
                update_data["similarityScore"] = similarity_score
            
            question = await self.db.question.update(
                where={"id": question_id},
                data=update_data
            )
            return question
            
        except Exception as e:
            logger.error(f"Error assigning question to topic: {e}")
            raise
    
    async def sync_to_google_sheets(self) -> Dict:
        """
        Sync database questions to Google Sheets for reporting.
        Clears the sheet and writes all questions fresh.
        """
        try:
            # Get all questions from database
            questions = await self.get_all_questions()
            
            # Convert to Google Sheets format
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
            
            # Clear and write all data
            result = sheets_service.clear_and_write_questions(
                questions=sheets_data,
                sheet_id=settings.QUESTIONS_SHEET_ID
            )
            
            logger.info(f"Synced {len(sheets_data)} questions to Google Sheets")
            
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
    
    async def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about questions in the database.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            total_questions = await self.db.question.count()
            
            questions_with_embeddings = await self.db.question.count(
                where={"embedding": {"isEmpty": False}}
            )
            
            questions_with_topics = await self.db.question.count(
                where={"topicId": {"not": None}}
            )
            
            new_topic_questions = await self.db.question.count(
                where={"isNewTopic": True}
            )
            
            # Get unique countries
            questions = await self.db.question.find_many(
                where={"country": {"not": None}},
                select={"country": True}
            )
            unique_countries = len(set(q.country for q in questions if q.country))
            
            return {
                "total_questions": total_questions,
                "questions_with_embeddings": questions_with_embeddings,
                "questions_with_topics": questions_with_topics,
                "new_topic_questions": new_topic_questions,
                "unique_countries": unique_countries,
                "embedding_coverage": (
                    (questions_with_embeddings / total_questions * 100)
                    if total_questions > 0 else 0
                ),
                "topic_coverage": (
                    (questions_with_topics / total_questions * 100)
                    if total_questions > 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
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