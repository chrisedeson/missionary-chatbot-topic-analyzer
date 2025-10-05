"""Question extraction and deduplication"""

import logging
from typing import List, Dict, Set

from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

class QuestionExtractor:
    """Handles question extraction from CSV rows and deduplication"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
    
    def extract_questions_from_dataframe(self, df, question_col_idx: int) -> tuple[List[Dict], int, List[str]]:
        """
        Extract and clean questions from DataFrame.
        
        Returns:
            Tuple of (questions_list, kwargs_processed_count, cleaning_errors)
        """
        questions = []
        kwargs_processed = 0
        cleaning_errors = []
        
        for idx, row in df.iterrows():
            try:
                raw_question = str(row.iloc[question_col_idx])
                
                # Check if this looks like kwargs
                if ('{' in raw_question and '"content"' in raw_question) or \
                   ('[' in raw_question and '"content"' in raw_question):
                    # Extract from kwargs
                    extracted = self.text_cleaner.extract_question_from_kwargs(raw_question)
                    if extracted:
                        clean_question = self.text_cleaner.clean_question_text(extracted)
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
                            logger.info(f"Skipping empty question at index {idx}")
                else:
                    # Regular question
                    clean_question = self.text_cleaner.clean_question_text(raw_question)
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
                    else:
                        logger.info(f"Skipping empty question at index {idx}")
            
            except Exception as e:
                cleaning_errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return questions, kwargs_processed, cleaning_errors
    
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
        seen: Set[tuple] = set()
        unique_questions = []
        duplicates_removed = 0
        
        for question in questions:
            # Create a key from timestamp and cleaned question text
            timestamp = question.get('metadata', {}).get('date', '')
            question_text = question.get('extracted_question', '')
            
            # Clean the question text for comparison (remove ACM prefix, normalize whitespace)
            clean_text = self.text_cleaner.clean_acm_prefix(question_text).strip().lower()
            
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