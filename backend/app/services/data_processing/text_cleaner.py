"""Text cleaning utilities for question processing"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TextCleaner:
    """Handles text cleaning and question extraction"""
    
    @staticmethod
    def clean_acm_prefix(question: str) -> str:
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
    
    @staticmethod
    def extract_question_from_kwargs(kwargs_content: str) -> Optional[str]:
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
    
    @classmethod
    def clean_question_text(cls, text: str) -> Optional[str]:
        """Clean and validate question text"""
        if not text or not isinstance(text, str):
            logger.debug(f"Skipping non-string or empty text: {repr(text)}")
            return None
        
        # First, remove ACM prefix if present
        text = cls.clean_acm_prefix(text)
        
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