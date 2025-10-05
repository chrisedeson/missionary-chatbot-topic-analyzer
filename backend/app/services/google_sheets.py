"""
Google Sheets service for BYU Pathway Missionary Chatbot Topic Analyzer
Handles reading from and writing to Google Sheets with column validation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google.auth.exceptions import RefreshError
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
import time
import re
from difflib import get_close_matches

from app.core.config import settings

logger = logging.getLogger(__name__)

class ColumnMismatchError(Exception):
    """Raised when Google Sheets columns don't match expected format"""
    def __init__(self, message: str, expected_columns: List[str], found_columns: List[str], suggestions: Dict[str, str] = None):
        self.message = message
        self.expected_columns = expected_columns
        self.found_columns = found_columns
        self.suggestions = suggestions or {}
        super().__init__(self.message)

class GoogleSheetsService:
    """Service for Google Sheets integration with robust column validation"""
    
    def __init__(self):
        self.client = None
        self.last_error = None
        self._initialize_client()
    
    def _get_credentials_dict(self) -> Dict[str, Any]:
        """Build credentials dictionary from environment variables"""
        return {
            "type": settings.GOOGLE_SERVICE_ACCOUNT_TYPE,
            "project_id": settings.GOOGLE_SERVICE_ACCOUNT_PROJECT_ID,
            "private_key_id": settings.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID,
            "private_key": settings.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY.replace('\\n', '\n'),
            "client_email": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL,
            "client_id": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_ID,
            "auth_uri": settings.GOOGLE_SERVICE_ACCOUNT_AUTH_URI,
            "token_uri": settings.GOOGLE_SERVICE_ACCOUNT_TOKEN_URI,
            "auth_provider_x509_cert_url": settings.GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_CERT_URL,
            "client_x509_cert_url": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL,
            "universe_domain": settings.GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN
        }
    
    def _initialize_client(self) -> bool:
        """Initialize Google Sheets client"""
        try:
            credentials_dict = self._get_credentials_dict()
            
            # Create credentials from dictionary
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
            
            self.client = gspread.authorize(credentials)
            logger.info("Google Sheets client initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = f"Failed to initialize Google Sheets client: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def normalize_column_name(self, column_name: str) -> str:
        """
        Normalize column names to handle variations in case, spacing, and underscores
        
        Examples:
            'Time Stamp' -> 'timestamp'
            'time_stamp' -> 'timestamp'
            'TimeStamp' -> 'timestamp'
            'user language' -> 'userlanguage'
        """
        if not isinstance(column_name, str):
            return str(column_name).lower()
        
        # Convert to lowercase, remove spaces and underscores
        normalized = re.sub(r'[_\s\-]+', '', column_name.lower().strip())
        return normalized
    
    def validate_columns(self, sheet_columns: List[str], expected_columns: List[str]) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Validate that sheet columns match expected format with flexible matching
        
        Args:
            sheet_columns: Column names from the Google Sheet
            expected_columns: Expected column names ['Time Stamp', 'Country', 'User Language', 'State', 'Question']
        
        Returns:
            Tuple of (is_valid, column_mapping, missing_columns)
        """
        # Normalize expected columns for comparison
        expected_normalized = {self.normalize_column_name(col): col for col in expected_columns}
        
        # Create mapping from sheet columns to expected columns
        column_mapping = {}
        found_normalized = set()
        
        for sheet_col in sheet_columns:
            normalized_sheet = self.normalize_column_name(sheet_col)
            found_normalized.add(normalized_sheet)
            
            if normalized_sheet in expected_normalized:
                column_mapping[sheet_col] = expected_normalized[normalized_sheet]
        
        # Find missing columns
        missing_normalized = set(expected_normalized.keys()) - found_normalized
        missing_columns = [expected_normalized[norm] for norm in missing_normalized]
        
        is_valid = len(missing_columns) == 0
        
        return is_valid, column_mapping, missing_columns
    
    def suggest_column_fixes(self, sheet_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        """
        Suggest possible column name fixes using fuzzy matching
        
        Returns:
            Dict mapping sheet column names to suggested expected column names
        """
        suggestions = {}
        expected_normalized = {self.normalize_column_name(col): col for col in expected_columns}
        
        for sheet_col in sheet_columns:
            normalized_sheet = self.normalize_column_name(sheet_col)
            
            if normalized_sheet not in expected_normalized:
                # Find close matches
                close_matches = get_close_matches(
                    normalized_sheet, 
                    list(expected_normalized.keys()), 
                    n=1, 
                    cutoff=0.6
                )
                
                if close_matches:
                    suggested_expected = expected_normalized[close_matches[0]]
                    suggestions[sheet_col] = suggested_expected
        
        return suggestions
    
    def get_sheet_structure(self, sheet_id: str, worksheet_name: str = None) -> Dict[str, Any]:
        """
        Get the structure of a Google Sheet including columns and basic info
        
        Returns:
            Dict with sheet structure information
        """
        if not self.client:
            raise Exception("Google Sheets client not initialized")
        
        try:
            spreadsheet = self.client.open_by_key(sheet_id)
            
            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1
            
            # Get all values to analyze structure
            all_values = worksheet.get_all_values()
            
            if not all_values:
                return {
                    "columns": [],
                    "total_rows": 0,
                    "has_data": False,
                    "worksheet_name": worksheet.title
                }
            
            # First row is assumed to be headers
            columns = all_values[0] if all_values else []
            
            return {
                "columns": columns,
                "total_rows": len(all_values),
                "has_data": len(all_values) > 1,
                "worksheet_name": worksheet.title,
                "sample_data": all_values[1:6] if len(all_values) > 1 else []  # First 5 data rows
            }
            
        except Exception as e:
            logger.error(f"Error getting sheet structure: {e}")
            raise
    
    def write_questions_to_sheet(self, questions: List[Dict], sheet_id: str, worksheet_name: str = None) -> Dict[str, Any]:
        """
        Write processed questions to Google Sheets with column validation and deduplication
        
        Args:
            questions: List of question dictionaries with metadata
            sheet_id: Google Sheets ID
            worksheet_name: Optional worksheet name (defaults to first sheet)
        
        Returns:
            Dict with write results and statistics including duplicates info
        """
        if not self.client:
            raise Exception("Google Sheets client not initialized")
        
        expected_columns = ['Time Stamp', 'Country', 'User Language', 'State', 'Question']
        
        try:
            # Get sheet structure first
            structure = self.get_sheet_structure(sheet_id, worksheet_name)
            sheet_columns = structure['columns']
            
            # Validate columns
            is_valid, column_mapping, missing_columns = self.validate_columns(sheet_columns, expected_columns)
            
            if not is_valid:
                suggestions = self.suggest_column_fixes(sheet_columns, expected_columns)
                raise ColumnMismatchError(
                    f"Google Sheets columns don't match expected format. Missing: {missing_columns}",
                    expected_columns,
                    sheet_columns,
                    suggestions
                )
            
            # Open the worksheet
            spreadsheet = self.client.open_by_key(sheet_id)
            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1
            
            # Read existing data for deduplication
            existing_data = []
            try:
                existing_rows = worksheet.get_all_records()
                logger.info(f"Successfully read {len(existing_rows)} existing rows from Google Sheets")
                
                # Print first few rows for debugging
                if existing_rows:
                    logger.info(f"Sample existing row: {existing_rows[0]}")
                    logger.info(f"Available columns in existing data: {list(existing_rows[0].keys()) if existing_rows else 'No rows'}")
                else:
                    logger.warning("No existing rows found in Google Sheets")
            except Exception as e:
                logger.error(f"Error reading existing data from Google Sheets: {e}")
                existing_rows = []
            
            logger.info(f"Found {len(existing_rows)} existing rows in Google Sheets for deduplication check")
            
            # Create set of existing question signatures for fast lookup
            existing_signatures = set()
            existing_questions = set()  # Also track just question text for additional safety
            
            for i, row in enumerate(existing_rows):
                # Create a signature based on question text and timestamp
                question_text = str(row.get('Question', '')).strip().lower()
                timestamp = str(row.get('Time Stamp', '')).strip()
                
                logger.debug(f"Row {i}: Question='{question_text[:30]}...', Timestamp='{timestamp}'")
                
                # Add question text alone for broad duplicate detection
                if question_text:
                    existing_questions.add(question_text)
                    logger.debug(f"Added to existing_questions: '{question_text[:30]}...'")
                
                # Add question + timestamp signature for exact duplicate detection
                if question_text and timestamp:
                    signature = f"{question_text}|{timestamp}"
                    existing_signatures.add(signature)
                    logger.debug(f"Added signature: {signature[:50]}...")
            
            logger.info(f"Created {len(existing_signatures)} signatures and {len(existing_questions)} unique questions")
            logger.info(f"First 3 existing questions: {list(existing_questions)[:3]}")
            
            # Prepare data for writing with deduplication
            rows_to_append = []
            duplicate_count = 0
            exact_duplicate_count = 0
            question_duplicate_count = 0
            
            for i, question in enumerate(questions):
                metadata = question.get('metadata', {})
                question_text = question.get('extracted_question', '').strip()
                timestamp = str(metadata.get('date', '')).strip()
                
                logger.info(f"Processing question {i+1}/{len(questions)}: '{question_text[:50]}...' with timestamp: '{timestamp}'")
                
                # Check for question duplicates (question text only)
                question_only = question_text.lower() if question_text else ""
                
                is_duplicate = False
                
                # For now, let's be more aggressive and only check question text
                # This will prevent any duplicate questions regardless of timestamp
                if question_only and question_only in existing_questions:
                    duplicate_count += 1
                    is_duplicate = True
                    logger.info(f"DUPLICATE FOUND: '{question_only[:50]}...' already exists in sheet")
                else:
                    logger.info(f"NEW QUESTION: '{question_only[:50]}...' not found in existing {len(existing_questions)} questions")
                
                # TODO: Re-enable timestamp checking later
                # Priority 1: Check exact signature match (question + timestamp)
                # if exact_signature and exact_signature in existing_signatures:
                #     exact_duplicate_count += 1
                #     is_duplicate = True
                #     logger.debug(f"Exact duplicate found: {exact_signature[:50]}...")
                
                # Priority 2: Check question-only match (for safety, even if timestamp differs)
                # elif question_only and question_only in existing_questions:
                #     question_duplicate_count += 1
                #     is_duplicate = True
                #     logger.debug(f"Question duplicate found: {question_only[:50]}...")
                
                if is_duplicate:
                    continue
                
                row_data = []
                
                # Map data to columns in the order they appear in the sheet
                for sheet_col in sheet_columns:
                    expected_col = column_mapping.get(sheet_col)
                    
                    if expected_col == 'Time Stamp':
                        row_data.append(timestamp)
                    elif expected_col == 'Country':
                        row_data.append(str(metadata.get('country', '')))
                    elif expected_col == 'User Language':
                        row_data.append(str(metadata.get('language', '')))
                    elif expected_col == 'State':
                        row_data.append(str(metadata.get('state', '')))
                    elif expected_col == 'Question':
                        row_data.append(question_text)
                    else:
                        row_data.append('')  # Empty for unmapped columns
                
                rows_to_append.append(row_data)
            
            # Append only new unique data to sheet
            if rows_to_append:
                worksheet.append_rows(rows_to_append)
                logger.info(f"Successfully wrote {len(rows_to_append)} new questions to Google Sheets, skipped {duplicate_count} duplicates")
            else:
                logger.info(f"No new questions to write - all {duplicate_count} were duplicates")
            
            return {
                "status": "success",
                "rows_written": len(rows_to_append),
                "duplicates_skipped": duplicate_count,
                "total_processed": len(questions),
                "existing_rows_count": len(existing_rows),
                "sheet_id": sheet_id,
                "worksheet_name": worksheet.title,
                "column_mapping": column_mapping
            }
            
        except ColumnMismatchError:
            raise  # Re-raise column mismatch errors
        except Exception as e:
            error_message = str(e)
            
            # Provide more helpful error messages for common issues
            if "403" in error_message and "permission" in error_message.lower():
                error_message = (
                    "Google Sheets access denied. Please ensure the service account "
                    f"({settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL}) has been granted "
                    "Editor access to the Google Sheet. Share the sheet with this email address."
                )
            elif "404" in error_message:
                error_message = (
                    f"Google Sheet not found (ID: {sheet_id[:20]}...). "
                    "Please verify the QUESTIONS_SHEET_ID in your environment configuration."
                )
            elif "401" in error_message:
                error_message = (
                    "Google Sheets authentication failed. Please verify your service account "
                    "credentials in the environment configuration."
                )
            
            logger.error(f"Error writing to Google Sheets: {error_message}")
            raise Exception(f"Failed to write to Google Sheets: {error_message}")
    
    def clear_and_write_questions(self, questions: List[Dict], sheet_id: str, worksheet_name: str = None) -> Dict[str, Any]:
        """
        Clear Google Sheets completely and write all questions fresh from database.
        Simple approach without deduplication complexity.
        
        Args:
            questions: List of question dictionaries with simple structure
            sheet_id: Google Sheets ID
            worksheet_name: Optional worksheet name (defaults to first sheet)
        
        Returns:
            Dict with write results
        """
        if not self.client:
            raise Exception("Google Sheets client not initialized")
        
        try:
            # Open the spreadsheet
            spreadsheet = self.client.open_by_key(sheet_id)
            
            if worksheet_name:
                try:
                    worksheet = spreadsheet.worksheet(worksheet_name)
                except WorksheetNotFound:
                    logger.warning(f"Worksheet '{worksheet_name}' not found, using first sheet")
                    worksheet = spreadsheet.sheet1
            else:
                worksheet = spreadsheet.sheet1
            
            # Clear the entire sheet
            worksheet.clear()
            
            # Set up headers
            headers = ['Time Stamp', 'Country', 'User Language', 'State', 'Question']
            
            # Prepare data rows
            rows_to_write = [headers]  # Start with headers
            
            for question in questions:
                row_data = [
                    question.get('date', ''),  # Time Stamp
                    question.get('country', ''),  # Country
                    question.get('language', ''),  # User Language
                    question.get('state', ''),  # State
                    question.get('extracted_question', '')  # Question
                ]
                rows_to_write.append(row_data)
            
            # Write all data at once (headers + data)
            if rows_to_write:
                worksheet.update('A1', rows_to_write)
                logger.info(f"Successfully cleared and wrote {len(questions)} questions to Google Sheets")
            
            return {
                "status": "success",
                "rows_written": len(questions),
                "sheet_cleared": True,
                "sheet_id": sheet_id,
                "worksheet_name": worksheet.title
            }
            
        except Exception as e:
            error_message = str(e)
            
            # Provide helpful error messages
            if "403" in error_message and "permission" in error_message.lower():
                error_message = (
                    "Google Sheets access denied. Please ensure the service account "
                    f"({settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL}) has been granted "
                    "Editor access to the Google Sheet."
                )
            elif "404" in error_message:
                error_message = f"Google Sheet not found (ID: {sheet_id[:20]}...)"
            elif "401" in error_message:
                error_message = "Google Sheets authentication failed."
            
            logger.error(f"Error clearing and writing to Google Sheets: {error_message}")
            raise Exception(f"Failed to clear and write to Google Sheets: {error_message}")

    def read_questions_from_sheet(self, sheet_id: str, worksheet_name: str = None) -> pd.DataFrame:
        """
        Read questions from Google Sheets and return as DataFrame
        
        Returns:
            DataFrame with questions data
        """
        if not self.client:
            raise Exception("Google Sheets client not initialized")
        
        try:
            spreadsheet = self.client.open_by_key(sheet_id)
            
            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1
            
            # Get all records as list of dictionaries
            records = worksheet.get_all_records()
            
            if not records:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            logger.info(f"Successfully read {len(df)} rows from Google Sheets")
            return df
            
        except Exception as e:
            logger.error(f"Error reading from Google Sheets: {e}")
            raise Exception(f"Failed to read from Google Sheets: {str(e)}")
    
    def test_connection(self, sheet_id: str) -> Dict[str, Any]:
        """
        Test connection to a specific Google Sheet
        
        Returns:
            Dict with connection test results
        """
        try:
            if not self.client and not self._initialize_client():
                return {
                    "status": "error",
                    "message": "Failed to initialize Google Sheets client",
                    "accessible": False
                }
            
            # Try to access the spreadsheet
            spreadsheet = self.client.open_by_key(sheet_id)
            
            return {
                "status": "success",
                "message": f"Successfully connected to sheet: {spreadsheet.title}",
                "accessible": True,
                "sheet_title": spreadsheet.title,
                "worksheets": [ws.title for ws in spreadsheet.worksheets()]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "accessible": False
            }

# Global service instance
google_sheets_service = GoogleSheetsService()