# backend/app/services/data_processing/csv_analyzer.py

"""CSV analysis and structure detection"""

import pandas as pd
import logging
from typing import Dict, Any, Tuple

from .types import CSVStructure

logger = logging.getLogger(__name__)

class CSVAnalyzer:
    """Handles CSV file analysis and structure detection"""
    
    @staticmethod
    def detect_csv_structure(df: pd.DataFrame) -> Tuple[CSVStructure, pd.DataFrame]:
        """
        Analyze CSV structure and detect columns.
        Handle cases with/without headers, different column orders.
        
        Returns:
            Tuple of (CSVStructure, processed_dataframe)
        """
        structure_dict = {
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
                structure_dict["has_headers"] = True
                df.columns = first_row.astype(str)
                df = df.drop(df.index[0]).reset_index(drop=True)
                structure_dict["total_rows"] = len(df)  # Update count after header removal
        
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
            
            structure_dict["kwargs_rows"] += kwargs_count
            
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
                structure_dict["question_column"] = i
                structure_dict["columns"][f"col_{i}"] = {
                    "name": col,
                    "type": "question",
                    "kwargs_count": kwargs_count,
                    "question_score": question_score
                }
                break
        
        structure = CSVStructure(
            has_headers=structure_dict["has_headers"],
            columns=structure_dict["columns"],
            total_rows=structure_dict["total_rows"],
            question_column=structure_dict["question_column"],
            kwargs_rows=structure_dict["kwargs_rows"]
        )
        
        return structure, df