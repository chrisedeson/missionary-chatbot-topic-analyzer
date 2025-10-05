"""
Data Processing Module - Modular Structure

This module provides a complete data processing pipeline for CSV files containing questions.
It has been refactored from a monolithic service into focused, testable components.

Components:
- types.py: Type definitions and data classes
- text_cleaner.py: Text cleaning and question extraction utilities  
- csv_analyzer.py: CSV structure detection and analysis
- question_extractor.py: Question extraction and deduplication logic
- processor.py: Main orchestrator that coordinates all components

Legacy Compatibility:
The module maintains backward compatibility by re-exporting the main classes
and providing aliases for existing code that imports from the old structure.
"""

# Import all components
from .types import ProcessingResult, CSVStructure, ColumnValidation
from .text_cleaner import TextCleaner
from .csv_analyzer import CSVAnalyzer
from .question_extractor import QuestionExtractor
from .processor import DataProcessor, DataProcessingService

# Global service instance for backward compatibility
data_processing_service = DataProcessor()

# Export public API
__all__ = [
    # Main classes
    'DataProcessor',
    'DataProcessingService',  # Legacy alias
    
    # Type definitions
    'ProcessingResult',
    'CSVStructure', 
    'ColumnValidation',
    
    # Component classes
    'TextCleaner',
    'CSVAnalyzer',
    'QuestionExtractor',
    
    # Global instance
    'data_processing_service'
]