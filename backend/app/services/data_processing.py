"""
Data Processing Service - Legacy Export

This file has been refactored into modular components.
For backward compatibility, we re-export from the new modular structure.

New structure:
- data_processing/types.py: Type definitions
- data_processing/text_cleaner.py: Text cleaning utilities  
- data_processing/csv_analyzer.py: CSV analysis and validation
- data_processing/question_extractor.py: Question extraction
- data_processing/processor.py: Main processing orchestrator
- data_processing/__init__.py: Module exports
"""

# Re-export from new modular structure for backward compatibility
from .data_processing import (
    DataProcessor,
    ProcessingResult,
    CSVStructure, 
    ColumnValidation,
    data_processing_service
)

# Legacy class alias
DataProcessingService = DataProcessor

__all__ = [
    'DataProcessingService',  # Legacy alias
    'DataProcessor',
    'ProcessingResult',
    'CSVStructure',
    'ColumnValidation', 
    'data_processing_service'
]