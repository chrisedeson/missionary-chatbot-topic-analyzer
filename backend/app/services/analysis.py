"""
Analysis service for managing topic analysis runs and database integration.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from app.services.hybrid_analysis import HybridTopicAnalyzer
from app.core.database import get_db
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Service for managing analysis runs and database operations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.analyzer = HybridTopicAnalyzer()
        self.active_runs: Dict[str, Dict] = {}
    
    async def start_analysis(
        self, 
        mode: str = "all",
        sample_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Start a new topic analysis run.
        
        Args:
            mode: "sample" or "all"
            sample_size: Number of questions to sample (if mode is "sample")
            progress_callback: Function to call with progress updates
            
        Returns:
            run_id: Unique identifier for this analysis run
        """
        run_id = str(uuid.uuid4())
        
        logger.info(f"Starting analysis run {run_id} (mode: {mode})")
        
        # Create analysis run record
        db = next(get_db())
        
        try:
            # Create run record in database
            run_data = {
                'id': run_id,
                'status': 'running',
                'mode': mode,
                'sample_size': sample_size,
                'started_at': datetime.utcnow(),
                'settings': {
                    'similarity_threshold': self.analyzer.similarity_threshold,
                    'embedding_model': self.analyzer.embedding_model,
                    'gpt_model': self.analyzer.gpt_model,
                    'umap_n_components': self.analyzer.umap_n_components,
                    'hdbscan_min_cluster_size': self.analyzer.hdbscan_min_cluster_size,
                    'random_seed': self.analyzer.random_seed
                }
            }
            
            # Store run info in memory for progress tracking
            self.active_runs[run_id] = {
                **run_data,
                'progress': {'stage': 'initialization', 'progress': 0, 'message': 'Starting analysis...'}
            }
            
            # Start analysis in background
            asyncio.create_task(self._run_analysis(run_id, mode, sample_size, progress_callback))
            
            logger.info(f"Analysis run {run_id} started successfully")
            return run_id
            
        except Exception as e:
            logger.error(f"Error starting analysis run {run_id}: {e}")
            raise
        finally:
            db.close()
    
    async def _run_analysis(
        self, 
        run_id: str, 
        mode: str, 
        sample_size: Optional[int],
        progress_callback: Optional[Callable] = None
    ):
        """
        Execute the analysis pipeline in the background.
        """
        db = next(get_db())
        
        try:
            logger.info(f"Executing analysis pipeline for run {run_id}")
            
            # Update progress callback to store in memory
            async def update_progress(stage: str, progress: float, message: str):
                if run_id in self.active_runs:
                    self.active_runs[run_id]['progress'] = {
                        'stage': stage,
                        'progress': progress,
                        'message': message,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                
                # Call external progress callback if provided
                if progress_callback:
                    await progress_callback(stage, progress, message)
            
            # Step 1: Load questions from database
            await update_progress('loading_data', 5, 'Loading questions from database...')
            
            questions = await self._load_questions(db, mode, sample_size)
            
            if not questions:
                raise Exception("No questions found in database")
            
            logger.info(f"Loaded {len(questions)} questions for analysis")
            
            # Step 2: Load existing topics (if any)
            await update_progress('loading_topics', 10, 'Loading existing topics...')
            
            existing_topics = await self._load_existing_topics(db)
            
            logger.info(f"Loaded {len(existing_topics)} existing topics")
            
            # Step 3: Run hybrid analysis
            await update_progress('analysis_start', 15, 'Starting hybrid topic analysis...')
            
            results = await self.analyzer.run_hybrid_analysis(
                questions=questions,
                existing_topics=existing_topics,
                progress_callback=update_progress
            )
            
            # Step 4: Save results to database
            await update_progress('saving_results', 90, 'Saving results to database...')
            
            await self._save_analysis_results(db, run_id, results)
            
            # Step 5: Update run status
            await update_progress('completed', 100, 'Analysis completed successfully!')
            
            if run_id in self.active_runs:
                self.active_runs[run_id]['status'] = 'completed'
                self.active_runs[run_id]['completed_at'] = datetime.utcnow()
                self.active_runs[run_id]['results_summary'] = results['summary']
            
            logger.info(f"Analysis run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis run {run_id}: {e}")
            
            # Update run status to failed
            if run_id in self.active_runs:
                self.active_runs[run_id]['status'] = 'failed'
                self.active_runs[run_id]['error'] = str(e)
                self.active_runs[run_id]['failed_at'] = datetime.utcnow()
            
            # Update progress to show error
            if progress_callback:
                await progress_callback('error', 0, f'Analysis failed: {str(e)}')
                
        finally:
            db.close()
    
    async def _load_questions(
        self, 
        db, 
        mode: str, 
        sample_size: Optional[int]
    ) -> List[str]:
        """
        Load questions from database based on mode and sample size.
        """
        try:
            # This would be replaced with actual Prisma queries
            # For now, return mock data that matches the expected format
            
            if mode == "sample" and sample_size:
                # Load sample of questions
                logger.info(f"Loading {sample_size} sample questions")
                # Mock implementation - replace with actual database query
                return [
                    "How do I prepare for a mission?",
                    "What should I expect as a missionary?",
                    "How do I overcome homesickness?",
                    "What are the best study methods?",
                    "How do I build testimony?",
                    "What if I struggle with the language?",
                    "How do I work with companions?",
                    "What about challenging investigators?",
                    "How do I stay motivated?",
                    "What if I get discouraged?"
                ][:sample_size]
            else:
                # Load all questions
                logger.info("Loading all questions from database")
                # Mock implementation - replace with actual database query
                return [
                    "How do I prepare for a mission?",
                    "What should I expect as a missionary?",
                    "How do I overcome homesickness?",
                    "What are the best study methods?",
                    "How do I build testimony?",
                    "What if I struggle with the language?",
                    "How do I work with companions?",
                    "What about challenging investigators?",
                    "How do I stay motivated?",
                    "What if I get discouraged?",
                    "How do I handle rejection?",
                    "What about difficult areas?",
                    "How do I help inactive members?",
                    "What if I'm not seeing success?",
                    "How do I improve my teaching?",
                    "What about safety concerns?",
                    "How do I handle stress?",
                    "What if I'm struggling with rules?",
                    "How do I develop charity?",
                    "What about returning home?"
                ]
                
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise
    
    async def _load_existing_topics(self, db) -> List[Dict]:
        """
        Load existing topics and their representative embeddings.
        """
        try:
            # This would be replaced with actual Prisma queries
            # For now, return empty list since we're starting fresh
            logger.info("Loading existing topics from database")
            
            # Mock implementation - in production this would query the topics table
            # and return topics with their representative embeddings
            existing_topics = []
            
            logger.info(f"Loaded {len(existing_topics)} existing topics")
            return existing_topics
            
        except Exception as e:
            logger.error(f"Error loading existing topics: {e}")
            raise
    
    async def _save_analysis_results(
        self, 
        db, 
        run_id: str, 
        results: Dict[str, Any]
    ):
        """
        Save analysis results to database.
        """
        try:
            logger.info(f"Saving analysis results for run {run_id}")
            
            # Save new topics
            for topic_data in results['new_topics']:
                topic_id = str(uuid.uuid4())
                
                # Calculate representative embedding (centroid of cluster questions)
                cluster_questions = [
                    q for q in results['clustered_questions'] 
                    if q['cluster_id'] == topic_data['cluster_id'] and not q['is_noise']
                ]
                
                if cluster_questions:
                    # Calculate centroid embedding
                    embeddings = [q['embedding'] for q in cluster_questions]
                    representative_embedding = list(np.mean(embeddings, axis=0))
                else:
                    representative_embedding = [0.0] * 1536  # Default embedding
                
                # This would be replaced with actual Prisma topic creation
                logger.info(f"Would save topic: {topic_data['name']} with {topic_data['question_count']} questions")
            
            # Save question classifications
            for similar_q in results['similar_questions']:
                # This would update questions table with topic assignments
                logger.info(f"Would classify question to existing topic: {similar_q['matched_topic_name']}")
            
            # Save clustered questions
            for cluster_q in results['clustered_questions']:
                if not cluster_q['is_noise']:
                    # This would update questions table with new topic assignments
                    logger.info(f"Would assign question to new cluster: {cluster_q['cluster_id']}")
            
            # Update analysis run with results
            # This would update the analysis_runs table with final results
            logger.info(f"Analysis results saved for run {run_id}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results for run {run_id}: {e}")
            raise
    
    def get_run_status(self, run_id: str) -> Optional[Dict]:
        """
        Get the current status of an analysis run.
        """
        return self.active_runs.get(run_id)
    
    def get_run_progress(self, run_id: str) -> Optional[Dict]:
        """
        Get the current progress of an analysis run.
        """
        run_info = self.active_runs.get(run_id)
        return run_info.get('progress') if run_info else None
    
    async def get_analysis_history(self, limit: int = 20) -> List[Dict]:
        """
        Get history of analysis runs.
        """
        try:
            # For now, return the active runs (mock implementation)
            # TODO: Replace with actual Prisma query when database is connected
            history = []
            
            for run_id, run_data in self.active_runs.items():
                history.append({
                    'id': run_id,
                    'status': run_data['status'],
                    'mode': run_data['mode'],
                    'started_at': run_data['started_at'].isoformat() if isinstance(run_data['started_at'], datetime) else run_data['started_at'],
                    'completed_at': run_data.get('completed_at', {}).isoformat() if run_data.get('completed_at') and hasattr(run_data.get('completed_at'), 'isoformat') else run_data.get('completed_at'),
                    'summary': run_data.get('results_summary', {})
                })
            
            # Sort by start time, most recent first
            history.sort(key=lambda x: x['started_at'], reverse=True)
            
            return history[:limit]
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            raise
    
    async def get_topics(self, run_id: Optional[str] = None) -> List[Dict]:
        """
        Get topics from a specific run or all topics.
        """
        db = next(get_db())
        
        try:
            # This would be replaced with actual Prisma queries
            # For now, return mock topics
            
            if run_id and run_id in self.active_runs:
                run_data = self.active_runs[run_id]
                if 'results_summary' in run_data:
                    # Return topics from this specific run
                    return []  # Would return actual topics from database
            
            # Return all topics
            logger.info("Loading all topics from database")
            
            # Mock topics data
            return [
                {
                    'id': 'topic-1',
                    'name': 'Mission Preparation',
                    'description': 'Questions about preparing for missionary service',
                    'question_count': 5,
                    'confidence_score': 0.85,
                    'keywords': ['preparation', 'mission', 'ready'],
                    'representative_questions': [
                        'How do I prepare for a mission?',
                        'What should I expect as a missionary?'
                    ]
                },
                {
                    'id': 'topic-2', 
                    'name': 'Companion Relations',
                    'description': 'Questions about working with missionary companions',
                    'question_count': 3,
                    'confidence_score': 0.78,
                    'keywords': ['companion', 'relationships', 'teamwork'],
                    'representative_questions': [
                        'How do I work with companions?',
                        'What if my companion is difficult?'
                    ]
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            raise
        finally:
            db.close()
    
    async def export_results(self, run_id: str) -> Dict[str, Any]:
        """
        Export analysis results for a specific run.
        """
        run_data = self.active_runs.get(run_id)
        
        if not run_data:
            raise ValueError(f"Analysis run {run_id} not found")
        
        if run_data['status'] != 'completed':
            raise ValueError(f"Analysis run {run_id} is not completed")
        
        # Get topics for this run
        topics = await self.get_topics(run_id)
        
        return {
            'run_id': run_id,
            'run_info': {
                'status': run_data['status'],
                'mode': run_data['mode'],
                'started_at': run_data['started_at'].isoformat() if isinstance(run_data['started_at'], datetime) else run_data['started_at'],
                'completed_at': run_data.get('completed_at', {}).isoformat() if run_data.get('completed_at') and hasattr(run_data.get('completed_at'), 'isoformat') else run_data.get('completed_at'),
                'settings': run_data['settings']
            },
            'summary': run_data.get('results_summary', {}),
            'topics': topics,
            'export_timestamp': datetime.utcnow().isoformat()
        }

# Global instance
analysis_service = AnalysisService()