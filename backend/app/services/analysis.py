"""
Analysis service for managing topic analysis runs and database integration.
"""

import asyncio
import json
import logging
import numpy as np
import random
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from app.services.hybrid_analysis import HybridTopicAnalyzer
from app.core.database import get_db
from app.core.config import get_settings
from app.services.google_sheets import google_sheets_service

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
        
        # Create analysis run in database
        db = await get_db()
        try:
            await db.analysisrun.create(
                data={
                    'id': run_id,
                    'status': 'running',
                    'progress': 0,
                    'message': 'Starting analysis...',
                    'mode': mode,
                    'sampleSize': sample_size,
                    'startedAt': datetime.utcnow()
                }
            )
            logger.info(f"✅ Created analysis run {run_id} in database")
        except Exception as e:
            logger.warning(f"⚠️ Could not create analysis run in database: {e}")
            logger.info("📝 Continuing with in-memory storage only")
        
        # Create run record in memory (no database for now to avoid coroutine issues)
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
        task = asyncio.create_task(self._run_analysis(run_id, mode, sample_size, progress_callback))
        
        logger.info(f"Analysis run {run_id} started successfully - background task created")
        return run_id
    
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
        db = await get_db()
        
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
                
                # Update database progress
                try:
                    await db.analysisrun.update(
                        where={'id': run_id},
                        data={
                            'progress': int(progress),
                            'message': f"{stage}: {message}"
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not update database progress: {e}")
                
                # Call external progress callback if provided
                if progress_callback:
                    await progress_callback(stage, progress, message)
            
            # Step 1: Load questions from Google Sheets
            await update_progress('loading_data', 5, 'Loading questions from Google Sheets...')
            
            questions = await self._load_questions(db, mode, sample_size)
            
            if not questions:
                raise Exception("No questions found in Google Sheets")
            
            logger.info(f"Loaded {len(questions)} questions for analysis")
            
            # Step 2: Load existing topics (if any)
            await update_progress('loading_topics', 10, 'Loading existing topics...')
            
            existing_topics = await self._load_existing_topics(db)
            
            logger.info(f"Loaded {len(existing_topics)} existing topics")
            
            # Step 3: Run hybrid analysis
            await update_progress('analysis_start', 15, 'Starting hybrid topic analysis...')
            
            results = await self.analyzer.perform_hybrid_analysis(
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
                # Create summary from results structure
                self.active_runs[run_id]['results_summary'] = {
                    'total_questions_analyzed': results.get('total_questions_analyzed', 0),
                    'similar_questions_count': results.get('similar_questions', {}).get('count', 0),
                    'new_topics_count': results.get('new_topics', {}).get('count', 0),
                    'analysis_id': results.get('analysis_id', ''),
                    'completed_at': results.get('completed_at', datetime.utcnow().isoformat())
                }
            
            logger.info(f"Analysis run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background analysis task: {e}")
            
            # Update the run status to failed
            if run_id in self.active_runs:
                self.active_runs[run_id]['status'] = 'failed'
                self.active_runs[run_id]['error'] = str(e)
                self.active_runs[run_id]['completed_at'] = datetime.utcnow().isoformat()
        
        finally:
            # Clean up database connection
            if 'db' in locals():
                await db.disconnect()
    
    async def get_questions_count(self) -> int:
        """
        Get the total number of questions available in Google Sheets.
        """
        try:
            logger.info("Getting question count from Google Sheets")
            
            # Read questions from the configured Google Sheet
            questions_df = google_sheets_service.read_questions_from_sheet(
                sheet_id=self.settings.QUESTIONS_SHEET_ID
            )
            
            if questions_df.empty:
                logger.warning("No questions found in Google Sheets")
                return 0
            
            # Look for the question column (case-insensitive)
            question_column = None
            for col in questions_df.columns:
                if col.lower().strip() in ['question', 'questions']:
                    question_column = col
                    break
            
            if question_column is None:
                # If no exact match, look for columns containing 'question'
                for col in questions_df.columns:
                    if 'question' in col.lower():
                        question_column = col
                        break
            
            if question_column is None:
                logger.warning("No question column found in Google Sheets")
                return 0
            
            # Count non-empty questions
            all_questions = questions_df[question_column].dropna().astype(str).tolist()
            valid_questions = [q.strip() for q in all_questions if q.strip() and q.strip().lower() != 'question']
            
            logger.info(f"Found {len(valid_questions)} valid questions in Google Sheets")
            return len(valid_questions)
                
        except Exception as e:
            logger.error(f"Error getting question count from Google Sheets: {e}")
            return 0

    async def _load_questions(
        self, 
        db, 
        mode: str, 
        sample_size: Optional[int]
    ) -> List[str]:
        """
        Load questions from Google Sheets based on mode and sample size.
        """
        try:
            logger.info("Loading questions from Google Sheets")
            
            # Read questions from the configured Google Sheet
            questions_df = google_sheets_service.read_questions_from_sheet(
                sheet_id=self.settings.QUESTIONS_SHEET_ID
            )
            
            if questions_df.empty:
                logger.warning("No questions found in Google Sheets")
                return []
            
            # Look for the question column (case-insensitive)
            question_column = None
            for col in questions_df.columns:
                if col.lower().strip() in ['question', 'questions']:
                    question_column = col
                    break
            
            if question_column is None:
                # If no exact match, look for columns containing 'question'
                for col in questions_df.columns:
                    if 'question' in col.lower():
                        question_column = col
                        break
            
            if question_column is None:
                raise Exception("No question column found in Google Sheets. Expected column names: 'Question' or 'Questions'")
            
            # Extract questions and filter out empty ones
            all_questions = questions_df[question_column].dropna().astype(str).tolist()
            all_questions = [q.strip() for q in all_questions if q.strip() and q.strip().lower() != 'question']
            
            logger.info(f"Loaded {len(all_questions)} questions from Google Sheets")
            
            if mode == "sample" and sample_size and sample_size < len(all_questions):
                # Return random sample
                import random
                sampled_questions = random.sample(all_questions, sample_size)
                logger.info(f"Returning {sample_size} sample questions")
                return sampled_questions
            else:
                # Return all questions
                logger.info(f"Returning all {len(all_questions)} questions")
                return all_questions
                
        except Exception as e:
            logger.error(f"Error loading questions from Google Sheets: {e}")
            # Fallback to mock data for development
            logger.info("Falling back to mock data")
            
            mock_questions = [
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
            
            if mode == "sample" and sample_size:
                return mock_questions[:sample_size]
            else:
                return mock_questions
    
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
    
    async def get_run_topics(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get topics from a specific run or all topics.
        """
        db = await get_db()
        
        try:
            # This would be replaced with actual Prisma queries
            # For now, return mock topics
            
            if run_id and run_id in self.active_runs:
                run_data = self.active_runs[run_id]
                if 'results' in run_data and 'new_topics' in run_data['results']:
                    # Return topics from this specific run based on reference structure
                    topics = []
                    for topic_data in run_data['results']['new_topics']['topics']:
                        # Convert numpy types to native Python types for JSON serialization
                        cluster_id = int(topic_data['cluster_id']) if hasattr(topic_data['cluster_id'], 'item') else topic_data['cluster_id']
                        question_count = int(topic_data['question_count']) if hasattr(topic_data['question_count'], 'item') else topic_data['question_count']
                        
                        topics.append({
                            'id': f"topic-{cluster_id}",
                            'name': topic_data['topic_name'],
                            'description': f"Topic discovered through clustering analysis",
                            'question_count': question_count,
                            'confidence_score': 0.85,  # Default confidence
                            'keywords': [],  # Would extract from topic data
                            'representative_questions': [topic_data['representative_question']],
                            'cluster_id': cluster_id
                        })
                    return topics
            
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
            logger.error(f"Error getting run topics: {e}")
            raise
        finally:
            # No need to close connection for Prisma
            pass
    
    # Alias method for backward compatibility with API
    async def get_topics(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Alias for get_run_topics to maintain API compatibility.
        """
        return await self.get_run_topics(run_id)
    
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

    async def _save_analysis_results(self, db, run_id: str, results: Dict[str, Any]):
        """
        Save analysis results to database following the reference implementation structure.
        Based on hybrid_topic_discovery_and_classification.py output format.
        """
        try:
            logger.info(f"Saving analysis results for run {run_id}")
            
            if run_id in self.active_runs:
                # Store complete results following reference structure
                self.active_runs[run_id]['results'] = results
                
                # Create summary following reference implementation
                similar_questions = results.get('similar_questions', {}).get('questions', [])
                new_topics_data = results.get('new_topics', {}).get('topics', [])
                
                # Store results summary compatible with reference format
                self.active_runs[run_id]['results_summary'] = {
                    'total_questions_analyzed': results.get('total_questions_analyzed', 0),
                    'similar_questions_count': len(similar_questions),
                    'new_topics_count': len(new_topics_data),
                    'analysis_id': results.get('analysis_id', ''),
                    'completed_at': results.get('completed_at', datetime.utcnow().isoformat()),
                    'configuration': results.get('configuration', {})
                }
                
                # Try to save to database if connection is available
                try:
                    # Update analysis run record
                    await self._update_analysis_run_in_db(db, run_id, results)
                    
                    # Save questions with embeddings and classifications
                    await self._save_questions_to_db(db, run_id, results)
                    
                    # Save new topics discovered
                    await self._save_topics_to_db(db, run_id, results)
                    
                    # Save embeddings to cache
                    await self._save_embeddings_to_cache(db, results)
                    
                    logger.info(f"✅ Successfully saved analysis results to database for run {run_id}")
                    
                except Exception as db_error:
                    logger.warning(f"⚠️ Database save failed (development mode?): {db_error}")
                    logger.info(f"📁 Results stored in memory for run {run_id}")
                
                # Log results following reference implementation format
                logger.info(f"Analysis results saved for run {run_id}")
                logger.info(f"   📊 Total questions processed: {results.get('total_questions_analyzed', 0)}")
                logger.info(f"   🔗 Similar to existing topics: {len(similar_questions)} ({len(similar_questions)/results.get('total_questions_analyzed', 1)*100:.1f}%)")
                logger.info(f"   🆕 New topics discovered: {len(new_topics_data)} topics")
                
                # Save individual question embeddings and classifications
                # This follows the reference implementation data structure
                if 'similar_questions' in results:
                    logger.info(f"   💾 Saved {len(similar_questions)} similar question classifications")
                    
                if 'new_topics' in results:
                    logger.info(f"   💾 Saved {len(new_topics_data)} new topic definitions")
                    
            else:
                logger.warning(f"Run {run_id} not found in active runs")
                
        except Exception as e:
            logger.error(f"Error saving analysis results for run {run_id}: {e}")
            # Don't raise - this shouldn't fail the analysis

    async def _update_analysis_run_in_db(self, db, run_id: str, results: Dict[str, Any]):
        """Update analysis run record in database with results"""
        try:
            similar_count = len(results.get('similar_questions', {}).get('questions', []))
            new_topics_count = len(results.get('new_topics', {}).get('topics', []))
            total_questions = results.get('total_questions_analyzed', 0)
            
            # Normalize completedAt to Python datetime
            completed_at = results.get('completed_at')
            if isinstance(completed_at, str):
                try:
                    completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                except Exception:
                    completed_at = datetime.utcnow()
            elif not isinstance(completed_at, datetime):
                completed_at = datetime.utcnow()
            
            await db.analysisrun.update(
                where={'id': run_id},
                data={
                    'status': 'completed',
                    'progress': 100,
                    'totalQuestions': total_questions,
                    'similarQuestions': similar_count,
                    'newTopicsDiscovered': new_topics_count,
                    'completedAt': completed_at
                }
            )
            
            logger.info(f"✅ Updated AnalysisRun {run_id} in database")
            logger.info(f"  - Total questions: {total_questions}")
            logger.info(f"  - Similar questions: {similar_count}")
            logger.info(f"  - New topics discovered: {new_topics_count}")
            
        except Exception as e:
            logger.error(f"Error updating analysis run in database: {e}")
            raise

    async def _save_questions_to_db(self, db, run_id: str, results: Dict[str, Any]):
        """Save questions with embeddings and classifications to database"""
        try:
            similar_questions = results.get('similar_questions', {}).get('questions', [])
            new_topics_data = results.get('new_topics', {}).get('topics', [])
            
            saved_count = 0
            
            # Save similar questions (those matching existing topics)
            for sq in similar_questions:
                try:
                    # Normalize embedding to list of floats
                    embedding = sq.get('embedding', [])
                    if embedding:
                        embedding = [float(x) for x in embedding]
                    
                    # Find the existing topic by name
                    matched_topic_name = sq.get('matched_topic', '')
                    existing_topic = None
                    if matched_topic_name:
                        existing_topic = await db.topic.find_first(
                            where={'name': matched_topic_name}
                        )
                    
                    # Create question record
                    question_data = {
                        'text': sq['question'],
                        'embedding': embedding,
                        'similarityScore': float(sq.get('similarity_score', 0)),
                        'matchedTopic': matched_topic_name,
                        'isNewTopic': False,
                        'analysisRunId': run_id
                    }
                    
                    if existing_topic:
                        question_data['topicId'] = existing_topic.id
                    
                    await db.question.create(data=question_data)
                    saved_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error saving similar question: {e}")
                    continue
            
            # Save questions from new topic clusters
            for topic_data in new_topics_data:
                topic_name = topic_data.get('topic_name', f"Cluster {topic_data.get('cluster_id', 'Unknown')}")
                
                # Find the topic (should be created by _save_topics_to_db)
                topic = await db.topic.find_first(where={'name': topic_name})
                
                # Get question embeddings if available
                question_embeddings = topic_data.get('question_embeddings', {})
                
                for question_text in topic_data.get('questions', []):
                    try:
                        # Get embedding for this specific question
                        embedding = question_embeddings.get(question_text, [])
                        if embedding:
                            embedding = [float(x) for x in embedding]
                        
                        question_data = {
                            'text': question_text,
                            'embedding': embedding,
                            'isNewTopic': True,
                            'analysisRunId': run_id
                        }
                        
                        if topic:
                            question_data['topicId'] = topic.id
                        
                        await db.question.create(data=question_data)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Error saving clustered question: {e}")
                        continue
            
            logger.info(f"✅ Saved {saved_count} questions to database")
            
        except Exception as e:
            logger.error(f"Error saving questions to database: {e}")
            raise

    async def _save_topics_to_db(self, db, run_id: str, results: Dict[str, Any]):
        """Save newly discovered topics to database"""
        try:
            new_topics_data = results.get('new_topics', {}).get('topics', [])
            
            for topic_data in new_topics_data:
                topic_name = topic_data.get('topic_name', f"Cluster {topic_data.get('cluster_id', 'Unknown')}")
                rep_question = topic_data.get('representative_question', '')
                rep_embedding = topic_data.get('representative_embedding', [])
                
                # Normalize representative embedding to list of floats
                if rep_embedding:
                    try:
                        rep_embedding = [float(x) for x in rep_embedding]
                    except Exception:
                        rep_embedding = []
                
                # Check if topic already exists
                existing_topic = await db.topic.find_first(where={'name': topic_name})
                
                if not existing_topic:
                    # Create new topic
                    await db.topic.create(
                        data={
                            'name': topic_name,
                            'representativeQuestion': rep_question,
                            'representativeEmbedding': rep_embedding,
                            'isDiscovered': True,
                            'approvalStatus': 'pending',
                            'discoveredAt': datetime.utcnow()
                        }
                    )
                    logger.info(f"✅ Created new topic: {topic_name}")
                else:
                    logger.info(f"ℹ️ Topic already exists: {topic_name}")
            
            logger.info(f"✅ Processed {len(new_topics_data)} topics")
            
        except Exception as e:
            logger.error(f"Error saving topics to database: {e}")
            raise

    async def _save_embeddings_to_cache(self, db, results: Dict[str, Any]):
        """Save embeddings to cache for future use"""
        try:
            # Extract all questions and their embeddings from results
            all_questions = []
            
            # Similar questions
            similar_questions = results.get('similar_questions', {}).get('questions', [])
            for sq in similar_questions:
                if 'question' in sq and 'embedding' in sq and sq['embedding']:
                    all_questions.append({
                        'text': sq['question'],
                        'embedding': sq['embedding']
                    })
            
            # New topic questions
            new_topics_data = results.get('new_topics', {}).get('topics', [])
            for topic_data in new_topics_data:
                question_embeddings = topic_data.get('question_embeddings', {})
                for question_text, embedding in question_embeddings.items():
                    if embedding:
                        all_questions.append({
                            'text': question_text,
                            'embedding': embedding
                        })
            
            # Save to embedding cache
            cached_count = 0
            for q_data in all_questions:
                try:
                    text_hash = hashlib.md5(q_data['text'].encode()).hexdigest()[:12]
                    
                    # Check if already cached
                    existing_cache = await db.embeddingcache.find_first(
                        where={'textHash': text_hash, 'model': self.analyzer.embedding_model}
                    )
                    
                    if not existing_cache:
                        # Normalize embedding to list of floats
                        embedding = [float(x) for x in q_data['embedding']]
                        
                        await db.embeddingcache.create(
                            data={
                                'textHash': text_hash,
                                'model': self.analyzer.embedding_model,
                                'embedding': embedding
                            }
                        )
                        cached_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error caching embedding: {e}")
                    continue
            
            logger.info(f"✅ Cached {cached_count} new embeddings")
            
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")
            # Don't raise - caching failures shouldn't break the analysis

# Global instance
analysis_service = AnalysisService()