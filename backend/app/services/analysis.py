# backend/app/services/analysis.py

import asyncio
import hashlib
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import backoff
from openai import OpenAI, AsyncOpenAI, APIStatusError

from app.core.database import db as prisma
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
GPT_MODEL = "gpt-4"  # Using GPT-4 as GPT-5 isn't available yet
SIMILARITY_THRESHOLD = 0.70
BATCH_SIZE = 100  # Batch size for embedding generation
CLUSTERING_BATCH_SIZE = 1000

# Initialize OpenAI clients
client = OpenAI(api_key=settings.OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class EmbeddingService:
    """Service for handling embeddings with database caching"""
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate MD5 hash of text for cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    @staticmethod
    async def get_cached_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
        """Get embedding from database cache if available"""
        text_hash = EmbeddingService._hash_text(text)
        
        cached = await prisma.embeddingcache.find_unique(
            where={"textHash": text_hash}
        )
        
        if cached and cached.model == model:
            # Update accessedAt timestamp
            await prisma.embeddingcache.update(
                where={"id": cached.id},
                data={"accessedAt": datetime.utcnow()}
            )
            return cached.embedding
        return None
    
    @staticmethod
    async def save_embedding_to_cache(text: str, model: str, embedding: List[float]):
        """Save embedding to database cache"""
        text_hash = EmbeddingService._hash_text(text)
        
        await prisma.embeddingcache.upsert(
            where={"textHash": text_hash},
            data={
                "create": {
                    "textHash": text_hash,
                    "model": model,
                    "embedding": embedding,
                    "accessedAt": datetime.utcnow()
                },
                "update": {
                    "embedding": embedding,
                    "model": model,
                    "accessedAt": datetime.utcnow()
                }
            }
        )
    
    @staticmethod
    def clean_question(question: str) -> str:
        """Remove ACM question prefix from questions before processing"""
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        import re
        pattern = r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*'
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else question
    
    @staticmethod
    async def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """Get embedding for text with database caching support"""
        cleaned_text = EmbeddingService.clean_question(text)
        
        # Try to load from cache first
        cached_embedding = await EmbeddingService.get_cached_embedding(cleaned_text, model)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        try:
            response = client.embeddings.create(
                model=model,
                input=cleaned_text.replace("\n", " ")
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            await EmbeddingService.save_embedding_to_cache(cleaned_text, model, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {cleaned_text[:50]}...")
            logger.error(f"Error details: {e}")
            # Return zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSIONS
    
    @staticmethod
    async def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """Get embeddings for multiple texts with true batch processing and database caching"""
        cleaned_texts = [EmbeddingService.clean_question(text) for text in texts]
        embeddings = []
        cache_hits = 0
        api_calls = 0
        
        logger.info(f"Generating embeddings for {len(cleaned_texts)} texts...")
        
        # Process in batches for API efficiency
        for i in range(0, len(cleaned_texts), BATCH_SIZE):
            batch_texts = cleaned_texts[i:i+BATCH_SIZE]
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                cached_embedding = await EmbeddingService.get_cached_embedding(text, model)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=uncached_texts
                    )
                    
                    new_embeddings = [data.embedding for data in response.data]
                    api_calls += len(uncached_texts)
                    
                    # Fill in the uncached embeddings and save to cache
                    cache_tasks = []
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings[idx] = embedding
                        # Schedule cache save
                        cache_tasks.append(
                            EmbeddingService.save_embedding_to_cache(
                                batch_texts[idx], model, embedding
                            )
                        )
                    
                    # Wait for all cache operations to complete
                    await asyncio.gather(*cache_tasks)
                    
                    # Rate limiting
                    if api_calls > 0 and api_calls % 50 == 0:
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    # Fill with zero vectors for failed embeddings
                    for idx in uncached_indices:
                        batch_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS
            
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Embedding generation complete: {cache_hits} cache hits, {api_calls} API calls")
        return embeddings

class AnalysisService:
    """Main analysis service implementing hybrid topic discovery"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.active_runs = {}  # Track active analysis runs
    
    async def find_best_topic_match(self, question_embedding: List[float]) -> Optional[Dict]:
        """Find the best matching topic for a question embedding using cosine similarity"""
        if not question_embedding or len(question_embedding) != EMBEDDING_DIMENSIONS:
            logger.error(f"Invalid question embedding dimension")
            return None
        
        # Get all topics with their representative embeddings
        # Note: Cannot filter array fields with "not None" in Prisma Python, so we filter in code
        topics = await prisma.topic.find_many()
        
        best_distance = float('inf')
        best_match = None
        
        for topic in topics:
            # Skip topics without embeddings or with empty embeddings
            if not topic.representativeEmbedding or len(topic.representativeEmbedding) == 0:
                continue
                
            if len(topic.representativeEmbedding) != EMBEDDING_DIMENSIONS:
                logger.warning(f"Invalid embedding dimension for topic {topic.name}")
                continue
            
            try:
                distance = cosine(question_embedding, topic.representativeEmbedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = {
                        'topic_id': topic.id,
                        'topic_name': topic.name,
                        'subtopic': topic.subtopic,
                        'distance': distance,
                        'similarity': 1 - distance
                    }
            except Exception as e:
                logger.error(f"Failed to calculate similarity for topic {topic.name}: {e}")
                continue
        
        return best_match
    
    async def classify_by_similarity(self, questions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Classify questions by similarity to existing topics"""
        logger.info(f"Starting similarity-based classification for {len(questions)} questions")
        
        # Extract question texts
        question_texts = [q['text'] for q in questions]
        
        # Generate embeddings for all questions
        question_embeddings = await self.embedding_service.get_embeddings_batch(question_texts)
        
        similar_questions = []
        remaining_questions = []
        
        for i, (question, embedding) in enumerate(zip(questions, question_embeddings)):
            if embedding and len(embedding) == EMBEDDING_DIMENSIONS:
                best_match = await self.find_best_topic_match(embedding)
                
                if best_match and best_match['similarity'] >= SIMILARITY_THRESHOLD:
                    similar_questions.append({
                        **question,
                        'embedding': embedding,
                        'matched_topic_id': best_match['topic_id'],
                        'matched_topic_name': best_match['topic_name'],
                        'matched_subtopic': best_match['subtopic'],
                        'similarity_score': best_match['similarity'],
                        'is_new_topic': False
                    })
                else:
                    remaining_questions.append({
                        **question,
                        'embedding': embedding,
                        'is_new_topic': True
                    })
            else:
                logger.warning(f"Invalid embedding for question: {question['text'][:50]}...")
                remaining_questions.append({
                    **question,
                    'embedding': [0.0] * EMBEDDING_DIMENSIONS,
                    'is_new_topic': True
                })
        
        logger.info(f"Similarity classification complete: {len(similar_questions)} similar, {len(remaining_questions)} remaining")
        return similar_questions, remaining_questions
    
    async def save_classification_results(self, similar_questions: List[Dict], remaining_questions: List[Dict], analysis_run_id: str):
        """Save classification results to database"""
        # Save similar questions (matched to existing topics)
        for question in similar_questions:
            await prisma.question.update(
                where={"id": question['id']},
                data={
                    "embedding": question['embedding'],
                    "similarityScore": float(question['similarity_score']),
                    "matchedTopic": question['matched_topic_name'],
                    "isNewTopic": False,
                    "topicId": question['matched_topic_id'],
                    "analysisRunId": analysis_run_id
                }
            )
        
        # Save remaining questions (for clustering)
        for question in remaining_questions:
            await prisma.question.update(
                where={"id": question['id']},
                data={
                    "embedding": question['embedding'],
                    "isNewTopic": True,
                    "analysisRunId": analysis_run_id
                }
            )
    
    @backoff.on_exception(
        backoff.expo,
        (APIStatusError, asyncio.TimeoutError),
        max_tries=3
    )
    async def generate_topic_name(self, questions: List[str], keywords: str) -> str:
        """Generate a topic name using GPT"""
        sample_questions = questions[:10]
        questions_text = "\n".join([f"- {q}" for q in sample_questions])
        
        prompt = f"""
        Based on the following student questions and keywords, generate a concise, descriptive topic name.

        QUESTIONS:
        {questions_text}

        KEYWORDS: {keywords}

        Instructions:
        - Your answer must be ONLY the topic name (2–8 words), no extra text.
        - It should clearly describe the shared theme of the questions.
        - Avoid generic labels like "General Questions" or "Miscellaneous."
        - Do not include "Topic name:" or quotation marks.
        - Use simple, natural English that sounds clear to a student or teacher.

        Now generate the topic name for the questions above:
        """
        
        try:
            response = await async_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, descriptive topic names for student question categories."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            topic_name = response.choices[0].message.content.strip()
            topic_name = topic_name.replace("Topic name:", "").strip()
            topic_name = topic_name.strip('"\'')
            
            if len(topic_name) > 100 or len(topic_name) < 3:
                return f"Topic: {keywords[:50]}"
                
            return topic_name
            
        except Exception as e:
            logger.warning(f"GPT topic naming failed: {e}")
            return f"Topic: {keywords[:50]}"
    
    async def discover_new_topics(self, remaining_questions: List[Dict]) -> List[Dict]:
        """Discover new topics from remaining questions using clustering"""
        if not remaining_questions:
            return []
        
        logger.info(f"Starting topic discovery for {len(remaining_questions)} questions")
        
        # Extract embeddings for clustering
        embeddings = np.array([q['embedding'] for q in remaining_questions])
        
        # Simple clustering implementation - in production you might use HDBSCAN or similar
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.5, min_samples=3, metric='cosine').fit(embeddings)
        cluster_labels = clustering.labels_
        
        # Group questions by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(remaining_questions[i])
        
        logger.info(f"Found {len(clusters)} clusters from {len(remaining_questions)} questions")
        
        # Generate topic names for each cluster
        new_topics = []
        for cluster_id, cluster_questions in clusters.items():
            if len(cluster_questions) < 3:  # Skip very small clusters
                continue
                
            questions_text = [q['text'] for q in cluster_questions]
            
            # Extract keywords (simplified - in production use proper keyword extraction)
            all_text = ' '.join(questions_text)
            words = all_text.lower().split()
            from collections import Counter
            common_words = [word for word, count in Counter(words).most_common(10) if len(word) > 3]
            keywords = ', '.join(common_words[:5])
            
            # Generate topic name
            topic_name = await self.generate_topic_name(questions_text, keywords)
            
            # Select representative question (closest to centroid)
            cluster_embeddings = np.array([q['embedding'] for q in cluster_questions])
            centroid = np.mean(cluster_embeddings, axis=0)
            similarities = [cosine_similarity([emb], [centroid])[0][0] for emb in cluster_embeddings]
            rep_question_idx = np.argmax(similarities)
            representative_question = cluster_questions[rep_question_idx]['text']
            representative_embedding = cluster_questions[rep_question_idx]['embedding']
            
            new_topics.append({
                'name': topic_name,
                'questions': cluster_questions,
                'representative_question': representative_question,
                'representative_embedding': representative_embedding,
                'question_count': len(cluster_questions)
            })
        
        return new_topics
    
    async def save_new_topics(self, new_topics: List[Dict], analysis_run_id: str):
        """Save newly discovered topics to database"""
        for topic_data in new_topics:
            # Create new topic
            topic = await prisma.topic.create({
                'name': topic_data['name'],
                'representativeQuestion': topic_data['representative_question'],
                'representativeEmbedding': topic_data['representative_embedding'],
                'isDiscovered': True,
                'discoveredAt': datetime.utcnow(),
                'approvalStatus': 'pending'
            })
            
            # Update questions with new topic ID
            question_ids = [q['id'] for q in topic_data['questions']]
            await prisma.question.update_many(
                where={'id': {'in': question_ids}},
                data={'topicId': topic.id}
            )
    
    async def run_analysis(self, analysis_run_id: str, mode: str = "all", sample_size: Optional[int] = None):
        """Run the complete hybrid analysis pipeline"""
        try:
            # Update analysis run status
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "status": "running",
                    "progress": 10,
                    "message": "Loading questions..."
                }
            )
            
            # Load questions based on mode
            # Note: Allow re-analysis of all questions, not just unprocessed ones
            if mode == "sample" and sample_size:
                questions = await prisma.question.find_many(
                    take=sample_size
                )
            else:
                questions = await prisma.question.find_many()
            
            if not questions:
                await prisma.analysisrun.update(
                    where={"id": analysis_run_id},
                    data={
                        "status": "completed",
                        "progress": 100,
                        "message": "No questions to process",
                        "completedAt": datetime.utcnow()
                    }
                )
                return
            
            questions_dict = [q.dict() for q in questions]
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 30,
                    "message": "Running similarity classification..."
                }
            )
            
            # Step 1: Similarity-based classification
            similar_questions, remaining_questions = await self.classify_by_similarity(questions_dict)
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 60,
                    "message": "Saving classification results..."
                }
            )
            
            # Save classification results
            await self.save_classification_results(similar_questions, remaining_questions, analysis_run_id)
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 70,
                    "message": "Discovering new topics..."
                }
            )
            
            # Step 2: Topic discovery for remaining questions
            new_topics = await self.discover_new_topics(remaining_questions)
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 90,
                    "message": "Saving new topics..."
                }
            )
            
            # Save new topics
            await self.save_new_topics(new_topics, analysis_run_id)
            
            # Update analysis run with final results
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "status": "completed",
                    "progress": 100,
                    "totalQuestions": len(questions),
                    "similarQuestions": len(similar_questions),
                    "newTopicsDiscovered": len(new_topics),
                    "completedAt": datetime.utcnow(),
                    "message": "Analysis completed successfully"
                }
            )
            
            logger.info(f"Analysis completed: {len(similar_questions)} similar, {len(new_topics)} new topics")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "status": "failed",
                    "error": str(e),
                    "failedAt": datetime.utcnow(),
                    "message": f"Analysis failed: {str(e)}"
                }
            )
            raise
    
    async def get_questions_count(self) -> int:
        """Get count of questions available for analysis"""
        # Count all questions (allow re-analysis)
        count = await prisma.question.count()
        return count
    
    async def start_analysis(self, mode: str = "all", sample_size: Optional[int] = None) -> str:
        """Start a new analysis run and return the run ID"""
        # Create a new analysis run
        create_data = {
            "id": str(uuid.uuid4()),
            "status": "pending",
            "progress": 0,
            "message": "Analysis queued",
            "mode": mode,
            "startedAt": datetime.utcnow()
        }
        
        if sample_size is not None:
            create_data["sampleSize"] = sample_size
        
        analysis_run = await prisma.analysisrun.create(create_data)
        
        run_id = analysis_run.id
        self.active_runs[run_id] = {
            "status": "pending",
            "started_at": datetime.utcnow(),
            "progress": {}
        }
        
        # Start analysis in background
        asyncio.create_task(self.run_analysis(run_id, mode, sample_size))
        
        return run_id
    
    def get_run_status(self, run_id: str) -> Optional[Dict]:
        """Get status of an analysis run"""
        return self.active_runs.get(run_id)
    
    async def get_analysis_history(self, limit: int = 20) -> List[Dict]:
        """Get history of analysis runs"""
        runs = await prisma.analysisrun.find_many(
            take=limit,
            order={"startedAt": "desc"}
        )
        return [run.dict() for run in runs]
    
    async def get_topics(self, run_id: str) -> List[Dict]:
        """Get topics discovered in a specific analysis run"""
        # Get all questions for this run grouped by topic
        questions = await prisma.question.find_many(
            where={"analysisRunId": run_id},
            include={"topic": True}
        )
        
        # Group by topic
        topics_dict = {}
        for q in questions:
            if q.topicId:
                if q.topicId not in topics_dict:
                    topics_dict[q.topicId] = {
                        "id": q.topic.id,
                        "name": q.topic.name,
                        "representative_question": q.topic.representativeQuestion,
                        "questions": []
                    }
                topics_dict[q.topicId]["questions"].append(q.text)
        
        return list(topics_dict.values())
    
    async def export_results(self, run_id: str) -> Dict:
        """Export complete results for an analysis run"""
        # Get run info
        run = await prisma.analysisrun.find_unique(where={"id": run_id})
        if not run:
            raise ValueError(f"Analysis run {run_id} not found")
        
        # Get all questions and topics
        topics = await self.get_topics(run_id)
        
        return {
            "run_id": run_id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "total_questions": run.total_questions,
            "similar_questions": run.similar_questions,
            "new_topics_discovered": run.new_topics_discovered,
            "topics": topics
        }


# Factory function for dependency injection
async def get_analysis_service() -> AnalysisService:
    """Factory function to get AnalysisService instance"""
    return AnalysisService()