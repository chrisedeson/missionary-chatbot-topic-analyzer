# backend/app/services/analysis.py

"""
Hybrid Topic Discovery and Classification System

This implements the EXACT algorithm from hybrid_topic_discovery_and_classification.py
with database integration for the missionary chatbot topic analyzer.

EXACT CONFIGURATION FROM REFERENCE (DO NOT CHANGE):
- Similarity threshold: 0.70
- GPT model: gpt-5-nano
- Embedding model: text-embedding-3-small  
- UMAP components: 5
- HDBSCAN min cluster size: 3
- Random seed: 42
- Representative question method: centroid
"""

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

# ML imports - EXACT from reference
try:
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
except ImportError as e:
    logging.warning(f"ML dependencies not available: {e}. Install with: pip install umap-learn hdbscan bertopic")

from app.core.database import db as prisma
from app.core.config import settings

logger = logging.getLogger(__name__)

# ====================================================================
# EXACT CONFIGURATION FROM REFERENCE (DO NOT CHANGE)
# ====================================================================

SIMILARITY_THRESHOLD = 0.70  # Minimum similarity to match existing topics
REPRESENTATIVE_QUESTION_METHOD = "centroid"  # Options: "centroid" or "frequent"

# OpenAI and GPT settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Default for text-embedding-3-small
GPT_MODEL = "gpt-5-nano"  # Options: "gpt-5-nano" or "gpt-5-mini"

# Clustering settings
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Tighter clusters
RANDOM_SEED = 42

# Batch processing
BATCH_SIZE = 100  # Batch size for embedding generation

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
    def clean_question(question: str) -> str:
        """
        Remove ACM question prefix from questions before processing.
        EXACT from reference file.
        """
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        import re
        pattern = r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*'
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else question
    
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
    async def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """
        Get embedding for text with database caching support.
        EXACT from reference file.
        """
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
        """
        Get embeddings for multiple texts with batch processing and database caching.
        EXACT from reference file.
        """
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


class GPT5Config:
    """
    Configuration for GPT-5 models with proper parameter handling.
    EXACT from reference file.
    """
    
    def __init__(self, model: str = GPT_MODEL):
        self.MODEL = model
        self.MAX_COMPLETION_TOKENS = 1000
        self.TEMPERATURE = 1  # GPT-5 requires temperature = 1
        self.MAX_RETRIES = 3
    
    def get_api_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get API parameters for GPT-5 models"""
        params = {
            "model": self.MODEL,
            "messages": messages,
            "max_completion_tokens": self.MAX_COMPLETION_TOKENS,
        }
        # Only add temperature if it's not the default
        if self.TEMPERATURE != 1:
            params["temperature"] = self.TEMPERATURE
        return params


# Initialize GPT-5 configuration
gpt5_config = GPT5Config(GPT_MODEL)


class AnalysisService:
    """
    Main analysis service implementing hybrid topic discovery.
    EXACT algorithm from reference file.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.active_runs = {}  # Track active analysis runs
    
    async def find_best_topic_match(self, question_embedding: List[float]) -> Optional[Dict]:
        """
        Find the best matching topic for a question embedding using cosine similarity.
        EXACT from reference file.
        """
        if not question_embedding or len(question_embedding) != EMBEDDING_DIMENSIONS:
            logger.error(f"Invalid question embedding dimension")
            return None
        
        # Get all topics with their representative embeddings
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
    
    async def classify_by_similarity(self, questions: List[Dict], threshold: float = SIMILARITY_THRESHOLD) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify questions by similarity to existing topics.
        EXACT from reference file - Step 1: Similarity Filtering.
        """
        logger.info(f"Starting similarity-based classification for {len(questions)} questions")
        logger.info(f"Similarity threshold: {threshold}")
        
        # Extract question texts
        question_texts = [q['text'] for q in questions]
        
        # Generate embeddings for all questions
        question_embeddings = await self.embedding_service.get_embeddings_batch(question_texts)
        
        similar_questions = []
        remaining_questions = []
        
        for i, (question, embedding) in enumerate(zip(questions, question_embeddings)):
            if embedding and len(embedding) == EMBEDDING_DIMENSIONS:
                best_match = await self.find_best_topic_match(embedding)
                
                if best_match and best_match['similarity'] >= threshold:
                    similar_questions.append({
                        **question,
                        'embedding': embedding,
                        'matched_topic_id': best_match['topic_id'],
                        'matched_topic_name': best_match['topic_name'],
                        'matched_subtopic': best_match['subtopic'],
                        'similarity_score': best_match['similarity'],
                        'is_new_topic': False
                    })
                    logger.debug(f"Matched question to topic: {best_match['topic_name']} (similarity: {best_match['similarity']:.3f})")
                else:
                    remaining_questions.append({
                        **question,
                        'embedding': embedding,
                        'is_new_topic': True
                    })
                    logger.debug(f"Question needs clustering: {question['text'][:50]}...")
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
        max_tries=gpt5_config.MAX_RETRIES,
        base=2,
        max_value=60
    )
    async def generate_topic_name_gpt5(self, questions: List[str], keywords: str) -> str:
        """
        Generate a topic name using GPT-5 with retry logic.
        EXACT PROMPT from reference file - DO NOT CHANGE A SINGLE LETTER.
        """
        # Limit to top 10 questions for context
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

Example:
Questions:
- When does registration open?
- What are the fall 2025 enrollment deadlines?
Keywords: registration, deadlines

Topic name: Fall 2025 Registration Deadlines

Now generate the topic name for the questions above:
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert at creating clear, descriptive topic names for student question categories."},
                {"role": "user", "content": prompt}
            ]
            
            api_params = gpt5_config.get_api_params(messages)
            response = await async_client.chat.completions.create(**api_params)
            
            topic_name = response.choices[0].message.content.strip()
            
            # Clean up the response
            topic_name = topic_name.replace("Topic name:", "").strip()
            topic_name = topic_name.strip('"\'')
            
            # Validate and limit length
            if len(topic_name) > 100 or len(topic_name) < 3:
                raise ValueError(f"Invalid topic name length: {topic_name}")
            
            return topic_name
            
        except Exception as e:
            logger.warning(f"GPT-5 topic naming failed: {e}")
            # Fallback to keyword-based name
            return f"Topic: {keywords[:50]}"
    
    def select_representative_question(self, cluster_questions: List[Dict], method: str = REPRESENTATIVE_QUESTION_METHOD) -> Tuple[str, List[float]]:
        """
        Select representative question for a cluster.
        EXACT from reference file.
        """
        questions = [q['question'] for q in cluster_questions]
        
        if method == "centroid":
            # Find question closest to cluster centroid
            embeddings = np.array([q['embedding'] for q in cluster_questions])
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate distances to centroid
            distances = [cosine(emb, centroid) for emb in embeddings]
            closest_idx = np.argmin(distances)
            
            return questions[closest_idx], embeddings[closest_idx].tolist()
        
        elif method == "frequent":
            # Select shortest question as proxy for most common pattern
            shortest_idx = min(range(len(questions)), key=lambda i: len(questions[i]))
            return questions[shortest_idx], cluster_questions[shortest_idx]['embedding']
        
        else:
            # Default to first question
            return questions[0], cluster_questions[0]['embedding']
    
    async def discover_new_topics(self, remaining_questions: List[Dict]) -> List[Dict]:
        """
        Discover new topics from remaining questions using clustering.
        EXACT from reference file - Step 2: Clustering-Based Topic Discovery.
        """
        if not remaining_questions:
            return []
        
        logger.info(f"Starting topic discovery for {len(remaining_questions)} questions")
        logger.info(f"UMAP n_components: {UMAP_N_COMPONENTS}, HDBSCAN min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZE}")
        
        # Extract embeddings and questions
        questions = [q['text'] for q in remaining_questions]
        embeddings = np.array([q['embedding'] for q in remaining_questions])
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Step 1: Dimensionality reduction with UMAP
        logger.info(f"Reducing dimensions: {embeddings.shape[1]} → {UMAP_N_COMPONENTS}")
        umap_model = UMAP(
            n_components=UMAP_N_COMPONENTS,
            min_dist=0.0,
            metric='cosine',
            random_state=RANDOM_SEED
        )
        reduced_embeddings = umap_model.fit_transform(embeddings)
        logger.info(f"UMAP reduction complete: {reduced_embeddings.shape}")
        
        # Step 2: Clustering with HDBSCAN
        logger.info(f"Clustering with HDBSCAN (min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZE})")
        hdbscan_model = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        clusters = hdbscan_model.fit_predict(reduced_embeddings)
        
        # Analyze clustering results
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        n_clusters = len(unique_clusters[unique_clusters != -1])  # Exclude noise cluster (-1)
        unclustered_count = counts[unique_clusters == -1][0] if -1 in unique_clusters else 0
        clustered_count = len(clusters) - unclustered_count
        
        logger.info(f"Clustering complete!")
        logger.info(f"Number of clusters found: {n_clusters}")
        logger.info(f"Questions clustered: {clustered_count} ({clustered_count/len(questions)*100:.1f}%)")
        logger.info(f"Questions not clustered (noise): {unclustered_count} ({unclustered_count/len(questions)*100:.1f}%)")
        
        if n_clusters == 0:
            logger.warning("No clusters found - adjusting parameters might help")
            return []
        
        # Step 3: BERTopic for topic extraction
        logger.info("Running BERTopic for topic extraction...")
        topic_model = BERTopic(
            embedding_model=None,  # Use our precomputed embeddings
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=False
        )
        
        topics, probabilities = topic_model.fit_transform(questions, embeddings)
        topic_info = topic_model.get_topic_info()
        logger.info(f"BERTopic analysis complete - {len(topic_info)-1} topics discovered")
        
        # Group questions by cluster
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:  # Skip noise
                continue
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append({
                'question': questions[i],
                'embedding': embeddings[i].tolist(),
                'question_index': i,  # Track original index
                **remaining_questions[i]
            })
        
        # Get topic keywords from BERTopic
        topic_map = topic_info.set_index("Topic")["Representation"].to_dict()
        
        # Generate topic names and select representative questions
        new_topics = []
        from collections import Counter
        
        for cluster_id, cluster_questions in cluster_dict.items():
            if len(cluster_questions) < 3:  # Skip very small clusters
                continue
            
            # Extract keywords from BERTopic or fallback to simple word frequency
            topic_id = clusters[cluster_questions[0]['question_index']]
            topic_rep = topic_map.get(topic_id, [])
            
            if isinstance(topic_rep, list) and len(topic_rep) > 0:
                keywords = ", ".join(topic_rep[:5])
            else:
                # Fallback: simple word frequency
                all_text = ' '.join([q['question'] for q in cluster_questions])
                words = all_text.lower().split()
                common_words = [word for word, count in Counter(words).most_common(10) if len(word) > 3]
                keywords = ', '.join(common_words[:5])
            
            # Generate topic name using GPT-5
            questions_text = [q['question'] for q in cluster_questions]
            topic_name = await self.generate_topic_name_gpt5(questions_text, keywords)
            
            # Select representative question
            representative_question, representative_embedding = self.select_representative_question(
                cluster_questions,
                REPRESENTATIVE_QUESTION_METHOD
            )
            
            new_topics.append({
                'name': topic_name,
                'questions': cluster_questions,
                'representative_question': representative_question,
                'representative_embedding': representative_embedding,
                'question_count': len(cluster_questions)
            })
        
        logger.info(f"Topic discovery complete: {len(new_topics)} new topics created")
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
        """
        Run the complete hybrid analysis pipeline.
        EXACT workflow from reference file.
        """
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
                    "progress": 50,
                    "message": "Saving classification results..."
                }
            )
            
            # Save classification results
            await self.save_classification_results(similar_questions, remaining_questions, analysis_run_id)
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 60,
                    "message": "Discovering new topics with UMAP+HDBSCAN..."
                }
            )
            
            # Step 2: Topic discovery for remaining questions
            new_topics = await self.discover_new_topics(remaining_questions)
            
            await prisma.analysisrun.update(
                where={"id": analysis_run_id},
                data={
                    "progress": 85,
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
            logger.error(f"Analysis failed: {e}", exc_info=True)
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
        questions = await prisma.question.find_many(
            where={"analysisRunId": run_id},
            include={"topic": True}
        )
        
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
        run = await prisma.analysisrun.find_unique(where={"id": run_id})
        if not run:
            raise ValueError(f"Analysis run {run_id} not found")
        
        topics = await self.get_topics(run_id)
        
        return {
            "run_id": run_id,
            "status": run.status,
            "started_at": run.startedAt.isoformat() if run.startedAt else None,
            "completed_at": run.completedAt.isoformat() if run.completedAt else None,
            "total_questions": run.totalQuestions,
            "similar_questions": run.similarQuestions,
            "new_topics_discovered": run.newTopicsDiscovered,
            "topics": topics
        }


# Factory function for dependency injection
async def get_analysis_service() -> AnalysisService:
    """Factory function to get AnalysisService instance"""
    return AnalysisService()