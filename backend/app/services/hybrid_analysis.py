"""
Hybrid Topic Discovery and Classification System

This module implements the exact hybrid analysis algorithm from the reference script
with database-backed embedding caching integrated with the analysis.py service.

EXACT CONFIGURATION FROM REFERENCE:
- Similarity threshold: 0.70
- GPT model: gpt-5-nano  
- Embedding model: text-embedding-3-small
- UMAP components: 5
- HDBSCAN min cluster size: 3
- Random seed: 42
- Representative question method: centroid
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime
import uuid

# ML imports
try:
    import umap
    from hdbscan import HDBSCAN
    from scipy.spatial.distance import cosine
    from openai import AsyncOpenAI, APIStatusError
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Optional dependencies not available: {e}")

# App imports
from app.core.config import settings
from app.core.database import get_db
from app.services.analysis import EmbeddingService
from prisma import Prisma
import backoff

logger = logging.getLogger(__name__)

# ====================================================================
# EXACT CONFIGURATION FROM REFERENCE (DO NOT CHANGE)
# ====================================================================

# Processing mode settings
EVAL_MODE = "sample"  # Options: "sample" or "all"
SAMPLE_SIZE = 2000     # Number of questions to evaluate in sample mode

# Similarity filtering settings
SIMILARITY_THRESHOLD = 0.70  # Minimum similarity to match existing topics
REPRESENTATIVE_QUESTION_METHOD = "centroid"  # Options: "centroid" or "frequent"

# OpenAI and GPT settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Default for text-embedding-3-small
GPT_MODEL = "gpt-5-nano"  # Options: "gpt-5-nano" or "gpt-5-mini"

# Clustering settings
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Tighter clusters (was 15)
RANDOM_SEED = 42

@dataclass
class ClusterResult:
    """Clustering result structure"""
    cluster_id: int
    topic_name: str
    questions: List[str]
    representative_question: str
    question_count: int


class GPT5Config:
    """Configuration for GPT-5 models with proper parameter handling (EXACT FROM REFERENCE)"""

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
            "temperature": self.TEMPERATURE  # Always include temperature for GPT-5
        }
        return params


class HybridTopicAnalyzer:
    """
    Hybrid Topic Discovery and Classification System
    
    Implements the EXACT algorithm from hybrid_topic_discovery_and_classification.py
    with database-backed embedding caching.
    """
    
    def __init__(self, db: Prisma):
        """Initialize with EXACT reference settings and database connection"""
        self.db = db
        
        # Initialize embedding service with database caching
        self.embedding_service = EmbeddingService(db)
        
        # OpenAI Configuration (EXACT from reference)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL  # "text-embedding-3-small"
        self.gpt_model = GPT_MODEL  # "gpt-5-nano"
        
        # Analysis Configuration (EXACT from reference)
        self.similarity_threshold = SIMILARITY_THRESHOLD  # 0.70
        self.representative_question_method = REPRESENTATIVE_QUESTION_METHOD  # "centroid"
        self.processing_mode = EVAL_MODE  # "sample" 
        self.sample_size = SAMPLE_SIZE  # 2000
        
        # Clustering Configuration (EXACT from reference)
        self.umap_n_components = UMAP_N_COMPONENTS  # 5
        self.min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE  # 3
        self.random_seed = RANDOM_SEED  # 42
        
        # Initialize GPT-5 configuration (EXACT from reference)
        self.gpt5_config = GPT5Config(self.gpt_model)
        
        logger.info(f"Initialized HybridTopicAnalyzer with EXACT reference configuration:")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   GPT model: {self.gpt_model}")
        logger.info(f"   Embedding model: {self.embedding_model}")
        logger.info(f"   Random seed: {self.random_seed}")
        logger.info(f"   Database caching: ENABLED")
    
    @staticmethod
    def clean_acm_prefix(question: str) -> str:
        """
        Remove ACM question prefix from questions before processing (EXACT FROM REFERENCE).
        """
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        pattern = r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*'
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else question
    
    def _clean_question_text(self, text: str) -> str:
        """Clean and normalize question text (EXACT from reference)"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove ACM prefix if present
        text = self.clean_acm_prefix(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not text:
            return ""
        
        # Remove common noise
        text = re.sub(r'^(Question:\s*|Q:\s*|\d+\.\s*)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        progress_callback: Optional[Callable] = None
    ) -> List[List[float]]:
        """
        Get embeddings using database-backed caching from EmbeddingService.
        This replaces the pickle file caching with database caching.
        """
        # Clean all texts first
        cleaned_texts = [self._clean_question_text(text) for text in texts]
        
        # Use the EmbeddingService which has database caching
        embeddings = await self.embedding_service.get_embeddings_batch(
            cleaned_texts,
            model=self.embedding_model,
            batch_size=1000  # Match reference batch size
        )
        
        # Progress callback support
        if progress_callback:
            await progress_callback(
                "embedding_generation", 
                100, 
                f"Generated embeddings for {len(embeddings)} texts"
            )
        
        return embeddings

    @backoff.on_exception(
        backoff.expo,
        (APIStatusError, asyncio.TimeoutError),
        max_tries=3,
        base=2,
        max_value=60
    )
    async def generate_topic_name_gpt5(self, questions: List[str], keywords: str) -> str:
        """Generate a topic name using GPT-5 with retry logic (EXACT FROM REFERENCE)"""

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
- Be specific and descriptive.

Example:
Questions:
- When does registration open?
- What are the fall 2025 enrollment deadlines?
Keywords: registration, deadlines

Output: Fall 2025 Registration Deadlines

Now generate the topic name:
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert at creating clear, descriptive topic names for student question categories. Always respond with just the topic name, nothing else."},
                {"role": "user", "content": prompt}
            ]

            api_params = self.gpt5_config.get_api_params(messages)
            logger.info(f"Calling GPT-5 with model: {api_params['model']}")
            
            response = await self.client.chat.completions.create(**api_params)

            topic_name = response.choices[0].message.content
            
            if topic_name:
                topic_name = topic_name.strip()
                # Clean up the response
                topic_name = topic_name.replace("Topic name:", "").strip()
                topic_name = topic_name.replace("Output:", "").strip()
                topic_name = topic_name.strip('"\'')

            # Validate and limit length
            if not topic_name or len(topic_name) > 100 or len(topic_name) < 3:
                logger.warning(f"Invalid topic name: '{topic_name}'")
                raise ValueError(f"Invalid topic name: {topic_name}")

            return topic_name

        except Exception as e:
            logger.warning(f"GPT-5 topic naming failed: {e}")
            # Fallback to keyword-based name
            if keywords and keywords.strip():
                fallback_name = f"Topic: {keywords[:47]}"
                logger.info(f"Using fallback topic name: {fallback_name}")
                return fallback_name
            else:
                return "Untitled Topic"

    async def perform_hybrid_analysis(
        self, 
        questions: List[str], 
        existing_topics: Optional[Dict[str, List[str]]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid topic discovery and classification (EXACT FROM REFERENCE WORKFLOW)
        
        Returns analysis results with similar questions and new topics discovered
        """
        logger.info(f"Starting hybrid analysis for {len(questions)} questions")
        
        try:
            # Step 1: Prepare data
            if progress_callback:
                await progress_callback("data_preparation", 5, "Preparing data for analysis...")
            
            # Apply sample mode if configured
            if self.processing_mode == "sample" and len(questions) > self.sample_size:
                np.random.seed(self.random_seed)
                sample_indices = np.random.choice(len(questions), self.sample_size, replace=False)
                eval_questions = [questions[i] for i in sample_indices]
                logger.info(f"Sample mode: Using {len(eval_questions)} questions (seed {self.random_seed})")
            else:
                eval_questions = questions
                logger.info(f"Full mode: Using all {len(eval_questions)} questions")
            
            # Step 2: Generate embeddings for all questions
            if progress_callback:
                await progress_callback("embedding_generation", 10, "Generating embeddings for questions...")
            
            question_embeddings = await self.get_embeddings_batch(
                eval_questions, 
                progress_callback=progress_callback
            )
            
            # Step 3: Similarity analysis
            similar_questions = []
            remaining_questions = []
            remaining_embeddings = []
            
            if existing_topics:
                if progress_callback:
                    await progress_callback("similarity_analysis", 30, "Analyzing similarity to existing topics...")
                
                # Flatten existing topics
                existing_questions = []
                for topic_name, topic_questions in existing_topics.items():
                    for q in topic_questions:
                        existing_questions.append(q)
                
                # Generate embeddings for existing topics
                existing_embeddings = await self.get_embeddings_batch(existing_questions)
                
                # Classify each question
                for question, embedding in zip(eval_questions, question_embeddings):
                    if embedding and len(embedding) == EMBEDDING_DIMENSIONS:
                        best_distance = float('inf')
                        best_match = None
                        
                        for existing_q, existing_emb in zip(existing_questions, existing_embeddings):
                            if existing_emb and len(existing_emb) == EMBEDDING_DIMENSIONS:
                                try:
                                    distance = cosine(embedding, existing_emb)
                                    similarity = 1 - distance
                                    
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_match = {
                                            'matched_question': existing_q,
                                            'similarity': similarity
                                        }
                                except Exception as e:
                                    logger.debug(f"Error calculating similarity: {e}")
                                    continue
                        
                        # Check threshold
                        if best_match and best_match['similarity'] >= self.similarity_threshold:
                            similar_questions.append({
                                'question': question,
                                'matched_topic': "Existing Topic",
                                'matched_question': best_match['matched_question'],
                                'similarity_score': best_match['similarity']
                            })
                        else:
                            remaining_questions.append(question)
                            remaining_embeddings.append(embedding)
                    else:
                        remaining_questions.append(question)
                        remaining_embeddings.append([0.0] * EMBEDDING_DIMENSIONS)
                        
                logger.info(f"Similarity: {len(similar_questions)} similar, {len(remaining_questions)} remaining")
            else:
                remaining_questions = eval_questions
                remaining_embeddings = question_embeddings
            
            # Step 4: Clustering
            new_topics = []
            if remaining_questions:
                if progress_callback:
                    await progress_callback("clustering", 60, f"Clustering {len(remaining_questions)} questions...")
                
                # Filter valid embeddings
                embeddings_array = np.array([
                    emb for emb in remaining_embeddings 
                    if emb and len(emb) == EMBEDDING_DIMENSIONS
                ])
                valid_questions = [
                    q for i, q in enumerate(remaining_questions) 
                    if remaining_embeddings[i] and len(remaining_embeddings[i]) == EMBEDDING_DIMENSIONS
                ]
                
                if len(embeddings_array) > 0:
                    # UMAP dimensionality reduction
                    logger.info(f"UMAP: {embeddings_array.shape[1]} → {self.umap_n_components}")
                    umap_model = umap.UMAP(
                        n_components=self.umap_n_components,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=self.random_seed
                    )
                    reduced_embeddings = umap_model.fit_transform(embeddings_array)
                    
                    # HDBSCAN clustering
                    logger.info(f"HDBSCAN clustering (min_size: {self.min_cluster_size})")
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        metric="euclidean",
                        cluster_selection_method="eom"
                    )
                    clusters = hdbscan_model.fit_predict(reduced_embeddings)
                    
                    # Generate topic names
                    if progress_callback:
                        await progress_callback("topic_naming", 80, "Generating topic names...")
                    
                    unique_clusters = set(clusters) - {-1}
                    logger.info(f"Found {len(unique_clusters)} clusters")
                    
                    for cluster_id in unique_clusters:
                        cluster_mask = clusters == cluster_id
                        cluster_questions = [q for i, q in enumerate(valid_questions) if cluster_mask[i]]
                        
                        if len(cluster_questions) >= self.min_cluster_size:
                            # Generate keywords
                            all_words = []
                            for q in cluster_questions[:5]:
                                words = re.findall(r'\b\w+\b', q.lower())
                                all_words.extend(words[:3])
                            keywords = ", ".join(list(set(all_words))[:5])
                            
                            # Generate topic name
                            try:
                                topic_name = await self.generate_topic_name_gpt5(cluster_questions, keywords)
                            except Exception as e:
                                logger.warning(f"Topic naming failed for cluster {cluster_id}: {e}")
                                topic_name = f"Topic {cluster_id}: {keywords[:30]}"
                            
                            # Select representative question
                            if self.representative_question_method == "centroid":
                                cluster_embeddings = embeddings_array[cluster_mask]
                                centroid = np.mean(cluster_embeddings, axis=0)
                                distances = [cosine(emb, centroid) for emb in cluster_embeddings]
                                closest_idx = np.argmin(distances)
                                representative_question = cluster_questions[closest_idx]
                            else:
                                representative_question = min(cluster_questions, key=len)
                            
                            new_topics.append(ClusterResult(
                                cluster_id=int(cluster_id),
                                topic_name=topic_name,
                                questions=cluster_questions,
                                representative_question=representative_question,
                                question_count=len(cluster_questions)
                            ))
            
            # Step 5: Prepare results
            if progress_callback:
                await progress_callback("finalizing", 95, "Finalizing analysis results...")
            
            results = {
                "analysis_id": str(uuid.uuid4()),
                "total_questions_analyzed": len(eval_questions),
                "processing_mode": self.processing_mode,
                "sample_size": len(eval_questions),
                "similarity_threshold": self.similarity_threshold,
                "random_seed": self.random_seed,
                "similar_questions": {
                    "count": len(similar_questions),
                    "questions": similar_questions
                },
                "new_topics": {
                    "count": len(new_topics),
                    "topics": [asdict(topic) for topic in new_topics]
                },
                "configuration": {
                    "embedding_model": self.embedding_model,
                    "gpt_model": self.gpt_model,
                    "umap_n_components": self.umap_n_components,
                    "min_cluster_size": self.min_cluster_size,
                    "similarity_threshold": self.similarity_threshold
                },
                "completed_at": datetime.now().isoformat()
            }
            
            if progress_callback:
                await progress_callback(
                    "completed", 
                    100, 
                    f"Analysis complete: {len(similar_questions)} similar, {len(new_topics)} new topics"
                )
            
            logger.info(f"Hybrid analysis completed successfully")
            logger.info(f"   Similar questions: {len(similar_questions)}")
            logger.info(f"   New topics: {len(new_topics)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}", exc_info=True)
            if progress_callback:
                await progress_callback("error", 0, f"Analysis failed: {str(e)}")
            raise


# Global factory function
async def get_hybrid_analyzer() -> HybridTopicAnalyzer:
    """Factory function to get HybridTopicAnalyzer with database connection"""
    db = await get_db()
    return HybridTopicAnalyzer(db)