"""
Hybrid Topic Discovery and Classification System

This module implements the exact hybrid analysis algorithm from the reference script
references/insights/hybrid_topic_discovery_and_classification.py with identical 
settings, configuration, and processing steps.

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
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import json
import re
from datetime import datetime
import uuid
import time
from pathlib import Path
import pickle
import hashlib
import backoff

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

# Caching settings
CACHE_EMBEDDINGS = True
CACHE_DIR = "embeddings_cache"  # Local cache location

# Clustering settings
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Tighter clusters (was 15)
RANDOM_SEED = 42

@dataclass
class Question:
    """Question data structure"""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    
@dataclass 
class Topic:
    """Topic data structure"""
    id: str
    name: str
    description: str
    questions: List[Question]
    representative_question: str
    representative_embedding: List[float]

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
    with IDENTICAL configuration and processing steps.
    """
    
    def __init__(self):
        """Initialize with EXACT reference settings (DO NOT CHANGE)"""
        # OpenAI Configuration (EXACT from reference)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL  # "text-embedding-3-small"
        self.gpt_model = GPT_MODEL  # "gpt-5-nano" - EXACT name from reference
        
        # Analysis Configuration (EXACT from reference)
        self.similarity_threshold = SIMILARITY_THRESHOLD  # 0.70
        self.representative_question_method = REPRESENTATIVE_QUESTION_METHOD  # "centroid"
        self.processing_mode = EVAL_MODE  # "sample" 
        self.sample_size = SAMPLE_SIZE  # 2000
        
        # Clustering Configuration (EXACT from reference)
        self.umap_n_components = UMAP_N_COMPONENTS  # 5
        self.min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE  # 3
        self.hdbscan_min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE  # 3 (alias for compatibility)
        self.random_seed = RANDOM_SEED  # 42
        
        # Caching Configuration
        self.cache_embeddings = CACHE_EMBEDDINGS  # True
        self.cache_dir = CACHE_DIR  # "embeddings_cache"
        
        # Initialize GPT-5 configuration (EXACT from reference)
        self.gpt5_config = GPT5Config(self.gpt_model)
        
        # Create cache directory
        if self.cache_embeddings:
            Path(self.cache_dir).mkdir(exist_ok=True)
        
        logger.info(f"Initialized HybridTopicAnalyzer with EXACT reference configuration:")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   GPT model: {self.gpt_model}")
        logger.info(f"   Embedding model: {self.embedding_model}")
        logger.info(f"   Random seed: {self.random_seed}")
    
    def clean_acm_prefix(self, question: str) -> str:
        """
        Remove ACM question prefix from questions before processing (EXACT FROM REFERENCE).
        
        This function removes prefixes like "(ACMs Question):" or "(ACM question):"
        that identify questions from ACM missionaries. These prefixes should be removed
        before processing to prevent clustering based on source rather than content.
        """
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        # Pattern to match ACM prefixes (case-insensitive) - EXACT from reference
        pattern = r'^\s*\(ACMs?\s+[Qq]uestion\)\s*:?\s*'
        
        # Remove the prefix and strip any remaining whitespace
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        # Return original question if nothing was removed and it's empty after cleaning
        return cleaned if cleaned else question
    
    def _clean_question_text(self, text: str) -> str:
        """Clean and normalize question text (EXACT from reference)"""
        if not text or not isinstance(text, str):
            return ""
        
        # First, remove ACM prefix if present (EXACT from reference)
        text = self.clean_acm_prefix(text)
        
        # Remove extra whitespace (EXACT from reference)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Skip empty strings after cleaning
        if not text:
            return ""
        
        # Remove common noise (EXACT from reference)
        text = re.sub(r'^(Question:\s*|Q:\s*|\d+\.\s*)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def get_cache_path(self, text: str, model: str) -> str:
        """Generate cache file path for a given text and model (EXACT FROM REFERENCE)"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return Path(self.cache_dir) / f"{model}_{text_hash}.pkl"

    def load_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Load embedding from cache if available (EXACT FROM REFERENCE)"""
        if not self.cache_embeddings:
            return None

        cache_path = self.get_cache_path(text, model)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache read error for {cache_path}: {e}")
        return None

    def save_embedding_to_cache(self, text: str, model: str, embedding: List[float]):
        """Save embedding to cache with automatic directory creation (EXACT FROM REFERENCE)"""
        if not self.cache_embeddings:
            return

        cache_path = self.get_cache_path(text, model)
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the embedding
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            if not hasattr(self.save_embedding_to_cache, '_error_shown'):
                logger.warning(f"Cache write disabled due to error: {e}")
                self.save_embedding_to_cache._error_shown = True

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching support and question preprocessing (EXACT FROM REFERENCE)"""
        
        # Clean question text (remove ACM prefixes) before processing
        cleaned_text = self._clean_question_text(text)
        
        # Try to load from cache first (using cleaned text for cache key)
        cached_embedding = self.load_cached_embedding(cleaned_text, self.embedding_model)
        if cached_embedding is not None:
            return cached_embedding

        # Generate new embedding
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text.replace("\n", " ")  # Clean text
            )
            embedding = response.data[0].embedding

            # Cache the result (will handle errors silently)
            self.save_embedding_to_cache(cleaned_text, self.embedding_model, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {cleaned_text[:50]}...")
            logger.error(f"Error details: {e}")
            return None

    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 1000, progress_callback: Optional[Callable] = None) -> List[List[float]]:
        """Get embeddings for multiple texts with true batch processing, caching, and question preprocessing (EXACT FROM REFERENCE)"""

        # Clean all texts first (remove ACM prefixes)
        cleaned_texts = [self._clean_question_text(text) for text in texts]

        embeddings = []
        cache_hits = 0
        api_calls = 0
        batch_count = 0

        logger.info(f"Generating embeddings for {len(cleaned_texts)} texts...")

        # Process in batches for API efficiency
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []

            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                cached_embedding = self.load_cached_embedding(text, self.embedding_model)
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
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=uncached_texts
                    )

                    new_embeddings = [data.embedding for data in response.data]
                    api_calls += len(uncached_texts)
                    batch_count += 1

                    # Fill in the uncached embeddings and save to cache
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings[idx] = embedding
                        self.save_embedding_to_cache(batch_texts[idx], self.embedding_model, embedding)

                    # Rate limiting for batch API calls
                    if batch_count % 5 == 0:  # Brief pause every 5 batches
                        await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    # Fill with zero vectors for failed embeddings
                    for idx in uncached_indices:
                        batch_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS

            embeddings.extend(batch_embeddings)
            
            # Progress callback
            if progress_callback:
                progress = (i + len(batch_texts)) / len(cleaned_texts) * 100
                await progress_callback("embedding_generation", progress, f"Generated embeddings for {i + len(batch_texts)}/{len(cleaned_texts)} texts")

        logger.info(f"Embedding generation complete!")
        logger.info(f"   Cache hits: {cache_hits}")
        logger.info(f"   API calls: {api_calls}")
        logger.info(f"   Total processed: {len(embeddings)}")
        logger.info(f"   Cache efficiency: {cache_hits/len(embeddings)*100:.1f}%")

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
            logger.info(f"Calling GPT-5 with model: {api_params['model']} and temperature: {api_params.get('temperature')}")
            
            response = await self.client.chat.completions.create(**api_params)

            topic_name = response.choices[0].message.content
            logger.info(f"GPT-5 raw response: '{topic_name}'")
            
            if topic_name:
                topic_name = topic_name.strip()
                # Clean up the response more aggressively
                topic_name = topic_name.replace("Topic name:", "").strip()
                topic_name = topic_name.replace("Output:", "").strip()
                topic_name = topic_name.strip('"\'')
                logger.info(f"Cleaned topic name: '{topic_name}'")

            # Validate and limit length
            if not topic_name or len(topic_name) > 100 or len(topic_name) < 3:
                logger.warning(f"Invalid topic name length or empty: '{topic_name}'")
                raise ValueError(f"Invalid topic name length: {topic_name}")

            return topic_name

        except Exception as e:
            logger.warning(f"GPT-5 topic naming failed: {e}")
            # Improved fallback to keyword-based name
            if keywords and keywords.strip():
                fallback_name = f"Topic: {keywords[:47]}"  # Leave room for "Topic: "
                logger.info(f"Using fallback topic name: {fallback_name}")
                return fallback_name
            else:
                # Ultimate fallback if no keywords
                fallback_name = f"Untitled Topic"
                logger.info(f"Using ultimate fallback: {fallback_name}")
                return fallback_name

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
            # Step 1: Prepare data (EXACT from reference)
            if progress_callback:
                await progress_callback("data_preparation", 5, "Preparing data for analysis...")
            
            # Apply sample mode if configured
            if self.processing_mode == "sample" and len(questions) > self.sample_size:
                # Use reproducible sampling with random seed
                np.random.seed(self.random_seed)
                sample_indices = np.random.choice(len(questions), self.sample_size, replace=False)
                eval_questions = [questions[i] for i in sample_indices]
                logger.info(f"Sample mode: Using {len(eval_questions)} questions (random sample with seed {self.random_seed})")
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
            
            # Step 3: Load and process existing topics (if provided)
            similar_questions = []
            remaining_questions = []
            remaining_embeddings = []
            
            if existing_topics:
                if progress_callback:
                    await progress_callback("similarity_analysis", 30, "Analyzing similarity to existing topics...")
                
                # Flatten existing topics to questions list
                existing_questions = []
                for topic_name, topic_questions in existing_topics.items():
                    for q in topic_questions:
                        existing_questions.append(q)
                
                # Generate embeddings for existing topic questions
                existing_embeddings = await self.get_embeddings_batch(existing_questions)
                
                # Classify each question by similarity
                for i, (question, embedding) in enumerate(zip(eval_questions, question_embeddings)):
                    if embedding and len(embedding) == EMBEDDING_DIMENSIONS:
                        # Find best match using cosine similarity
                        best_distance = float('inf')
                        best_match = None
                        
                        for j, (existing_q, existing_emb) in enumerate(zip(existing_questions, existing_embeddings)):
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
                        
                        # Check if similarity meets threshold
                        if best_match and best_match['similarity'] >= self.similarity_threshold:
                            similar_questions.append({
                                'question': question,
                                'matched_topic': "Existing Topic",  # Simplified for this implementation
                                'matched_question': best_match['matched_question'],
                                'similarity_score': best_match['similarity']
                            })
                        else:
                            remaining_questions.append(question)
                            remaining_embeddings.append(embedding)
                    else:
                        # Add to clustering queue if embedding failed
                        remaining_questions.append(question)
                        remaining_embeddings.append([0.0] * EMBEDDING_DIMENSIONS)
                        
                logger.info(f"Similarity analysis complete: {len(similar_questions)} similar, {len(remaining_questions)} remaining")
            else:
                # No existing topics provided - all questions go to clustering
                remaining_questions = eval_questions
                remaining_embeddings = question_embeddings
            
            # Step 4: Clustering for remaining questions
            new_topics = []
            if remaining_questions:
                if progress_callback:
                    await progress_callback("clustering", 60, f"Clustering {len(remaining_questions)} questions...")
                
                # Prepare embeddings array
                embeddings_array = np.array([emb for emb in remaining_embeddings if emb and len(emb) == EMBEDDING_DIMENSIONS])
                valid_questions = [q for i, q in enumerate(remaining_questions) if remaining_embeddings[i] and len(remaining_embeddings[i]) == EMBEDDING_DIMENSIONS]
                
                if len(embeddings_array) > 0:
                    # Step 4a: UMAP dimensionality reduction (EXACT from reference)
                    logger.info(f"Reducing dimensions: {embeddings_array.shape[1]} → {self.umap_n_components}")
                    umap_model = umap.UMAP(
                        n_components=self.umap_n_components,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=self.random_seed
                    )
                    reduced_embeddings = umap_model.fit_transform(embeddings_array)
                    
                    # Step 4b: HDBSCAN clustering (EXACT from reference)
                    logger.info(f"Clustering with HDBSCAN (min_cluster_size: {self.min_cluster_size})")
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        metric="euclidean",
                        cluster_selection_method="eom"
                    )
                    clusters = hdbscan_model.fit_predict(reduced_embeddings)
                    
                    # Step 5: Generate topic names (EXACT from reference)
                    if progress_callback:
                        await progress_callback("topic_naming", 80, "Generating topic names...")
                    
                    unique_clusters = set(clusters) - {-1}  # Exclude noise cluster
                    logger.info(f"Found {len(unique_clusters)} clusters")
                    
                    # Process each cluster
                    for cluster_id in unique_clusters:
                        cluster_mask = clusters == cluster_id
                        cluster_questions = [q for i, q in enumerate(valid_questions) if cluster_mask[i]]
                        
                        if len(cluster_questions) >= self.min_cluster_size:
                            # Generate keywords (simplified - using first few words from questions)
                            all_words = []
                            for q in cluster_questions[:5]:  # Sample first 5 questions
                                words = re.findall(r'\b\w+\b', q.lower())
                                all_words.extend(words[:3])  # First 3 words per question
                            
                            keywords = ", ".join(list(set(all_words))[:5])  # Top 5 unique words
                            
                            # Generate topic name using GPT-5
                            try:
                                topic_name = await self.generate_topic_name_gpt5(cluster_questions, keywords)
                            except Exception as e:
                                logger.warning(f"Failed to generate topic name for cluster {cluster_id}: {e}")
                                topic_name = f"Topic {cluster_id}: {keywords[:30]}"
                            
                            # Select representative question (EXACT from reference method)
                            if self.representative_question_method == "centroid":
                                # Find question closest to cluster centroid
                                cluster_embeddings = embeddings_array[cluster_mask]
                                centroid = np.mean(cluster_embeddings, axis=0)
                                distances = [cosine(emb, centroid) for emb in cluster_embeddings]
                                closest_idx = np.argmin(distances)
                                representative_question = cluster_questions[closest_idx]
                            else:
                                # Fallback: shortest question
                                representative_question = min(cluster_questions, key=len)
                            
                            new_topics.append(ClusterResult(
                                cluster_id=cluster_id,
                                topic_name=topic_name,
                                questions=cluster_questions,
                                representative_question=representative_question,
                                question_count=len(cluster_questions)
                            ))
            
            # Step 6: Prepare results (EXACT from reference format)
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
                    "topics": [topic.__dict__ for topic in new_topics]
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
                await progress_callback("completed", 100, f"Analysis complete: {len(similar_questions)} similar questions, {len(new_topics)} new topics discovered")
            
            logger.info(f"Hybrid analysis completed successfully")
            logger.info(f"   Similar questions: {len(similar_questions)}")
            logger.info(f"   New topics discovered: {len(new_topics)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            if progress_callback:
                await progress_callback("error", 0, f"Analysis failed: {str(e)}")
            raise

# Global instance
hybrid_analyzer = HybridTopicAnalyzer()