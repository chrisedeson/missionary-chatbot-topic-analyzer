"""
Hybrid Topic Discovery and Classification System

This module implements the exact hybrid analysis algorithm from the reference script
with identical settings: similarity threshold 0.70, gpt-5-nano, text-embedding-3-small,
UMAP 5 components, HDBSCAN min cluster size 3, random seed 42.
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

# ML imports
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import umap
from openai import AsyncOpenAI

# App imports
from app.core.config import settings

logger = logging.getLogger(__name__)

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
    questions: List[Question]
    topic_name: str = ""
    topic_description: str = ""

class HybridTopicAnalyzer:
    """
    Hybrid Topic Discovery and Classification System
    
    Implements the exact algorithm from hybrid_topic_discovery_and_classification.py
    with identical configuration and processing steps.
    """
    
    def __init__(self):
        """Initialize with exact reference settings"""
        # OpenAI Configuration (exact from reference)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = settings.EMBEDDING_MODEL  # "text-embedding-3-small"
        self.chat_model = settings.CHAT_MODEL  # "gpt-5-nano"
        
        # Analysis Configuration (exact from reference)
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD  # 0.70
        self.representative_question_method = settings.REPRESENTATIVE_QUESTION_METHOD  # "centroid"
        self.processing_mode = settings.PROCESSING_MODE  # "sample"
        self.sample_size = settings.SAMPLE_SIZE  # 2000
        
        # Clustering Configuration (exact from reference)
        self.umap_n_components = settings.UMAP_N_COMPONENTS  # 5
        self.min_cluster_size = settings.MIN_CLUSTER_SIZE  # 3
        self.random_seed = settings.RANDOM_SEED  # 42
        
        # Caching Configuration
        self.cache_embeddings = settings.CACHE_EMBEDDINGS  # True
        self.cache_dir = settings.CACHE_DIR  # "embeddings_cache"
        
        logger.info(f"Initialized HybridTopicAnalyzer with similarity_threshold={self.similarity_threshold}")
    
    def _clean_question_text(self, text: str) -> str:
        """Clean and normalize question text (exact from reference)"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common prefixes that don't add semantic value
        prefixes_to_remove = [
            r'^(Question:\s*)',
            r'^(Q:\s*)',
            r'^(\d+\.\s*)',
            r'^(-\s*)',
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    async def _generate_embeddings(self, texts: List[str], progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Generate embeddings for text list (exact from reference)"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Clean texts
        cleaned_texts = [self._clean_question_text(text) for text in texts]
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(cleaned_texts) if text]
        
        if not valid_texts:
            logger.warning("No valid texts found for embedding generation")
            return np.array([])
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_texts = [text for _, text in batch]
            
            if progress_callback:
                progress = int((i / len(valid_texts)) * 100)
                await progress_callback("embeddings", progress, f"Generating embeddings: {i}/{len(valid_texts)}")
            
            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=settings.EMBEDDING_DIMENSIONS
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Add zero embeddings for failed batch
                zero_embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
                all_embeddings.extend([zero_embedding] * len(batch_texts))
        
        # Create final embedding matrix
        embedding_matrix = np.zeros((len(texts), settings.EMBEDDING_DIMENSIONS))
        for i, (original_idx, _) in enumerate(valid_texts):
            if i < len(all_embeddings):
                embedding_matrix[original_idx] = all_embeddings[i]
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return embedding_matrix
    
    def _classify_by_similarity(self, question_embeddings: np.ndarray, topic_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify questions by similarity to existing topics (exact from reference)"""
        logger.info(f"Classifying {len(question_embeddings)} questions against {len(topic_embeddings)} topics")
        
        if len(topic_embeddings) == 0:
            # No existing topics, all questions are new
            return np.array([-1] * len(question_embeddings)), np.array([0.0] * len(question_embeddings))
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_embeddings, topic_embeddings)
        
        # Find best matches
        best_topic_indices = np.argmax(similarities, axis=1)
        best_similarities = np.max(similarities, axis=1)
        
        # Apply similarity threshold
        classifications = np.where(
            best_similarities >= self.similarity_threshold,
            best_topic_indices,
            -1  # -1 indicates new topic needed
        )
        
        similar_count = np.sum(classifications != -1)
        logger.info(f"Classified {similar_count}/{len(question_embeddings)} questions as similar to existing topics")
        
        return classifications, best_similarities
    
    def _cluster_new_questions(self, embeddings: np.ndarray, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Cluster new questions using UMAP + HDBSCAN (exact from reference)"""
        if len(embeddings) < self.min_cluster_size:
            logger.info(f"Too few questions ({len(embeddings)}) for clustering, treating all as noise")
            return np.array([-1] * len(embeddings))
        
        logger.info(f"Clustering {len(embeddings)} new questions")
        
        if progress_callback:
            asyncio.create_task(progress_callback("clustering", 0, "Starting dimensionality reduction"))
        
        # UMAP dimensionality reduction (exact settings from reference)
        umap_reducer = umap.UMAP(
            n_components=self.umap_n_components,
            metric='cosine',
            random_state=self.random_seed,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1
        )
        
        try:
            reduced_embeddings = umap_reducer.fit_transform(embeddings)
            logger.info(f"UMAP reduced embeddings to {reduced_embeddings.shape}")
        except Exception as e:
            logger.error(f"UMAP failed: {e}")
            return np.array([-1] * len(embeddings))
        
        if progress_callback:
            asyncio.create_task(progress_callback("clustering", 50, "Performing HDBSCAN clustering"))
        
        # HDBSCAN clustering (exact settings from reference) 
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        try:
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            logger.info(f"HDBSCAN found {n_clusters} clusters with {n_noise} noise points")
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}")
            return np.array([-1] * len(embeddings))
    
    async def _generate_topic_names(self, cluster_results: List[ClusterResult], progress_callback: Optional[Callable] = None) -> List[ClusterResult]:
        """Generate topic names using GPT (exact from reference)"""
        logger.info(f"Generating names for {len(cluster_results)} clusters")
        
        for i, cluster in enumerate(cluster_results):
            if progress_callback:
                progress = int((i / len(cluster_results)) * 100)
                await progress_callback("naming", progress, f"Naming cluster {i+1}/{len(cluster_results)}")
            
            try:
                # Sample questions for topic naming
                sample_questions = [q.text for q in cluster.questions[:10]]
                
                # Create prompt for GPT
                prompt = f"""Analyze these student questions and generate a concise topic name and description:

Questions:
{chr(10).join(f"- {q}" for q in sample_questions)}

Provide a response in this exact JSON format:
{{
    "topic_name": "Short descriptive topic name (2-4 words)",
    "topic_description": "Brief description of what this topic covers (1-2 sentences)"
}}"""

                response = await self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    result = json.loads(content)
                    cluster.topic_name = result.get("topic_name", f"Topic {cluster.cluster_id}")
                    cluster.topic_description = result.get("topic_description", "")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse GPT response for cluster {cluster.cluster_id}")
                    cluster.topic_name = f"Topic {cluster.cluster_id}"
                    cluster.topic_description = ""
                    
            except Exception as e:
                logger.error(f"Error generating topic name for cluster {cluster.cluster_id}: {e}")
                cluster.topic_name = f"Topic {cluster.cluster_id}"
                cluster.topic_description = ""
        
        return cluster_results
    
    def _calculate_representative_embedding(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate representative embedding for a cluster (exact from reference)"""
        if len(embeddings) == 0:
            return np.zeros(settings.EMBEDDING_DIMENSIONS)
        
        if self.representative_question_method == "centroid":
            return np.mean(embeddings, axis=0)
        elif self.representative_question_method == "medoid":
            # Find the embedding closest to the centroid
            centroid = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            medoid_idx = np.argmin(distances)
            return embeddings[medoid_idx]
        else:
            return np.mean(embeddings, axis=0)
    
    async def analyze(
        self, 
        questions: List[Question], 
        existing_topics: List[Topic] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete hybrid topic analysis pipeline.
        
        Args:
            questions: List of Question objects to analyze
            existing_topics: List of existing Topic objects for similarity matching
            progress_callback: Async callback for progress updates
            
        Returns:
            Dictionary with analysis results including similar questions and new topics
        """
        logger.info(f"Starting hybrid topic analysis for {len(questions)} questions")
        
        if progress_callback:
            await progress_callback("initialization", 0, "Starting analysis")
        
        # Clean question texts
        cleaned_questions = []
        for q in questions:
            cleaned_text = self._clean_question_text(q.text)
            if cleaned_text:
                q.text = cleaned_text
                cleaned_questions.append(q)
        
        logger.info(f"Processing {len(cleaned_questions)} valid questions")
        
        if not cleaned_questions:
            return {
                "similar_questions": [],
                "new_topics": [],
                "total_questions": 0,
                "total_similar": 0,
                "total_new_topics": 0
            }
        
        # Generate embeddings for questions
        question_texts = [q.text for q in cleaned_questions]
        question_embeddings = await self._generate_embeddings(question_texts, progress_callback)
        
        # Update questions with embeddings
        for i, q in enumerate(cleaned_questions):
            q.embedding = question_embeddings[i].tolist()
        
        # Prepare existing topic embeddings
        existing_topic_embeddings = np.array([])
        if existing_topics:
            existing_topic_embeddings = np.array([topic.representative_embedding for topic in existing_topics])
        
        if progress_callback:
            await progress_callback("classification", 30, "Classifying questions by similarity")
        
        # Classify questions by similarity to existing topics
        classifications, similarities = self._classify_by_similarity(
            question_embeddings, 
            existing_topic_embeddings
        )
        
        # Separate similar and new questions
        similar_questions = []
        new_questions = []
        
        for i, (classification, similarity) in enumerate(zip(classifications, similarities)):
            question = cleaned_questions[i]
            
            if classification != -1:  # Similar to existing topic
                similar_questions.append({
                    "question": question,
                    "matched_topic": existing_topics[classification] if existing_topics else None,
                    "similarity_score": float(similarity)
                })
            else:  # New question
                new_questions.append(question)
        
        logger.info(f"Found {len(similar_questions)} similar questions, {len(new_questions)} new questions")
        
        # Cluster new questions
        new_topics = []
        if new_questions:
            if progress_callback:
                await progress_callback("clustering", 60, f"Clustering {len(new_questions)} new questions")
            
            new_embeddings = np.array([q.embedding for q in new_questions])
            cluster_labels = self._cluster_new_questions(new_embeddings, progress_callback)
            
            # Group questions by cluster
            clusters = {}
            for question, label in zip(new_questions, cluster_labels):
                if label == -1:  # Noise
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(question)
            
            # Create cluster results
            cluster_results = []
            for cluster_id, cluster_questions in clusters.items():
                cluster_results.append(ClusterResult(
                    cluster_id=cluster_id,
                    questions=cluster_questions
                ))
            
            if progress_callback:
                await progress_callback("naming", 80, f"Generating names for {len(cluster_results)} topics")
            
            # Generate topic names
            cluster_results = await self._generate_topic_names(cluster_results, progress_callback)
            
            # Convert to Topic objects
            for cluster in cluster_results:
                cluster_embeddings = np.array([q.embedding for q in cluster.questions])
                representative_embedding = self._calculate_representative_embedding(cluster_embeddings)
                
                # Find representative question (closest to centroid)
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                rep_idx = np.argmin(distances)
                representative_question = cluster.questions[rep_idx].text
                
                new_topic = Topic(
                    id=f"discovered_{cluster.cluster_id}_{datetime.now().isoformat()}",
                    name=cluster.topic_name,
                    description=cluster.topic_description,
                    questions=cluster.questions,
                    representative_question=representative_question,
                    representative_embedding=representative_embedding.tolist()
                )
                new_topics.append(new_topic)
        
        if progress_callback:
            await progress_callback("complete", 100, "Analysis complete")
        
        logger.info(f"Analysis complete: {len(similar_questions)} similar, {len(new_topics)} new topics")
        
        return {
            "similar_questions": similar_questions,
            "new_topics": new_topics,
            "total_questions": len(cleaned_questions),
            "total_similar": len(similar_questions),
            "total_new_topics": len(new_topics),
            "noise_questions": len([q for q in new_questions if q not in sum([t.questions for t in new_topics], [])]),
            "settings": {
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model,
                "chat_model": self.chat_model,
                "umap_components": self.umap_n_components,
                "min_cluster_size": self.min_cluster_size,
                "random_seed": self.random_seed
            }
        }