"""
Core hybrid topic analysis implementation based on the reference algorithm.
Maintains exact settings and configuration from hybrid_topic_discovery_and_classification.py
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import re
from pathlib import Path
import hashlib

from openai import AsyncOpenAI
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import umap
import hdbscan

from app.core.config import get_settings
from app.core.database import get_db

logger = logging.getLogger(__name__)

class HybridTopicAnalyzer:
    """
    Hybrid topic analysis implementation with exact configuration from reference.
    Combines similarity-based classification with clustering-based topic discovery.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        
        # Exact configuration from reference implementation
        self.similarity_threshold = 0.70
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536
        self.gpt_model = "gpt-5-nano"
        self.umap_n_components = 5
        self.hdbscan_min_cluster_size = 3
        self.random_seed = 42
        self.representative_question_method = "centroid"
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        
        logger.info(f"HybridTopicAnalyzer initialized with exact reference settings:")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  Embedding model: {self.embedding_model}")
        logger.info(f"  GPT model: {self.gpt_model}")
        logger.info(f"  UMAP components: {self.umap_n_components}")
        logger.info(f"  HDBSCAN min cluster size: {self.hdbscan_min_cluster_size}")
        logger.info(f"  Random seed: {self.random_seed}")

    def clean_question(self, question: str) -> str:
        """
        Remove ACM question prefix from questions before processing.
        Exact implementation from reference.
        """
        if not isinstance(question, str):
            return str(question) if question is not None else ""
        
        # Pattern to match ACM prefixes (case-insensitive)
        pattern = r'^\\s*\\(ACMs?\\s+[Qq]uestion\\)\\s*:?\\s*'
        
        # Remove the prefix and strip whitespace
        cleaned = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        return cleaned if cleaned else question

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using OpenAI API.
        """
        try:
            cleaned_text = self.clean_question(text)
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text.replace("\\n", " ")
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text[:50]}...")
            logger.error(f"Error details: {e}")
            return None

    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 1000,
        progress_callback=None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.
        """
        cleaned_texts = [self.clean_question(text) for text in texts]
        embeddings = []
        
        logger.info(f"Generating embeddings for {len(cleaned_texts)} texts...")
        
        if progress_callback:
            await progress_callback("embedding_generation", 0, "Starting embedding generation...")
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Progress update
                progress = min(100, (i + len(batch_texts)) / len(cleaned_texts) * 100)
                if progress_callback:
                    await progress_callback(
                        "embedding_generation", 
                        progress, 
                        f"Generated embeddings for {i + len(batch_texts)}/{len(cleaned_texts)} questions"
                    )
                
                # Rate limiting
                if i > 0 and i % (batch_size * 5) == 0:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Fill with zero vectors for failed embeddings
                batch_embeddings = [[0.0] * self.embedding_dimensions] * len(batch_texts)
                embeddings.extend(batch_embeddings)
        
        logger.info(f"Embedding generation complete! Generated {len(embeddings)} embeddings")
        return embeddings

    def find_best_topic_match(
        self, 
        question_embedding: List[float], 
        topic_embeddings: List[Dict]
    ) -> Optional[Dict]:
        """
        Find the best matching existing topic using cosine similarity.
        """
        if not topic_embeddings:
            return None
            
        best_match = None
        best_similarity = -1
        
        for topic_data in topic_embeddings:
            topic_embedding = topic_data['embedding']
            
            # Calculate cosine similarity
            similarity = 1 - cosine(question_embedding, topic_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'topic_id': topic_data['topic_id'],
                    'topic_name': topic_data['topic_name'],
                    'similarity_score': similarity,
                    'representative_question': topic_data['representative_question']
                }
        
        return best_match

    async def classify_by_similarity(
        self, 
        questions: List[str],
        topic_embeddings: List[Dict],
        progress_callback=None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify questions by similarity to existing topics.
        Returns (similar_questions, remaining_questions)
        """
        logger.info(f"Starting similarity-based classification with threshold {self.similarity_threshold}")
        
        if progress_callback:
            await progress_callback("similarity_classification", 0, "Starting similarity classification...")
        
        # Generate embeddings for all questions
        question_embeddings = await self.get_embeddings_batch(questions, progress_callback=progress_callback)
        
        similar_questions = []
        remaining_questions = []
        
        total_questions = len(questions)
        
        for i, (question, embedding) in enumerate(zip(questions, question_embeddings)):
            if embedding is None:
                # Skip questions with failed embeddings
                continue
                
            # Find best topic match
            best_match = self.find_best_topic_match(embedding, topic_embeddings)
            
            if best_match and best_match['similarity_score'] >= self.similarity_threshold:
                similar_questions.append({
                    'question': question,
                    'embedding': embedding,
                    'matched_topic_id': best_match['topic_id'],
                    'matched_topic_name': best_match['topic_name'],
                    'similarity_score': best_match['similarity_score'],
                    'representative_question': best_match['representative_question']
                })
            else:
                remaining_questions.append({
                    'question': question,
                    'embedding': embedding
                })
            
            # Progress update
            if progress_callback and (i + 1) % 100 == 0:
                progress = (i + 1) / total_questions * 100
                await progress_callback(
                    "similarity_classification",
                    progress,
                    f"Classified {i + 1}/{total_questions} questions"
                )
        
        similar_count = len(similar_questions)
        remaining_count = len(remaining_questions)
        
        logger.info(f"Similarity classification complete:")
        logger.info(f"  Similar to existing topics (≥{self.similarity_threshold}): {similar_count} ({similar_count/total_questions*100:.1f}%)")
        logger.info(f"  Remaining for clustering (<{self.similarity_threshold}): {remaining_count} ({remaining_count/total_questions*100:.1f}%)")
        
        return similar_questions, remaining_questions

    async def perform_clustering_analysis(
        self, 
        remaining_questions: List[Dict],
        progress_callback=None
    ) -> Optional[List[Dict]]:
        """
        Perform clustering analysis on questions that didn't match existing topics.
        """
        if not remaining_questions:
            logger.info("No questions remaining for clustering - all matched existing topics!")
            return None
            
        logger.info(f"Starting clustering-based topic discovery for {len(remaining_questions)} questions")
        
        if progress_callback:
            await progress_callback("clustering", 10, "Starting clustering analysis...")
        
        # Extract embeddings and questions
        questions = [q['question'] for q in remaining_questions]
        embeddings = np.array([q['embedding'] for q in remaining_questions])
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Step 1: Dimensionality reduction with UMAP
        logger.info(f"Reducing dimensions: {embeddings.shape[1]} → {self.umap_n_components}")
        
        if progress_callback:
            await progress_callback("clustering", 30, "Performing dimensionality reduction with UMAP...")
        
        umap_model = umap.UMAP(
            n_components=self.umap_n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_seed
        )
        reduced_embeddings = umap_model.fit_transform(embeddings)
        
        logger.info(f"UMAP reduction complete: {reduced_embeddings.shape}")
        
        # Step 2: Clustering with HDBSCAN
        logger.info(f"Clustering with HDBSCAN (min_cluster_size: {self.hdbscan_min_cluster_size})")
        
        if progress_callback:
            await progress_callback("clustering", 60, "Performing HDBSCAN clustering...")
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
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
        logger.info(f"  Number of clusters found: {n_clusters}")
        logger.info(f"  Questions clustered: {clustered_count} ({clustered_count/len(questions)*100:.1f}%)")
        logger.info(f"  Questions not clustered (noise): {unclustered_count} ({unclustered_count/len(questions)*100:.1f}%)")
        
        if progress_callback:
            await progress_callback("clustering", 80, f"Found {n_clusters} clusters, generating topics...")
        
        # Generate topics for each cluster
        clustered_questions = []
        
        for i, (question_data, cluster_id) in enumerate(zip(remaining_questions, clusters)):
            clustered_questions.append({
                **question_data,
                'cluster_id': int(cluster_id),
                'is_noise': cluster_id == -1
            })
        
        if progress_callback:
            await progress_callback("clustering", 100, "Clustering analysis complete")
        
        return clustered_questions

    async def generate_topic_names(
        self, 
        clustered_questions: List[Dict],
        progress_callback=None
    ) -> List[Dict]:
        """
        Generate topic names and descriptions for discovered clusters using GPT.
        """
        logger.info("Generating topic names for discovered clusters...")
        
        if progress_callback:
            await progress_callback("topic_extraction", 10, "Generating topic names...")
        
        # Group questions by cluster
        clusters = {}
        for q in clustered_questions:
            cluster_id = q['cluster_id']
            if cluster_id != -1:  # Skip noise
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(q['question'])
        
        topics = []
        total_clusters = len(clusters)
        
        for i, (cluster_id, cluster_questions) in enumerate(clusters.items()):
            try:
                # Select representative questions (up to 10)
                sample_questions = cluster_questions[:10]
                
                # Create prompt for GPT
                prompt = f"""Analyze these student questions and generate a concise topic name and description:

Questions:
{chr(10).join(f"- {q}" for q in sample_questions)}

Provide a response in this exact JSON format:
{{
    "topic_name": "Short descriptive topic name (2-4 words)",
    "description": "Brief description of what this topic covers (1-2 sentences)",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}\"\"\"
                
                response = await self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing student questions and identifying topics. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                # Parse GPT response
                try:
                    topic_data = json.loads(response.choices[0].message.content)
                    
                    topics.append({
                        'cluster_id': cluster_id,
                        'name': topic_data.get('topic_name', f'Topic {cluster_id}'),
                        'description': topic_data.get('description', 'No description available'),
                        'keywords': topic_data.get('keywords', []),
                        'question_count': len(cluster_questions),
                        'representative_questions': sample_questions[:5],
                        'confidence_score': 0.8  # Default confidence
                    })
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse GPT response for cluster {cluster_id}")
                    topics.append({
                        'cluster_id': cluster_id,
                        'name': f'Topic {cluster_id}',
                        'description': 'Auto-discovered topic',
                        'keywords': [],
                        'question_count': len(cluster_questions),
                        'representative_questions': sample_questions[:5],
                        'confidence_score': 0.6
                    })
                
                # Progress update
                if progress_callback:
                    progress = (i + 1) / total_clusters * 100
                    await progress_callback(
                        "topic_extraction",
                        progress,
                        f"Generated topic {i + 1}/{total_clusters}"
                    )
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating topic for cluster {cluster_id}: {e}")
                topics.append({
                    'cluster_id': cluster_id,
                    'name': f'Topic {cluster_id}',
                    'description': 'Error generating description',
                    'keywords': [],
                    'question_count': len(cluster_questions),
                    'representative_questions': sample_questions[:5],
                    'confidence_score': 0.5
                })
        
        logger.info(f"Generated {len(topics)} topic names")
        return topics

    async def run_hybrid_analysis(
        self, 
        questions: List[str],
        existing_topics: List[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run the complete hybrid topic analysis pipeline.
        """
        logger.info(f"Starting hybrid topic analysis for {len(questions)} questions")
        
        if progress_callback:
            await progress_callback("initialization", 0, "Starting hybrid analysis...")
        
        # Use existing topics if provided, otherwise empty list
        topic_embeddings = existing_topics or []
        
        # Step 1: Similarity-based classification
        similar_questions, remaining_questions = await self.classify_by_similarity(
            questions, topic_embeddings, progress_callback
        )
        
        # Step 2: Clustering-based topic discovery
        clustered_questions = None
        new_topics = []
        
        if remaining_questions:
            clustered_questions = await self.perform_clustering_analysis(
                remaining_questions, progress_callback
            )
            
            if clustered_questions:
                # Step 3: Generate topic names
                new_topics = await self.generate_topic_names(
                    clustered_questions, progress_callback
                )
        
        if progress_callback:
            await progress_callback("finalization", 100, "Analysis complete!")
        
        # Compile results
        results = {
            'summary': {
                'total_questions': len(questions),
                'similar_to_existing': len(similar_questions),
                'new_topics_discovered': len(new_topics),
                'unclustered_questions': len([q for q in (clustered_questions or []) if q['is_noise']]),
                'similarity_threshold': self.similarity_threshold
            },
            'similar_questions': similar_questions,
            'new_topics': new_topics,
            'clustered_questions': clustered_questions or [],
            'settings': {
                'similarity_threshold': self.similarity_threshold,
                'embedding_model': self.embedding_model,
                'gpt_model': self.gpt_model,
                'umap_n_components': self.umap_n_components,
                'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
                'random_seed': self.random_seed
            }
        }
        
        logger.info(f"Hybrid analysis complete! Found {len(new_topics)} new topics")
        return results