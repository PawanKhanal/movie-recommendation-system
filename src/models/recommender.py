"""Recommendation model module with memory optimization"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import heapq

class MovieRecommender:
    """Content-based recommendation model with memory optimization"""
    
    def __init__(self):
        self.movies_df = None
        self.vectors = None
        self.use_incremental = True
    
    def fit(self, movies_df: pd.DataFrame, vectors):
        """Fit the model with movie data and vectors"""
        self.movies_df = movies_df.reset_index(drop=True)
        self.vectors = vectors
        
        print(f"Model fitted with {len(movies_df)} movies")
    
    def recommend_by_index(self, idx: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations by movie index"""
        # Get the vector for the target movie
        target_vector = self.vectors[idx]
        
        # Handle sparse matrix
        if hasattr(target_vector, 'toarray'):
            target_vector = target_vector.toarray()
        
        target_vector = target_vector.reshape(1, -1)
        
        # Compute similarities incrementally
        batch_size = 5000
        similarities = []
        
        for i in range(0, self.vectors.shape[0], batch_size):
            batch = self.vectors[i:min(i + batch_size, self.vectors.shape[0])]
            
            # Handle sparse matrix
            if hasattr(batch, 'toarray'):
                batch = batch.toarray()
            if hasattr(target_vector, 'toarray'):
                target_vector = target_vector.toarray()
            
            batch_similarities = cosine_similarity(target_vector, batch)[0]
            similarities.extend(batch_similarities)
        
        similarities = np.array(similarities)
        
        # Get top-k similar movies (excluding self)
        if idx < len(similarities):
            similarities[idx] = -1
        
        top_k = min(top_k, len(similarities))
        top_indices = heapq.nlargest(top_k, range(len(similarities)), similarities.take)
        top_scores = similarities[top_indices]
        
        return self._format_recommendations(top_indices, top_scores)
    
    def recommend_by_title(self, title: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations by movie title with partial matching"""
        if 'title' not in self.movies_df.columns:
            raise ValueError("No title column found in dataset")
        
        # Clean the title for comparison
        title_clean = title.lower().strip()
        
        # Try exact match
        matches = self.movies_df[self.movies_df['title'].str.lower().str.strip() == title_clean]
        
        # If no exact match, try partial match
        if len(matches) == 0:
            matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(title_clean, na=False)]
            if len(matches) > 0:
                print(f"Found {len(matches)} movies matching '{title}'. Using: {matches.iloc[0]['title']}")
            else:
                # Try searching in the original column
                original_col = 'movie title - year'
                if original_col in self.movies_df.columns:
                    matches = self.movies_df[self.movies_df[original_col].str.lower().str.contains(title_clean, na=False)]
                    if len(matches) > 0:
                        print(f"Found in original titles: {matches.iloc[0][original_col]}")
        
        if len(matches) == 0:
            # Show suggestions
            print(f"\nMovie '{title}' not found. Here are some similar movies:")
            suggestions = self.movies_df[self.movies_df['title'].str.lower().str.startswith(title_clean[:3], na=False)].head(5)
            if len(suggestions) == 0:
                suggestions = self.movies_df.head(5)
            
            for i, row in suggestions.iterrows():
                print(f"  - {row['title']} ({row.get('genre', 'N/A')})")
            raise ValueError(f"Movie '{title}' not found")
        
        idx = matches.index[0]
        return self.recommend_by_index(idx, top_k)
    
    def recommend_by_description(self, description: str, vectorizer, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations by vague description"""
        desc_vector = vectorizer.transform([description])
        
        # Compute similarities incrementally
        batch_size = 5000
        similarities = []
        
        for i in range(0, self.vectors.shape[0], batch_size):
            batch = self.vectors[i:min(i + batch_size, self.vectors.shape[0])]
            
            # Handle sparse matrix
            if hasattr(batch, 'toarray'):
                batch = batch.toarray()
            if hasattr(desc_vector, 'toarray'):
                desc_vector = desc_vector.toarray()
            
            batch_similarities = cosine_similarity(desc_vector, batch)[0]
            similarities.extend(batch_similarities)
        
        similarities = np.array(similarities)
        
        # Get top-k similar movies
        top_k = min(top_k, len(similarities))
        top_indices = heapq.nlargest(top_k, range(len(similarities)), similarities.take)
        top_scores = similarities[top_indices]
        
        return self._format_recommendations(top_indices, top_scores)
    
    def recommend_random(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get random recommendations"""
        indices = np.random.choice(len(self.movies_df), top_k, replace=False)
        scores = [0.5] * top_k
        return self._format_recommendations(indices, scores)
    
    def _format_recommendations(self, indices, scores) -> List[Dict[str, Any]]:
        """Format recommendations as list of dictionaries"""
        recommendations = []
        
        for idx, score in zip(indices, scores):
            movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': movie.get('title', movie.get('movie title - year', 'N/A')),
                'similarity_score': float(score) if score >= 0 else 0,
                'genre': movie.get('genre', 'N/A'),
                'rating': movie.get('ratings', movie.get('rating', 'N/A')),
                'description': movie.get('description', '')[:150] + '...'
            })
        
        return recommendations