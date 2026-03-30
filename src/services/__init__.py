from typing import Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.vectorizer import MovieVectorizer
from src.models.recommender import MovieRecommender

class RecommendationService:
    """Main service for movie recommendations"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.vectorizer = MovieVectorizer()
        self.recommender = MovieRecommender()
        self.is_ready = False
    
    def initialize(self):
        """Initialize the recommendation system"""
        print("\n" + "="*50)
        print("INITIALIZING MOVIE RECOMMENDATION SYSTEM")
        print("="*50)
        
        df = self.data_loader.load_data('train')
        
        processed_df = self.preprocessor.preprocess(df)
        
        vectors = self.vectorizer.fit_transform(processed_df['cleaned_description'].tolist())
        
        self.recommender.fit(processed_df, vectors)
        
        self.is_ready = True
        print("\nSystem ready! You can now get recommendations.")
        
        return self
    
    def get_recommendations(self, 
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           genre: Optional[str] = None,
                           top_k: int = 5) -> list[dict[str, Any]]:
        """Get recommendations based on input"""
        
        if not self.is_ready:
            raise ValueError("System not initialized. Call initialize() first.")
        
        df = self.recommender.movies_df
        if genre:
            df = self.preprocessor.filter_by_genre(df, genre)
            if len(df) == 0:
                raise ValueError(f"No movies found for genre '{genre}'")
            
            vectors = self.vectorizer.transform(df['cleaned_description'].tolist())
            self.recommender.fit(df, vectors)
        
        if title:
            return self.recommender.recommend_by_title(title, top_k)
        elif description:
            from src.utils.helpers import clean_text
            cleaned_desc = clean_text(description)
            return self.recommender.recommend_by_description(cleaned_desc, self.vectorizer.tfidf, top_k)
        else:
            raise ValueError("Please provide either title or description")