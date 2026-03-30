from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.vectorizer import MovieVectorizer
from src.models.recommender import MovieRecommender
from src.utils.helpers import clean_text
from typing import List, Dict, Any, Optional

class RecommendationService:
    """Main service for movie recommendations"""
    
    def __init__(self, sample_size: int = None):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.vectorizer = MovieVectorizer()
        self.recommender = MovieRecommender()
        self.sample_size = sample_size
        self.is_ready = False
        self.original_df = None
    
    def initialize(self):
        """Initialize the recommendation system"""
        print("\n" + "="*50)
        print("INITIALIZING MOVIE RECOMMENDATION SYSTEM")
        print("="*50)
        
        df = self.data_loader.load_data('train')
        self.original_df = df.copy()
        
        if self.sample_size and self.sample_size < len(df):
            print(f"Sampling {self.sample_size} movies from {len(df)}...")
            df = df.sample(n=self.sample_size, random_state=42)
        
        processed_df = self.preprocessor.preprocess(df)
        
        if 'title' in processed_df.columns:
            print("\ Sample movies in dataset (try these titles):")
            sample_titles = processed_df['title'].head(15).tolist()
            for i, title in enumerate(sample_titles, 1):
                genre = processed_df.iloc[i-1].get('genre', 'N/A')
                print(f"   {i:2}. {title[:50]:50} ({genre})")
        else:
            print("\n No title column found")
        
        vectors = self.vectorizer.fit_transform(processed_df['cleaned_description'].tolist())
        
        self.recommender.fit(processed_df, vectors)
        
        self.is_ready = True
        print("\nSystem ready! You can now get recommendations.")
        print(f" Using {len(processed_df)} movies")
        
        return self
    
    def get_recommendations(self, 
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           genre: Optional[str] = None,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations based on input"""
        
        if not self.is_ready:
            raise ValueError("System not initialized. Call initialize() first.")
        
        if genre:
            if 'genre' in self.recommender.movies_df.columns:
                filtered_df = self.preprocessor.filter_by_genre(self.recommender.movies_df, genre)
                if len(filtered_df) == 0:
                    # Show available genres
                    available_genres = set()
                    for g in self.recommender.movies_df['genre'].dropna().head(200):
                        if isinstance(g, str):
                            for genre_item in g.split(','):
                                available_genres.add(genre_item.strip())
                    print(f"\n Available genres: {', '.join(sorted(list(available_genres))[:15])}")
                    raise ValueError(f"No movies found for genre '{genre}'")
                
                print(f"Found {len(filtered_df)} movies in genre '{genre}'")
                
                # Get vectors for filtered movies
                filtered_vectors = self.vectorizer.transform(filtered_df['cleaned_description'].tolist())
                
                # Create a temporary recommender for this genre
                temp_recommender = MovieRecommender()
                temp_recommender.fit(filtered_df, filtered_vectors)
                
                # Return random recommendations from this genre
                return temp_recommender.recommend_random(top_k)
            else:
                raise ValueError("Genre filtering not available")
        
        if title:
            if 'title' not in self.recommender.movies_df.columns:
                raise ValueError("Title search not available")
            return self.recommender.recommend_by_title(title, top_k)
        elif description:
            cleaned_desc = clean_text(description, use_stemming=True)
            return self.recommender.recommend_by_description(cleaned_desc, self.vectorizer.tfidf, top_k)
        else:
            raise ValueError("Please provide either title, description, or genre")