import pandas as pd
from typing import Optional, List
from src.utils.helpers import clean_text

class DataPreprocessor:
    """Handle text cleaning and preprocessing"""
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming
        self.title_col = None
        self.description_col = None
        self.genre_col = None
        self.rating_col = None
    
    def _detect_columns(self, df: pd.DataFrame):
        """Detect the correct column names"""
        title_candidates = ['title', 'movie_title', 'name', 'primary_title', 'movie title - year']
        for col in title_candidates:
            if col in df.columns:
                self.title_col = col
                break
        
        desc_candidates = ['description', 'plot', 'summary', 'overview']
        for col in desc_candidates:
            if col in df.columns:
                self.description_col = col
                break
        
        genre_candidates = ['genre', 'genres', 'expanded-genres']
        for col in genre_candidates:
            if col in df.columns:
                self.genre_col = col
                break
        
        rating_candidates = ['ratings', 'rating', 'average_rating', 'vote_average']
        for col in rating_candidates:
            if col in df.columns:
                self.rating_col = col
                break
        
        print(f"Detected columns:")
        print(f"  Title: {self.title_col}")
        print(f"  Description: {self.description_col}")
        print(f"  Genre: {self.genre_col}")
        print(f"  Rating: {self.rating_col}")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        print("Preprocessing data...")
        
        self._detect_columns(df)
        
        df = df.copy()
        
        if self.title_col and self.title_col != 'title':
            if ' - ' in df[self.title_col].iloc[0]:
                df['title'] = df[self.title_col].apply(lambda x: x.split(' - ')[0] if ' - ' in str(x) else x)
            else:
                df['title'] = df[self.title_col]
        
        if self.description_col and self.description_col != 'description':
            df['description'] = df[self.description_col]
        
        if self.genre_col and self.genre_col != 'genre':
            df['genre'] = df[self.genre_col]
        
        if self.rating_col and self.rating_col != 'ratings':
            df['ratings'] = df[self.rating_col]
        
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('')
        else:
            raise ValueError("No description column found in dataset!")
        
        df['cleaned_description'] = df['description'].apply(
            lambda x: clean_text(x, use_stemming=self.use_stemming)
        )
        
        initial_count = len(df)
        df = df[df['cleaned_description'].str.len() > 0]
        print(f"Removed {initial_count - len(df)} movies with empty descriptions")
        print(f"Final dataset size: {len(df)} movies")
        
        return df
    
    def filter_by_genre(self, df: pd.DataFrame, genre: str) -> pd.DataFrame:
        """Filter movies by genre (case-insensitive partial match)"""
        if 'genre' in df.columns:
            mask = df['genre'].astype(str).str.lower().str.contains(genre.lower(), na=False)
            return df[mask]
        return df
    
    def search_by_title(self, df: pd.DataFrame, title: str) -> Optional[pd.Series]:
        """Find movie by title (case-insensitive partial match)"""
        if 'title' in df.columns:
            matches = df[df['title'].str.lower() == title.lower()]
            if len(matches) > 0:
                return matches.iloc[0]
            
            matches = df[df['title'].str.lower().str.contains(title.lower(), na=False)]
            if len(matches) > 0:
                print(f"Found {len(matches)} movies matching '{title}'. Using: {matches.iloc[0]['title']}")
                return matches.iloc[0]
        
        return None