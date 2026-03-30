import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

class MovieVectorizer:
    """Handle TF-IDF vectorization with sparse matrix support"""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.tfidf = None
        self.vectors = None
    
    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        """Fit and transform texts to sparse vectors"""
        print(f"Vectorizing {len(texts)} documents...")
        
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            dtype=np.float32  
        )
        
        self.vectors = self.tfidf.fit_transform(texts)
        print(f"Vectorized to {self.vectors.shape[1]} features")
        print(f"Sparse matrix memory: {self.vectors.data.nbytes / 1024**2:.2f} MB")
        
        return self.vectors
    
    def transform(self, texts: list[str]) -> sp.csr_matrix:
        """Transform new texts to sparse vectors"""
        if self.tfidf is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.tfidf.transform(texts)