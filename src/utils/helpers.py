import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Clean and preprocess text
    
    Args:
        text: Input text string
        use_stemming: Whether to apply stemming
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'\d+', '', text)
    
    text = ' '.join(text.split())
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def format_recommendations(recommendations: list) -> str:
    """Format recommendations for display"""
    if not recommendations:
        return "\ No recommendations found.\n"
    
    output = "\n" + "="*60 + "\n"
    output += f"TOP {len(recommendations)} RECOMMENDATIONS\n"
    output += "="*60 + "\n"
    
    for i, rec in enumerate(recommendations, 1):
        output += f"\n{i}. {rec['title']}"
        output += f"\n   Similarity: {rec['similarity_score']:.3f}"
        output += f"\n   Genre: {rec['genre']}"
        output += f"\n   Rating: {rec['rating']}"
        output += f"\n   Description: {rec['description']}\n"
    
    return output