# 🎬 Movie Recommendation System

A content-based movie recommendation system using TF-IDF and cosine similarity.

## 📋 Overview

This system recommends movies based on plot descriptions using:
- **TF-IDF Vectorization** for text feature extraction
- **Cosine Similarity** for finding similar movies
- **NLP Preprocessing** (lowercase, punctuation removal, stopwords, stemming)

## 🔍 Explanation of Approach

### 1. Data Preprocessing
The system uses the IMDB Genres dataset from Hugging Face, containing over 238,000 movies with plot descriptions. The preprocessing pipeline includes:

- **Text Cleaning**: Converts text to lowercase, removes punctuation and numbers
- **Stopword Removal**: Removes common English words (the, a, an, etc.) that don't carry meaning
- **Stemming**: Reduces words to their root form using PorterStemmer (e.g., "running" → "run", "movies" → "movi")
- **Empty Description Filtering**: Removes movies without valid descriptions

### 2. Feature Extraction with TF-IDF
The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text descriptions into numerical vectors:

- **Term Frequency (TF)**: Measures how frequently a word appears in a document
- **Inverse Document Frequency (IDF)**: Reduces the weight of common words that appear across many documents
- **N-grams**: Uses both single words and word pairs (1-2 grams) to capture context
- **Feature Limit**: Limits to 5,000 most important features for memory efficiency

**Why TF-IDF instead of Bag-of-Words?**
- TF-IDF gives higher weight to important words while downweighting common words
- Better captures the uniqueness and relevance of terms in descriptions
- More accurate similarity measurements between movies

### 3. Similarity Computation
The system uses **Cosine Similarity** to measure how similar movies are:

- **Formula**: `cosine_similarity(A, B) = (A·B) / (||A|| × ||B||)`
- **Range**: 0 (completely different) to 1 (identical)
- **Memory Optimization**: Computes similarities on-the-fly using batch processing instead of storing a full similarity matrix

### 4. Recommendation Logic
The system provides three ways to get recommendations:

1. **By Movie Title**: Finds movies with similar plots to the specified title
2. **By Description**: Takes a vague description and finds movies with similar plots
3. **By Genre**: Filters movies by genre and returns random recommendations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt

```

##  Screenshots

**System Initialization**

![](image1.png)

**Genre Search Results**

![](image2.png)

**Title Filtering**

![](image3.png)

**Description Search**

![](image4.png)