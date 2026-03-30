import pandas as pd

class DataLoader:
    """Handle data loading from HuggingFace"""
    
    def __init__(self):
        self.dataset_name = "jquigl/imdb-genres"
    
    def load_data(self, split: str = "train") -> pd.DataFrame:
        """Load dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            print(f"Loading {split} data from {self.dataset_name}...")
            dataset = load_dataset(self.dataset_name)
            df = dataset[split].to_pandas()
            print(f"Loaded {len(df)} movies")
            
            # Print column names for debugging
            print(f"Available columns: {list(df.columns)}")
            
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise