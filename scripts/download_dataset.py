"""
Dataset Downloader for MIT Plagiarism Detection Dataset (SNLI-based)
This script downloads and prepares the SNLI dataset for plagiarism detection using Hugging Face datasets.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_snli_dataset(self):
        """Download SNLI dataset using Hugging Face datasets"""
        logger.info("Downloading SNLI dataset from Hugging Face...")
        
        try:
            # Load SNLI dataset
            dataset = load_dataset("snli")
            logger.info("SNLI dataset downloaded successfully!")
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading SNLI dataset: {str(e)}")
            return None
    
    def process_snli_for_plagiarism(self, dataset_split, output_name):
        """Process SNLI data to create plagiarism detection dataset"""
        logger.info(f"Processing SNLI {output_name} split for plagiarism detection...")
        
        # Convert to pandas DataFrame
        df = dataset_split.to_pandas()
        
        # Filter and clean data
        # Keep only entailment (similar/plagiarized) and contradiction (different)
        # Map: entailment -> 1 (plagiarized), contradiction -> 0 (not plagiarized)
        label_mapping = {
            'entailment': 1,      # Similar content (plagiarized)
            'contradiction': 0,   # Different content (not plagiarized)
            'neutral': 0          # Treat neutral as not plagiarized
        }
        
        # Apply label mapping - use 'label' column from SNLI
        # SNLI labels: 0=entailment, 1=neutral, 2=contradiction
        snli_to_text = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        # Convert numeric labels to text labels first
        df['label_text'] = df['label'].map(snli_to_text)
        
        # Filter out invalid labels (those that are -1 or NaN)
        df = df[df['label_text'].notna()].copy()
        
        # Apply plagiarism label mapping
        df['plagiarism_label'] = df['label_text'].map(label_mapping)
        
        # Select relevant columns
        processed_df = df[['premise', 'hypothesis', 'plagiarism_label']].copy()
        processed_df.columns = ['sentence1', 'sentence2', 'label']
        
        # Remove rows with missing values
        processed_df = processed_df.dropna()
        
        # Save processed data
        output_path = self.processed_dir / output_name
        processed_df.to_csv(output_path, sep='\t', index=False, header=False)
        
        logger.info(f"Processed dataset saved to {output_path}")
        logger.info(f"Dataset shape: {processed_df.shape}")
        logger.info(f"Label distribution: {processed_df['label'].value_counts().to_dict()}")
        
        return processed_df
    
    def download_and_process_all(self):
        """Download and process all SNLI datasets"""
        logger.info("Starting MIT Plagiarism Dataset download and processing...")
        
        # Download SNLI dataset
        dataset = self.download_snli_dataset()
        if dataset is None:
            logger.error("Failed to download SNLI dataset")
            return
        
        datasets = {}
        
        # Process each split
        for split_name in ['train', 'validation', 'test']:
            try:
                # Use 'validation' instead of 'dev' for SNLI
                actual_split = 'validation' if split_name == 'dev' else split_name
                
                if actual_split in dataset:
                    # Process for plagiarism detection
                    processed_name = f"plagiarism_{split_name}.txt"
                    processed_df = self.process_snli_for_plagiarism(dataset[actual_split], processed_name)
                    datasets[split_name] = processed_df
                else:
                    logger.warning(f"Split '{actual_split}' not found in dataset")
                
            except Exception as e:
                logger.error(f"Error processing {split_name}: {str(e)}")
                continue
        
        # Create combined dataset
        if datasets:
            logger.info("Creating combined dataset...")
            combined_df = pd.concat(datasets.values(), ignore_index=True)
            combined_path = self.processed_dir / "plagiarism_combined.txt"
            combined_df.to_csv(combined_path, sep='\t', index=False, header=False)
            
            logger.info(f"Combined dataset saved to {combined_path}")
            logger.info(f"Combined dataset shape: {combined_df.shape}")
            logger.info(f"Combined label distribution: {combined_df['label'].value_counts().to_dict()}")
        
        # Create sample data for testing
        self.create_sample_data()
        
        logger.info("Dataset download and processing completed!")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample test data...")
        
        sample_data = [
            ("Machine learning algorithms can process large datasets efficiently.", 
             "ML techniques are capable of handling big data effectively.", 1),
            ("The weather is nice today with clear skies.", 
             "Today's weather features beautiful clear skies.", 1),
            ("Python is a programming language.", 
             "The capital of France is Paris.", 0),
            ("Students should submit their assignments on time.", 
             "Pupils must turn in their homework punctually.", 1),
            ("The algorithm optimizes performance through iteration.", 
             "Cats are popular pets around the world.", 0)
        ]
        
        sample_df = pd.DataFrame(sample_data, columns=['sentence1', 'sentence2', 'label'])
        sample_path = self.base_dir / "samples" / "sample_plagiarism_data.txt"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_path, sep='\t', index=False, header=False)
        
        logger.info(f"Sample data created at {sample_path}")

def main():
    """Main function to download and process datasets"""
    print("🚀 MIT Plagiarism Detection Dataset Downloader")
    print("=" * 50)
    
    downloader = DatasetDownloader()
    downloader.download_and_process_all()
    
    print("\n✅ Dataset download and processing completed!")
    print("\nDataset files created:")
    print("- data/processed/plagiarism_train.txt")
    print("- data/processed/plagiarism_dev.txt") 
    print("- data/processed/plagiarism_test.txt")
    print("- data/processed/plagiarism_combined.txt")
    print("- data/samples/sample_plagiarism_data.txt")

if __name__ == "__main__":
    main()