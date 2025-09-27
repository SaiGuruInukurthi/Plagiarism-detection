#!/usr/bin/env python3
"""
Download SNLI dataset from Hugging Face and convert to plagiarism detection format
"""

import os
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_snli():
    """Download SNLI dataset from Hugging Face"""
    try:
        from datasets import load_dataset
        logger.info("📡 Downloading SNLI dataset from Hugging Face...")
        
        # Download the dataset
        dataset = load_dataset("snli")
        
        logger.info("✅ SNLI dataset downloaded successfully!")
        logger.info(f"📊 Dataset info:")
        for split_name, split_data in dataset.items():
            logger.info(f"  - {split_name}: {len(split_data)} samples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to download SNLI dataset: {str(e)}")
        return None

def process_snli_split(split_data, split_name):
    """Process a single split of SNLI data"""
    logger.info(f"🔧 Processing {split_name} split...")
    
    # Convert to pandas DataFrame
    df = split_data.to_pandas()
    
    # Filter out invalid labels (-1)
    df = df[df['label'] != -1].copy()
    
    # Map SNLI labels to plagiarism labels
    # SNLI: 0=entailment, 1=neutral, 2=contradiction
    # Plagiarism: 0=entailment (similar/plagiarized), 1=contradiction/neutral (different)
    label_mapping = {
        0: 1,  # entailment -> plagiarized (similar)
        1: 0,  # neutral -> not plagiarized
        2: 0   # contradiction -> not plagiarized
    }
    
    df['plagiarism_label'] = df['label'].map(label_mapping)
    
    # Select and rename columns
    processed_df = df[['premise', 'hypothesis', 'plagiarism_label']].copy()
    processed_df.columns = ['sentence1', 'sentence2', 'label']
    
    # Remove rows with missing values
    processed_df = processed_df.dropna()
    
    logger.info(f"✅ Processed {split_name}: {len(processed_df)} samples")
    logger.info(f"   Label distribution: {processed_df['label'].value_counts().to_dict()}")
    
    return processed_df

def save_processed_data():
    """Download, process and save SNLI data"""
    
    # Create directories
    base_dir = Path("data")
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SNLI dataset
    dataset = download_snli()
    if dataset is None:
        return False
    
    processed_datasets = {}
    
    # Process each split
    for split_name in ['train', 'validation', 'test']:
        if split_name in dataset:
            processed_df = process_snli_split(dataset[split_name], split_name)
            processed_datasets[split_name] = processed_df
            
            # Save individual split
            output_file = processed_dir / f"plagiarism_{split_name}.txt"
            processed_df.to_csv(output_file, sep='\t', index=False, header=False)
            logger.info(f"💾 Saved {split_name} to {output_file}")
    
    # Create combined dataset
    if processed_datasets:
        logger.info("🔗 Creating combined dataset...")
        combined_df = pd.concat(processed_datasets.values(), ignore_index=True)
        
        combined_file = processed_dir / "plagiarism_combined.txt"
        combined_df.to_csv(combined_file, sep='\t', index=False, header=False)
        
        logger.info(f"💾 Saved combined dataset to {combined_file}")
        logger.info(f"📊 Combined dataset: {len(combined_df)} samples")
        logger.info(f"🏷️ Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    return True

def main():
    print("🚀 SNLI Dataset Downloader for Plagiarism Detection")
    print("=" * 55)
    
    try:
        success = save_processed_data()
        
        if success:
            print("\n✅ SNLI dataset download and processing completed!")
            print("\n📁 Files created:")
            print("  - data/processed/plagiarism_train.txt")
            print("  - data/processed/plagiarism_validation.txt")
            print("  - data/processed/plagiarism_test.txt")
            print("  - data/processed/plagiarism_combined.txt")
            print("\n🎯 Ready for model training!")
        else:
            print("\n❌ Dataset download failed!")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()