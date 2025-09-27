#!/usr/bin/env python3
"""
Download SNLI dataset from Hugging Face using the correct dataset name
"""

import pandas as pd
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_snli_hf():
    """Download SNLI dataset from Hugging Face"""
    try:
        logger.info("📡 Importing datasets library...")
        from datasets import load_dataset
        
        logger.info("📡 Downloading SNLI dataset from stanfordnlp/snli...")
        
        # Download the dataset
        ds = load_dataset("stanfordnlp/snli")
        
        logger.info("✅ SNLI dataset downloaded successfully!")
        logger.info(f"📊 Dataset splits: {list(ds.keys())}")
        
        for split_name in ds.keys():
            logger.info(f"  - {split_name}: {len(ds[split_name])} samples")
        
        return ds
        
    except ImportError as e:
        logger.error(f"❌ Failed to import datasets: {str(e)}")
        logger.error("💡 Try: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to download SNLI dataset: {str(e)}")
        return None

def process_snli_split(split_data, split_name):
    """Process a single split of SNLI data"""
    logger.info(f"🔧 Processing {split_name} split...")
    
    # Convert to pandas DataFrame
    df = split_data.to_pandas()
    
    logger.info(f"📊 Original {split_name} shape: {df.shape}")
    logger.info(f"🏷️ Original labels: {df['label'].value_counts().to_dict()}")
    
    # Filter out invalid labels (-1)
    df = df[df['label'] != -1].copy()
    
    # Map SNLI labels to plagiarism labels
    # SNLI: 0=entailment, 1=neutral, 2=contradiction
    # For plagiarism: entailment (0) -> similar content -> 1 (plagiarized)
    #                neutral/contradiction (1,2) -> different content -> 0 (not plagiarized)
    label_mapping = {
        0: 1,  # entailment -> plagiarized (similar content)
        1: 0,  # neutral -> not plagiarized  
        2: 0   # contradiction -> not plagiarized
    }
    
    df['plagiarism_label'] = df['label'].map(label_mapping)
    
    # Select and rename columns
    processed_df = df[['premise', 'hypothesis', 'plagiarism_label']].copy()
    processed_df.columns = ['sentence1', 'sentence2', 'label']
    
    # Remove rows with missing values
    processed_df = processed_df.dropna()
    
    # Remove empty sentences
    processed_df = processed_df[
        (processed_df['sentence1'].str.strip() != '') & 
        (processed_df['sentence2'].str.strip() != '')
    ].copy()
    
    logger.info(f"✅ Processed {split_name}: {len(processed_df)} samples")
    logger.info(f"🏷️ Label distribution: {processed_df['label'].value_counts().to_dict()}")
    
    return processed_df

def save_processed_data():
    """Download, process and save SNLI data"""
    
    # Create directories
    base_dir = Path("data")
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download SNLI dataset
    logger.info("🚀 Starting SNLI dataset download...")
    dataset = download_snli_hf()
    if dataset is None:
        return False
    
    processed_datasets = {}
    
    # Process each split
    for split_name in dataset.keys():
        try:
            processed_df = process_snli_split(dataset[split_name], split_name)
            processed_datasets[split_name] = processed_df
            
            # Save individual split
            output_file = processed_dir / f"plagiarism_{split_name}.txt"
            processed_df.to_csv(output_file, sep='\t', index=False, header=False)
            logger.info(f"💾 Saved {split_name} to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {split_name}: {str(e)}")
            continue
    
    # Create combined dataset
    if processed_datasets:
        logger.info("🔗 Creating combined dataset...")
        combined_df = pd.concat(processed_datasets.values(), ignore_index=True)
        
        combined_file = processed_dir / "plagiarism_combined.txt"
        combined_df.to_csv(combined_file, sep='\t', index=False, header=False)
        
        logger.info(f"💾 Saved combined dataset to {combined_file}")
        logger.info(f"📊 Combined dataset: {len(combined_df)} samples")
        logger.info(f"🏷️ Combined label distribution: {combined_df['label'].value_counts().to_dict()}")
        
        # Show sample data
        logger.info("📋 Sample data:")
        for i, row in combined_df.head(3).iterrows():
            logger.info(f"  {i+1}. Sentence1: {row['sentence1'][:50]}...")
            logger.info(f"     Sentence2: {row['sentence2'][:50]}...")
            logger.info(f"     Label: {row['label']} ({'Plagiarized' if row['label']==1 else 'Not Plagiarized'})")
    
    return True

def main():
    print("🚀 SNLI Dataset Downloader (Hugging Face)")
    print("=" * 50)
    
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
            
            # Show file sizes
            processed_dir = Path("data/processed")
            for file_path in processed_dir.glob("plagiarism_*.txt"):
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    print(f"  📊 {file_path.name}: {line_count:,} samples")
        else:
            print("\n❌ Dataset download failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())