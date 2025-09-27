#!/usr/bin/env python3
"""
Alternative SNLI downloader using direct JSON file downloads
"""

import requests
import json
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SNLI direct download URLs
SNLI_URLS = {
    'train': 'https://nlp.stanford.edu/projects/snli/snli_1.0_train.jsonl',
    'validation': 'https://nlp.stanford.edu/projects/snli/snli_1.0_dev.jsonl',
    'test': 'https://nlp.stanford.edu/projects/snli/snli_1.0_test.jsonl'
}

def download_file(url, filename):
    """Download a file with progress bar"""
    logger.info(f"📡 Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        filepath = Path("data/raw") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)
        
        logger.info(f"✅ Downloaded {filename}")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {str(e)}")
        return None

def process_jsonl_file(filepath, split_name):
    """Process a JSONL file into plagiarism format"""
    logger.info(f"🔧 Processing {filepath}...")
    
    data = []
    valid_count = 0
    total_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                total_count += 1
                try:
                    item = json.loads(line.strip())
                    
                    # Skip items with invalid labels
                    if item.get('gold_label') == '-' or item.get('gold_label') is None:
                        continue
                    
                    # Map labels: entailment -> 1 (plagiarized), others -> 0 (not plagiarized)  
                    label_mapping = {
                        'entailment': 1,
                        'neutral': 0,
                        'contradiction': 0
                    }
                    
                    gold_label = item.get('gold_label')
                    if gold_label in label_mapping:
                        data.append({
                            'sentence1': item.get('sentence1', ''),
                            'sentence2': item.get('sentence2', ''),
                            'label': label_mapping[gold_label]
                        })
                        valid_count += 1
                        
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"✅ Processed {split_name}: {valid_count}/{total_count} valid samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Remove empty sentences
        df = df[(df['sentence1'].str.strip() != '') & (df['sentence2'].str.strip() != '')].copy()
        
        logger.info(f"📊 Final {split_name}: {len(df)} samples")
        logger.info(f"🏷️ Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error processing {filepath}: {str(e)}")
        return pd.DataFrame()

def download_and_process_snli():
    """Download and process all SNLI files"""
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_datasets = {}
    
    # Download and process each split
    for split_name, url in SNLI_URLS.items():
        filename = f"snli_{split_name}.jsonl"
        
        # Download file
        filepath = download_file(url, filename)
        if filepath is None or not filepath.exists():
            logger.error(f"❌ Skipping {split_name} - download failed")
            continue
            
        # Process file
        df = process_jsonl_file(filepath, split_name)
        if len(df) == 0:
            logger.error(f"❌ Skipping {split_name} - processing failed")
            continue
            
        all_datasets[split_name] = df
        
        # Save processed file
        output_file = processed_dir / f"plagiarism_{split_name}.txt"
        df.to_csv(output_file, sep='\t', index=False, header=False)
        logger.info(f"💾 Saved to {output_file}")
    
    # Create combined dataset
    if all_datasets:
        logger.info("🔗 Creating combined dataset...")
        combined_df = pd.concat(all_datasets.values(), ignore_index=True)
        
        combined_file = processed_dir / "plagiarism_combined.txt"
        combined_df.to_csv(combined_file, sep='\t', index=False, header=False)
        
        logger.info(f"💾 Combined dataset: {len(combined_df)} samples")
        logger.info(f"🏷️ Combined labels: {combined_df['label'].value_counts().to_dict()}")
        
        return True
    
    return False

def main():
    print("🚀 SNLI Direct Download for Plagiarism Detection")
    print("=" * 50)
    
    try:
        success = download_and_process_snli()
        
        if success:
            print("\n✅ SNLI dataset download completed!")
            print("\n📁 Files created:")
            print("  - data/raw/snli_train.jsonl")
            print("  - data/raw/snli_validation.jsonl") 
            print("  - data/raw/snli_test.jsonl")
            print("  - data/processed/plagiarism_train.txt")
            print("  - data/processed/plagiarism_validation.txt")
            print("  - data/processed/plagiarism_test.txt")
            print("  - data/processed/plagiarism_combined.txt")
            print("\n🎯 Ready for model training!")
        else:
            print("\n❌ Download failed!")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()