#!/usr/bin/env python3
"""
Main entry point for the Academic Plagiarism Detection System

Usage:
    python main.py --help                          # Show help
    python main.py --setup                         # Initial setup
    python main.py --create-dataset                # Create sample dataset
    python main.py --train                         # Train models
    python main.py --evaluate                      # Evaluate models
    python main.py --compare file1.txt file2.txt   # Compare two files
    python main.py --batch-process directory/      # Process directory
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from config import LOGGING_CONFIG, create_directories, validate_config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGGING_CONFIG['log_file'], mode='a')
        ]
    )
    return logging.getLogger(__name__)

def setup_project():
    """Initialize project setup"""
    logger = setup_logging()
    logger.info("🚀 Setting up Academic Plagiarism Detection System...")
    
    try:
        # Create necessary directories
        create_directories()
        logger.info("✅ Created project directories")
        
        # Validate configuration
        validate_config()
        logger.info("✅ Configuration validated")
        
        # Check if dataset exists
        from config import DATASET_CONFIG
        if not DATASET_CONFIG['combined_file'].exists():
            logger.warning("⚠️ Dataset not found. Run 'python main.py --create-dataset' first")
        else:
            logger.info("✅ Dataset found")
        
        logger.info("🎉 Project setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {str(e)}")
        return False

def create_sample_dataset():
    """Create sample dataset for testing"""
    logger = setup_logging()
    logger.info("📊 Creating sample dataset...")
    
    try:
        from scripts.create_sample_dataset import create_dataset_files
        datasets = create_dataset_files()
        
        logger.info("✅ Sample dataset created successfully!")
        print(f"📈 Dataset Statistics:")
        print(f"  - Total samples: {len(datasets['combined'])}")
        print(f"  - Training: {len(datasets['train'])}")
        print(f"  - Validation: {len(datasets['validation'])}")
        print(f"  - Test: {len(datasets['test'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {str(e)}")
        return False

def compare_files(file1_path, file2_path):
    """Compare two files for similarity"""
    logger = setup_logging()
    logger.info(f"🔍 Comparing files: {file1_path} vs {file2_path}")
    
    try:
        # Basic file existence check
        if not Path(file1_path).exists():
            raise FileNotFoundError(f"File not found: {file1_path}")
        if not Path(file2_path).exists():
            raise FileNotFoundError(f"File not found: {file2_path}")
        
        # Simple text comparison for now
        with open(file1_path, 'r', encoding='utf-8') as f1:
            text1 = f1.read().strip()
        
        with open(file2_path, 'r', encoding='utf-8') as f2:
            text2 = f2.read().strip()
        
        # Basic similarity calculation (placeholder)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, text1, text2).ratio()
        
        print(f"📊 Similarity Analysis:")
        print(f"  - File 1: {file1_path}")
        print(f"  - File 2: {file2_path}")
        print(f"  - Basic Similarity: {similarity:.2%}")
        
        if similarity > 0.8:
            print("  - 🚨 HIGH similarity detected - Potential plagiarism")
        elif similarity > 0.6:
            print("  - ⚠️ MEDIUM similarity detected - Review recommended")
        elif similarity > 0.4:
            print("  - 💡 LOW similarity detected - Minor similarities")
        else:
            print("  - ✅ LOW similarity - Files appear original")
        
        logger.info(f"✅ File comparison completed. Similarity: {similarity:.2%}")
        return similarity
        
    except Exception as e:
        logger.error(f"❌ File comparison failed: {str(e)}")
        return None

def show_status():
    """Show current project status"""
    logger = setup_logging()
    print("📋 Academic Plagiarism Detection System - Status")
    print("=" * 50)
    
    # Check directories
    from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, DATASET_CONFIG
    
    print("📁 Directory Status:")
    dirs_to_check = [
        ("Data", DATA_DIR),
        ("Models", MODELS_DIR),
        ("Results", RESULTS_DIR)
    ]
    
    for name, path in dirs_to_check:
        status = "✅ EXISTS" if path.exists() else "❌ MISSING"
        print(f"  - {name}: {status} ({path})")
    
    print("\n📊 Dataset Status:")
    dataset_files = [
        ("Training", DATASET_CONFIG['train_file']),
        ("Validation", DATASET_CONFIG['validation_file']),
        ("Test", DATASET_CONFIG['test_file']),
        ("Combined", DATASET_CONFIG['combined_file']),
        ("Sample", DATASET_CONFIG['sample_file'])
    ]
    
    for name, path in dataset_files:
        if path.exists():
            # Count lines in file
            try:
                with open(path, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"  - {name}: ✅ {line_count} samples ({path.name})")
            except:
                print(f"  - {name}: ✅ EXISTS ({path.name})")
        else:
            print(f"  - {name}: ❌ MISSING ({path.name})")
    
    print("\n🔧 Configuration:")
    from config import MODEL_CONFIG, SIMILARITY_WEIGHTS
    print(f"  - BERT Model: {MODEL_CONFIG['bert_model_name']}")
    print(f"  - Max Sequence Length: {MODEL_CONFIG['max_sequence_length']}")
    print(f"  - Batch Size: {MODEL_CONFIG['batch_size']}")
    print(f"  - Similarity Weights: {SIMILARITY_WEIGHTS}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Academic Plagiarism Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                         # Initial project setup
  python main.py --create-dataset                # Create sample dataset
  python main.py --status                        # Show project status
  python main.py --compare file1.txt file2.txt   # Compare two files
  
For more information, see README.md
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Initialize project setup')
    parser.add_argument('--create-dataset', action='store_true',
                       help='Create sample dataset')
    parser.add_argument('--status', action='store_true',
                       help='Show project status')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two files for similarity')
    parser.add_argument('--train', action='store_true',
                       help='Train plagiarism detection models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    parser.add_argument('--batch-process', metavar='DIRECTORY',
                       help='Process all files in a directory')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # If no arguments provided, show status
    if len(sys.argv) == 1:
        show_status()
        return
    
    # Execute based on arguments
    success = True
    
    if args.setup:
        success = setup_project()
        
    elif args.create_dataset:
        success = create_sample_dataset()
        
    elif args.status:
        show_status()
        
    elif args.compare:
        file1, file2 = args.compare
        similarity = compare_files(file1, file2)
        success = similarity is not None
        
    elif args.train:
        print("🔄 Model training functionality will be implemented in next phase")
        print("📝 See README.md for current capabilities")
        
    elif args.evaluate:
        print("🔄 Model evaluation functionality will be implemented in next phase")
        print("📝 See README.md for current capabilities")
        
    elif args.batch_process:
        print("🔄 Batch processing functionality will be implemented in next phase")
        print("📝 See README.md for current capabilities")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()