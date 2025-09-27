"""
Configuration file for Plagiarism Detection System
Contains all configurable parameters and settings.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Dataset configuration - SNLI-based plagiarism detection dataset
DATASET_CONFIG = {
    # SNLI dataset files (569K total samples)
    'train_file': PROCESSED_DATA_DIR / "plagiarism_train.txt",         # 549,367 samples
    'validation_file': PROCESSED_DATA_DIR / "plagiarism_validation.txt", # 9,842 samples
    'test_file': PROCESSED_DATA_DIR / "plagiarism_test.txt",           # 9,824 samples
    'combined_file': PROCESSED_DATA_DIR / "plagiarism_combined.txt",   # 569,033 samples
    
    # Dataset metadata
    'dataset_source': 'Stanford Natural Language Inference (SNLI)',
    'total_samples': 569033,
    'train_samples': 549367,
    'validation_samples': 9842,
    'test_samples': 9824,
    'label_distribution': {
        'not_plagiarized': 378920,  # Label 0 (~66%)
        'plagiarized': 190113       # Label 1 (~34%)
    }
}

# Model configuration
MODEL_CONFIG = {
    # BERT model settings
    'bert_model_name': 'bert-base-uncased',
    'max_sequence_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    
    # TF-IDF settings
    'tfidf_max_features': 10000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.95,
    
    # N-gram settings
    'ngram_sizes': [2, 3, 4],
    
    # Similarity thresholds
    'similarity_thresholds': {
        'high_plagiarism': 0.8,      # > 80% similarity = definite plagiarism
        'medium_plagiarism': 0.6,    # 60-80% = likely plagiarism
        'low_plagiarism': 0.4,       # 40-60% = possible plagiarism
        'no_plagiarism': 0.4         # < 40% = unlikely plagiarism
    }
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    'remove_stopwords': True,
    'lowercase': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'min_word_length': 2,
    'max_word_length': 50,
    'language': 'english',
    
    # File format support
    'supported_formats': ['.txt', '.docx', '.pdf', '.md'],
    
    # Preprocessing steps
    'preprocessing_steps': [
        'lowercase',
        'remove_extra_whitespace',
        'remove_special_chars',
        'tokenize',
        'remove_stopwords',
        'lemmatize'
    ]
}

# Similarity algorithm weights
SIMILARITY_WEIGHTS = {
    'tfidf_cosine': 0.3,        # 30% weight for TF-IDF cosine similarity
    'bert_semantic': 0.4,       # 40% weight for BERT semantic similarity
    'jaccard': 0.15,            # 15% weight for Jaccard similarity
    'ngram': 0.15               # 15% weight for N-gram similarity
}

# Reporting configuration
REPORT_CONFIG = {
    'output_format': 'html',  # Options: 'html', 'pdf', 'json', 'txt'
    'include_visualizations': True,
    'include_similarity_matrix': True,
    'include_highlighted_text': True,
    'max_similar_pairs': 10,
    
    # Visualization settings
    'figure_size': (12, 8),
    'color_scheme': 'viridis',
    'font_size': 12,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'logs' / 'plagiarism_detection.log',
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'num_workers': 4,
    'prefetch_factor': 2,
    'pin_memory': True,
    
    # Memory optimization
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'max_batch_size': 32,
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'confusion_matrix'
    ],
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'random_state': 42
}

# API configuration (for future web interface)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'allowed_extensions': ['.txt', '.docx', '.pdf'],
    'rate_limit': '100/hour'
}

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'development':
    MODEL_CONFIG['batch_size'] = 8
    MODEL_CONFIG['num_epochs'] = 1
    LOGGING_CONFIG['level'] = 'DEBUG'
    API_CONFIG['debug'] = True

elif os.getenv('ENVIRONMENT') == 'production':
    MODEL_CONFIG['batch_size'] = 32
    PERFORMANCE_CONFIG['use_gpu'] = True
    LOGGING_CONFIG['level'] = 'WARNING'
    API_CONFIG['debug'] = False

# Utility functions
def get_device():
    """Get the best available device (GPU/CPU)"""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() and PERFORMANCE_CONFIG['use_gpu'] else 'cpu'
    except ImportError:
        return 'cpu'

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, 
        LOGS_DIR, LOGGING_CONFIG['log_file'].parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Validate similarity weights sum to 1.0
    total_weight = sum(SIMILARITY_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Similarity weights sum to {total_weight}, should be 1.0")
    
    # Validate threshold ordering
    thresholds = MODEL_CONFIG['similarity_thresholds']
    if not (thresholds['high_plagiarism'] > thresholds['medium_plagiarism'] > 
            thresholds['low_plagiarism'] >= thresholds['no_plagiarism']):
        errors.append("Similarity thresholds are not properly ordered")
    
    # Validate file paths
    for config_name, config_dict in [('DATASET_CONFIG', DATASET_CONFIG)]:
        for key, path in config_dict.items():
            if key.endswith('_file') and not path.parent.exists():
                errors.append(f"Directory for {config_name}.{key} does not exist: {path.parent}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Initialize on import
if __name__ != "__main__":
    create_directories()
    # Note: validation is skipped on import to avoid issues during setup