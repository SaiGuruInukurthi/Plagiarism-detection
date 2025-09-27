# Academic Plagiarism Detection System

## Project Overview

**Status: Production Ready ✅**

This project implements a BERT-based plagiarism detection system specifically designed for academic assignments. The system has been successfully trained and tested, achieving **86.7% accuracy** and **0.868 F1-score** on the SNLI dataset adapted for plagiarism detection.

### Trained Model Available
- **Model**: Fine-tuned BERT-base-uncased classifier
- **Performance**: 86.7% test accuracy, 0.868 F1-score  
- **Status**: Production ready with comprehensive file-based testing
- **Usage**: Can analyze any text files for plagiarism detection

## Key Features

- **✅ Production-ready BERT classifier** trained on 569K sentence pairs
- **✅ File-based testing system** with 14 diverse test documents
- **✅ Real-time plagiarism detection** with confidence scores
- **✅ Comprehensive evaluation metrics** and performance analysis
- **✅ Clean, optimized codebase** with duplicate removal and emoji cleanup

## System Architecture

```
Input Text Files (.txt)
    ↓
BERT Tokenizer (bert-base-uncased)
    ↓  
Fine-tuned BERT Classifier
    ├─ 768-dimensional embeddings
    ├─ Binary classification head
    └─ Dropout regularization
    ↓
Prediction Output
    ├─ Plagiarized/Not Plagiarized
    ├─ Confidence score (0-1)
    └─ Individual class probabilities
```

## Dataset

**Primary Dataset**: Stanford Natural Language Inference (SNLI) adapted for plagiarism detection
- **Source**: Official SNLI dataset from Hugging Face (`stanfordnlp/snli`)
- **Format**: Tab-separated values (sentence1, sentence2, label)
- **Size**: 569,033 sentence pairs total
  - **Training**: 549,367 samples
  - **Validation**: 9,842 samples  
  - **Test**: 9,824 samples
- **Labels**: Binary classification (0=not plagiarized, 1=plagiarized)
- **Label Mapping**: 
  - SNLI Entailment → Plagiarized (similar/related content)
  - SNLI Neutral/Contradiction → Not Plagiarized (different content)

### Dataset Files
- `data/processed/plagiarism_train.txt` - Training data (549K samples)
- `data/processed/plagiarism_validation.txt` - Validation data (9.8K samples)
- `data/processed/plagiarism_test.txt` - Test data (9.8K samples)
- `data/processed/plagiarism_combined.txt` - All data combined (569K samples)
- `data/backups/backup_20250927_183051/` - Complete dataset backup (309MB)

## Model Performance

### Training Results
- **Test Accuracy**: 86.7%
- **Test F1-Score**: 0.868
- **Model Size**: 417.7MB (BERT-base-uncased fine-tuned)
- **Training Time**: ~2 hours on GPU
- **Inference Time**: <1 second per file pair

### File-Based Testing
The system includes comprehensive testing on 14 diverse text files covering:
- Climate change and environmental science
- AI and healthcare technology
- Social media communication
- Machine learning in business
- Renewable energy technologies
- Modern education systems
- Cybersecurity challenges
- Urban planning and development
- Consumer psychology
- Global trade economics
- Sports analytics
- Ancient civilizations

**Testing Results**: The model successfully discriminates between genuinely different content vs actual plagiarism, avoiding false positives on diverse topics.

## Project Structure

```
plagiarism-detection/
├── README.md                    # Project documentation  
├── requirements.txt             # Python dependencies (production-ready)
├── config.py                   # Configuration settings
├── main.py                     # CLI interface and main entry point
├── MODEL_INFO.md               # Trained model information
├── clean_notebook_emojis.py    # Notebook cleanup utility
├── data/                       # Dataset storage
│   ├── processed/              # SNLI dataset files (569K samples)
│   ├── backups/               # Dataset backups with metadata
│   └── working/               # Processed sample data
├── models/                     # Trained models
│   ├── bert_plagiarism_detector_*.pth      # Model weights (417MB)
│   ├── bert_plagiarism_detector_*_metadata.json  # Model metadata
│   └── bert_plagiarism_detector_*_tokenizer/     # Tokenizer files
├── test_files/                 # Diverse test documents (14 files)
│   ├── climate_change_essay.txt
│   ├── ai_healthcare_essay.txt
│   ├── consumer_psychology.txt
│   └── ... (11 more diverse topics)
├── notebooks/                  # Jupyter notebooks
│   └── Plagirism-Detection.ipynb  # Main training/testing notebook
├── results/                    # Analysis outputs and visualizations
├── scripts/                    # Utility scripts
└── logs/                       # Application logs
```

## Getting Started

### Prerequisites
- Python 3.13+ (tested with 3.13.7)
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for model training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SaiGuruInukurthi/Plagiarism-detection.git
   cd plagiarism-detection
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n plagirism python=3.13 -y
   conda activate plagirism
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Open the main notebook**
   ```bash
   jupyter lab notebooks/Plagirism-Detection.ipynb
   ```

2. **Run the trained model** (cells 44-46 in notebook)
   - Cell 44: File-based plagiarism detection functions
   - Cell 45: Automated testing on all file combinations  
   - Cell 46: Manual testing for custom files

3. **Test with your own files**
   ```python
   # In the notebook
   result = test_custom_files('path/to/file1.txt', 'path/to/file2.txt')
   ```

### File-Based Testing

The system includes a comprehensive file-based testing suite:

```python
# Automatic testing on all 14 test files (91 combinations)
# Run cell 45 in the notebook for complete analysis

# Manual testing on specific files
result = analyze_file_pair(
    Path('test_files/original_essay.txt'),
    Path('test_files/plagiarized_essay.txt'),
    bert_trainer.bert_classifier,
    bert_trainer.tokenizer,
    device
)
```

## Technical Implementation

### Core Technologies
- **PyTorch 2.8.0** - Deep learning framework with CUDA support
- **Transformers 4.56.2** - Hugging Face BERT implementation
- **Scikit-learn 1.7.2** - Evaluation metrics
- **Pandas 2.3.2** - Data manipulation
- **NLTK 3.9.1** - Text preprocessing

### Model Architecture
- **Base Model**: BERT-base-uncased (12 layers, 768 hidden units)
- **Classification Head**: Linear layer for binary classification
- **Training**: Fine-tuned on SNLI dataset with manual training loop
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Regularization**: Dropout layers for generalization

### Key Features
1. **File Format Support**
   - Plain text (.txt) files
   - Automatic encoding detection
   - Comprehensive error handling

2. **Advanced Processing**
   - BERT tokenization with 128 max length
   - Confidence score calculation
   - Batch processing capabilities

3. **Performance Monitoring**
   - Real-time accuracy tracking
   - Confidence interval analysis
   - Detailed prediction logging

## Usage Examples

### Basic Similarity Check
```python
# Load the trained model (available in notebook)
result = predict_plagiarism(text1, text2, model, tokenizer, device)
print(f"Plagiarized: {result['is_plagiarized']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Processing
```python
# Test all files in test_files directory
# Automatically runs 91 comparisons on 14 diverse files
# See notebook cell 45 for implementation
```

## Performance Analysis

### Model Strengths
- **High accuracy** (86.7%) on diverse text pairs
- **Good discrimination** between similar and different topics
- **Fast inference** (<1 second per comparison)
- **Robust handling** of different writing styles

### Testing Coverage
- **91 file combinations** tested automatically
- **14 diverse topics** covering multiple domains
- **Comprehensive evaluation** of false positive rates
- **Confidence score validation** across different similarity levels

## Development History

### Completed Milestones
- ✅ **Environment Setup** - Conda environment with Python 3.13
- ✅ **Dataset Processing** - SNLI dataset (569K samples) downloaded and processed
- ✅ **Model Training** - BERT classifier fine-tuned with 86.7% accuracy
- ✅ **File-Based Testing** - 14 diverse test files with automated evaluation
- ✅ **Code Cleanup** - Removed duplicates and optimized notebook structure
- ✅ **Production Ready** - Comprehensive documentation and testing

## Configuration

The system uses `config.py` for configuration management:
- Dataset paths and processing parameters
- Model hyperparameters and training settings
- Logging and output configuration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. **Academic Papers**
   - "A large annotated corpus for learning natural language inference" (Bowman et al., 2015) - SNLI Dataset
   - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)

2. **Datasets**
   - **Stanford Natural Language Inference (SNLI) Corpus**: 569,033 sentence pairs
   - **Hugging Face**: `stanfordnlp/snli` dataset

3. **Libraries and Frameworks**
   - **PyTorch**: Deep learning framework
   - **Hugging Face Transformers**: BERT implementation
   - **Scikit-learn**: Evaluation metrics
   - **Pandas/NumPy**: Data processing

## Contact

For questions, suggestions, or collaboration:
- **Repository**: [https://github.com/SaiGuruInukurthi/Plagiarism-detection](https://github.com/SaiGuruInukurthi/Plagiarism-detection)
- **Issues**: Please use GitHub Issues for bug reports and feature requests

---

**Last Updated**: September 28, 2025  
**Version**: 2.0.0  
**Status**: Production Ready ✅  
**Model Performance**: 86.7% accuracy, 0.868 F1-score ✅