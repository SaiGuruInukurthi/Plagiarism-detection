# Test Files for Plagiarism Detection

This directory contains sample text files for testing the trained BERT plagiarism detection model.

## Test File Descriptions

### 1. `original_essay.txt`
- **Topic**: Climate Change and Its Environmental Impact
- **Length**: ~300 words
- **Purpose**: Baseline original text for comparison

### 2. `plagiarized_essay.txt`
- **Topic**: Environmental Challenges of Climate Change (same as original)
- **Length**: ~300 words  
- **Similarity**: HIGH - Paraphrased version of original_essay.txt
- **Expected Result**: Should be detected as PLAGIARIZED

### 3. `different_topic_essay.txt`
- **Topic**: Artificial Intelligence in Healthcare
- **Length**: ~300 words
- **Similarity**: LOW - Completely different topic
- **Expected Result**: Should be detected as NOT PLAGIARIZED

### 4. `similar_topic_essay.txt`
- **Topic**: Global Warming and Environmental Changes
- **Length**: ~300 words
- **Similarity**: MEDIUM - Same topic but different content and structure
- **Expected Result**: Should be detected as NOT PLAGIARIZED (different enough content)

## Testing Combinations

### High Plagiarism Expected:
- `original_essay.txt` vs `plagiarized_essay.txt`

### Low Plagiarism Expected:
- `original_essay.txt` vs `different_topic_essay.txt`
- `original_essay.txt` vs `similar_topic_essay.txt`
- `different_topic_essay.txt` vs `similar_topic_essay.txt`

### Medium Plagiarism Expected:
- `original_essay.txt` vs `similar_topic_essay.txt` (same topic, different approach)

## Usage

Use these files with the file-based testing cell in the Jupyter notebook to evaluate model performance on real text documents.

---
*Created for BERT Plagiarism Detection Model Testing*