"""
Simple Dataset Creator for MIT Plagiarism Detection Project
Creates a sample plagiarism detection dataset without external dependencies.
"""

import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_plagiarism_dataset():
    """Create a comprehensive sample plagiarism dataset"""
    
    # Sample data with various types of plagiarism and similarity levels
    sample_data = [
        # High similarity / Plagiarism cases (label = 1)
        ("Machine learning algorithms can process large datasets efficiently and accurately.", 
         "ML techniques are capable of handling big data effectively and precisely.", 1),
        ("The weather is nice today with clear blue skies and sunshine.", 
         "Today's weather features beautiful clear skies with bright sunlight.", 1),
        ("Students should submit their assignments on time to avoid penalties.", 
         "Pupils must turn in their homework punctually to prevent deductions.", 1),
        ("The algorithm optimizes performance through iterative improvement.", 
         "This method enhances efficiency via repeated refinement processes.", 1),
        ("Artificial intelligence is revolutionizing the technology industry.", 
         "AI is transforming the tech sector in unprecedented ways.", 1),
        ("Climate change poses significant threats to global ecosystems.", 
         "Global warming presents serious dangers to worldwide environments.", 1),
        ("The research methodology involves collecting and analyzing data.", 
         "This study's approach includes gathering and examining information.", 1),
        ("Programming languages like Python are essential for data science.", 
         "Coding languages such as Python are crucial for data analysis.", 1),
        ("The company's revenue increased by 25% in the last quarter.", 
         "Corporate earnings rose by a quarter during the previous period.", 1),
        ("Online education has become increasingly popular during the pandemic.", 
         "Digital learning gained significant traction throughout the health crisis.", 1),
        
        # Medium similarity cases (label = 1)
        ("The sun rises in the east and sets in the west every day.",
         "Daily, the sun appears in the eastern horizon and disappears westward.", 1),
        ("Scientists have discovered a new species of butterfly in the Amazon.",
         "Researchers found an unknown butterfly variety in the Amazonian rainforest.", 1),
        ("The novel explores themes of love, loss, and redemption.",
         "This book examines concepts of romance, grief, and salvation.", 1),
        ("Social media platforms have changed how people communicate.",
         "Digital networking sites have transformed interpersonal communication methods.", 1),
        ("The experimental results support the proposed hypothesis.",
         "Study findings validate the suggested theoretical framework.", 1),
        
        # Low similarity / Non-plagiarism cases (label = 0)
        ("Python is a programming language used for web development.", 
         "The capital of France is Paris, a beautiful European city.", 0),
        ("Machine learning requires large amounts of training data.", 
         "Cats are popular pets that require daily care and attention.", 0),
        ("The movie was entertaining and had excellent special effects.", 
         "Mathematics is a fundamental subject taught in schools worldwide.", 0),
        ("Climate change is caused by greenhouse gas emissions.", 
         "The recipe calls for two cups of flour and three eggs.", 0),
        ("The company reported strong financial performance this quarter.", 
         "Soccer is the most popular sport played around the world.", 0),
        ("Online shopping has increased significantly during recent years.", 
         "The human brain contains approximately 86 billion neurons.", 0),
        ("The research paper was published in a prestigious journal.", 
         "Penguins are flightless birds that live in cold climates.", 0),
        ("Artificial intelligence can help solve complex problems.", 
         "The library closes at 9 PM on weekdays and 5 PM on weekends.", 0),
        ("Students need to develop critical thinking skills.", 
         "The ocean covers about 71% of the Earth's surface area.", 0),
        ("The algorithm processes data in real-time efficiently.", 
         "Chocolate is made from cocoa beans grown in tropical regions.", 0),
        
        # Additional diverse examples
        ("Deep learning models require substantial computational resources.", 
         "Neural networks need significant processing power for training.", 1),
        ("The conference presentation was well-received by the audience.", 
         "Dogs are loyal companions that bring joy to families.", 0),
        ("Economic growth depends on various factors including innovation.", 
         "Monetary expansion relies on multiple elements like technological advancement.", 1),
        ("The museum houses an impressive collection of ancient artifacts.", 
         "Vegetables are an important part of a healthy balanced diet.", 0),
        ("Renewable energy sources are becoming more cost-effective.", 
         "Clean power alternatives are increasingly economically viable.", 1),
        
        # Academic-specific examples
        ("The hypothesis was tested using statistical analysis methods.", 
         "This assumption was examined through mathematical evaluation techniques.", 1),
        ("Literature review revealed gaps in current research understanding.", 
         "Academic survey identified deficiencies in existing scholarly knowledge.", 1),
        ("Data collection involved surveys and interviews with participants.", 
         "Information gathering included questionnaires and discussions with subjects.", 1),
        ("The findings contribute to theoretical framework development.", 
         "These results advance conceptual model construction efforts.", 1),
        ("Peer review process ensures academic publication quality.", 
         "The best pizza toppings include cheese, pepperoni, and mushrooms.", 0),
        
        # Edge cases and challenging examples
        ("The quick brown fox jumps over the lazy dog.", 
         "A fast auburn fox leaps above the sleepy canine.", 1),
        ("To be or not to be, that is the question.", 
         "Existence versus non-existence represents the fundamental inquiry.", 1),
        ("Einstein's theory of relativity changed physics forever.", 
         "My grandmother makes the best apple pie in town.", 0),
        ("The database contains millions of customer records.", 
         "This storage system holds countless client information entries.", 1),
        ("Password security is crucial for protecting personal information.", 
         "Birds migrate south during winter to find warmer climates.", 0)
    ]
    
    return sample_data

def create_dataset_files():
    """Create dataset files in appropriate directories"""
    
    # Create base directories
    base_dir = Path("data")
    processed_dir = base_dir / "processed"
    samples_dir = base_dir / "samples"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    sample_data = create_sample_plagiarism_dataset()
    
    # Create DataFrame
    df = pd.DataFrame(sample_data, columns=['sentence1', 'sentence2', 'label'])
    
    # Split data (80% train, 10% validation, 10% test)
    total_samples = len(df)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save datasets
    train_path = processed_dir / "plagiarism_train.txt"
    val_path = processed_dir / "plagiarism_validation.txt"
    test_path = processed_dir / "plagiarism_test.txt"
    combined_path = processed_dir / "plagiarism_combined.txt"
    sample_path = samples_dir / "sample_plagiarism_data.txt"
    
    # Save without headers (tab-separated)
    train_df.to_csv(train_path, sep='\t', index=False, header=False)
    val_df.to_csv(val_path, sep='\t', index=False, header=False)
    test_df.to_csv(test_path, sep='\t', index=False, header=False)
    df.to_csv(combined_path, sep='\t', index=False, header=False)
    df.head(10).to_csv(sample_path, sep='\t', index=False, header=False)
    
    # Log statistics
    logger.info(f"Created dataset files:")
    logger.info(f"- Training set: {len(train_df)} samples -> {train_path}")
    logger.info(f"- Validation set: {len(val_df)} samples -> {val_path}")
    logger.info(f"- Test set: {len(test_df)} samples -> {test_path}")
    logger.info(f"- Combined set: {len(df)} samples -> {combined_path}")
    logger.info(f"- Sample set: 10 samples -> {sample_path}")
    
    # Log label distribution
    train_dist = train_df['label'].value_counts().to_dict()
    total_dist = df['label'].value_counts().to_dict()
    logger.info(f"Training label distribution: {train_dist}")
    logger.info(f"Overall label distribution: {total_dist}")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'combined': df
    }

def main():
    """Main function"""
    print("🚀 MIT Plagiarism Detection Dataset Creator")
    print("=" * 50)
    
    logger.info("Creating sample plagiarism detection dataset...")
    
    try:
        datasets = create_dataset_files()
        
        print(f"\n✅ Dataset creation completed successfully!")
        print(f"\n📊 Dataset Statistics:")
        print(f"- Total samples: {len(datasets['combined'])}")
        print(f"- Training samples: {len(datasets['train'])}")
        print(f"- Validation samples: {len(datasets['validation'])}")
        print(f"- Test samples: {len(datasets['test'])}")
        
        # Show label distribution
        label_dist = datasets['combined']['label'].value_counts()
        print(f"\n🏷️ Label Distribution:")
        print(f"- Plagiarized (1): {label_dist.get(1, 0)} samples")
        print(f"- Non-plagiarized (0): {label_dist.get(0, 0)} samples")
        
        print(f"\n📁 Files created:")
        print(f"- data/processed/plagiarism_train.txt")
        print(f"- data/processed/plagiarism_validation.txt")
        print(f"- data/processed/plagiarism_test.txt")
        print(f"- data/processed/plagiarism_combined.txt")
        print(f"- data/samples/sample_plagiarism_data.txt")
        
        print(f"\n🔧 Ready for model training and testing!")
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        print(f"❌ Dataset creation failed: {str(e)}")

if __name__ == "__main__":
    main()