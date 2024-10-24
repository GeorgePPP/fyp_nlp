# preprocessing/text_processor.py
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

class TextProcessor:
    def __init__(self, config) -> None:
        """
        Initialize text processor with configuration parameters.
        
        Args:
            config: Dictionary containing text preprocessing parameters
        """
        self.config = config
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Complete text preprocessing pipeline"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter by word length
        tokens = [
            token for token in tokens 
            if self.config['min_word_length'] <= len(token) <= self.config['max_word_length']
        ]
        print(f"Number of tokens: ": {len(tokens)})
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        return ' '.join(tokens)