# preprocessing/text_processor.py
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Complete text preprocessing pipeline"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)

    def extract_nlp_features(self, text):
        """Extract NLP features using spaCy"""
        doc = self.nlp(text)
        
        features = {
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'verbs': [token.text for token in doc if token.pos_ == "VERB"],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'dependency_pairs': [(token.text, token.dep_, token.head.text) 
                               for token in doc]
        }
        
        return features