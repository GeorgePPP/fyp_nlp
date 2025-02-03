import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
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
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos(self, tag):
        """
        Map POS tag to WordNet POS tag format
        """
        tag_map = {
            'J': 'a',  # Adjective
            'N': 'n',  # Noun
            'V': 'v',  # Verb
            'R': 'r'   # Adverb
        }
        return tag_map.get(tag[0], 'n')  # Default to noun if tag not found

    def preprocess(self, text):
        """Complete text preprocessing pipeline"""
    
        # Lowercase the text
        text = text.lower()
        # Remove any non-alphabet characters
        text = re.sub(r'[^a-z\s]', '', text)

        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]

        tokens = [word for word in lemmatized if word not in self.stop_words]
        
        return ' '.join(tokens)