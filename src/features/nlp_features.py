# features/nlp_features.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        """
        Initialize feature extractors with configuration parameters.
        
        Args:
            config: Dictionary containing feature extraction parameters
        """
        self.config = config
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            **self.config['tfidf']
        )
        
        # Initialize SVD
        self.svd = TruncatedSVD(
            **self.config['svd']
        )
        
        # Initialize LDA
        self.lda = LatentDirichletAllocation(
            **self.config['lda']
        )

    def fit_transform(self, texts):
        """Extract and transform features"""
        # TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF Matrix: {tfidf_matrix.shape}")
        print(f"• Samples: {tfidf_matrix.shape[0]}")
        print(f"• Vocabulary Size: {tfidf_matrix.shape[1]}")
        
        # Dimensionality reduction with SVD
        svd_features = self.svd.fit_transform(tfidf_matrix)
        print("\nSVD Features:")
        print(f"• Input: {tfidf_matrix.shape}")
        print(f"• Output: {svd_features.shape}")
        print(f"• Dimension Reduction: {tfidf_matrix.shape[1]} → {svd_features.shape[1]}")
        print(f"• Explained Variance Ratio: {self.svd.explained_variance_ratio_.sum():.2%}")
        
        # Topic modeling with LDA
        lda_features = self.lda.fit_transform(tfidf_matrix)
        print("\nLDA Features:")
        print(f"• Input: {tfidf_matrix.shape}")
        print(f"• Output: {lda_features.shape}")
        print(f"• Number of Topics: {lda_features.shape[1]}")
        
        # Combine all features
        combined_features = np.hstack([svd_features, lda_features])
        print("\nCombined Features:")
        print(f"• SVD Features: {svd_features.shape[1]} dimensions")
        print(f"• LDA Features: {lda_features.shape[1]} dimensions")
        print(f"• Total Features: {combined_features.shape[1]} dimensions")
        print("-" * 40)
        
        return combined_features

    def transform(self, texts):
        """Transform new texts using fitted extractors"""
        tfidf_matrix = self.vectorizer.transform(texts)
        svd_features = self.svd.transform(tfidf_matrix)
        lda_features = self.lda.transform(tfidf_matrix)
        return np.hstack([svd_features, lda_features])