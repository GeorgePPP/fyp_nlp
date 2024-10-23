# features/nlp_features.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import numpy as np

class FeatureExtractor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_df=0.7,
            min_df=5,
            max_features=max_features,
            stop_words='english'
        )
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.lda = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=50,
            learning_offset=50.,
            doc_topic_prior=0.1,
            topic_word_prior=0.1
        )

    def fit_transform(self, texts):
        """Extract and transform features"""
        # TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Dimensionality reduction with SVD
        svd_features = self.svd.fit_transform(tfidf_matrix)
        
        # Topic modeling with LDA
        lda_features = self.lda.fit_transform(tfidf_matrix)
        
        # Combine all features
        combined_features = np.hstack([svd_features, lda_features])
        
        return combined_features

    def transform(self, texts):
        """Transform new texts using fitted extractors"""
        tfidf_matrix = self.vectorizer.transform(texts)
        svd_features = self.svd.transform(tfidf_matrix)
        lda_features = self.lda.transform(tfidf_matrix)
        return np.hstack([svd_features, lda_features])