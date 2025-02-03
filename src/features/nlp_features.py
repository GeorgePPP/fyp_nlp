from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import numpy as np
import spacy
from collections import defaultdict
import pandas as pd
from scipy.sparse import csr_matrix, hstack

class FeatureExtractor:
    def __init__(self, config):
        """
        Initialize feature extractors with configuration parameters.
        
        Args:
            config: Dictionary containing feature extraction parameters with additional
                   linguistic feature configurations:
                   {
                       'tfidf': {...},
                       'svd': {...},
                       'lda': {...},
                       'linguistic': {
                           'use_pos': True,
                           'use_dep': True,
                           'use_ner': True
                       }
                   }
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
        
        # Initialize spaCy
        self.nlp = spacy.load('en_core_web_sm')
        
        # Feature names for linguistic features
        self.linguistic_feature_names = None

    def _extract_linguistic_features(self, texts):
        """Extract linguistic features from texts"""
        features_list = []
        feature_names = set()
        
        for text in texts:
            doc = self.nlp(text)
            features = {}
            
            if self.config['linguistic'].get('use_pos', True):
                # POS features
                pos_counts = defaultdict(int)
                for token in doc:
                    pos_counts[f'pos_{token.pos_}'] += 1
                total_tokens = len(doc)
                for pos, count in pos_counts.items():
                    features[pos] = count / total_tokens if total_tokens > 0 else 0
                
                # Special POS ratios
                features['pos_adj_adv_ratio'] = (
                    (pos_counts['pos_ADJ'] + pos_counts['pos_ADV']) / total_tokens 
                    if total_tokens > 0 else 0
                )
            
            if self.config['linguistic'].get('use_dep', True):
                # Dependency features
                features.update({
                    'dep_has_negation': int(any(token.dep_ == 'neg' for token in doc)),
                    'dep_root_verbs': len([token for token in doc if token.dep_ == 'ROOT' and token.pos_ == 'VERB']),
                    'dep_avg_subtree': np.mean([len(list(token.subtree)) for token in doc]) if len(doc) > 0 else 0,
                })
            
            if self.config['linguistic'].get('use_ner', True):
                # NER features
                ent_counts = defaultdict(int)
                for ent in doc.ents:
                    ent_counts[f'ner_{ent.label_}'] += 1
                features.update(ent_counts)
                features['ner_density'] = len(doc.ents) / len(doc) if len(doc) > 0 else 0
            
            feature_names.update(features.keys())
            features_list.append(features)
        
        # Convert to DataFrame with consistent columns
        self.linguistic_feature_names = sorted(feature_names)
        features_df = pd.DataFrame(features_list).fillna(0)
        features_df = features_df.reindex(columns=self.linguistic_feature_names, fill_value=0)
        
        return features_df.values

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
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(texts)
        print("\nLinguistic Features:")
        print(f"• Number of Features: {linguistic_features.shape[1]}")
        print(f"• Feature Types:", end=" ")
        categories = defaultdict(int)
        for name in self.linguistic_feature_names:
            prefix = name.split('_')[0]
            categories[prefix] += 1
        print(", ".join(f"{k}: {v}" for k, v in categories.items()))
        
        # Combine all features
        combined_features = np.hstack([
            svd_features, 
            lda_features,
            linguistic_features
        ])
        
        print("\nCombined Features:")
        print(f"• SVD Features: {svd_features.shape[1]} dimensions")
        print(f"• LDA Features: {lda_features.shape[1]} dimensions")
        print(f"• Linguistic Features: {linguistic_features.shape[1]} dimensions")
        print(f"• Total Features: {combined_features.shape[1]} dimensions")
        print("-" * 40)
            
        return combined_features

    def transform(self, texts):
        """Transform new texts using fitted extractors"""
        tfidf_matrix = self.vectorizer.transform(texts)
        svd_features = self.svd.transform(tfidf_matrix)
        lda_features = self.lda.transform(tfidf_matrix)
        linguistic_features = self._extract_linguistic_features(texts)
        return np.hstack([svd_features, lda_features, linguistic_features])

    def get_feature_names(self):
        """Get names of all features in the combined feature matrix"""
        svd_names = [f'svd_{i}' for i in range(self.svd.n_components_)]
        lda_names = [f'topic_{i}' for i in range(self.lda.n_components)]
        return svd_names + lda_names + self.linguistic_feature_names