from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

class TextClustering:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.clustering = AgglomerativeClustering(
            **self.config.CLUSTERING['hierarchical']
        )

    def fit_predict(self, features):
        """Perform hierarchical clustering"""
        scaled_features = self.scaler.fit_transform(features)
        return self.clustering.fit_predict(scaled_features)