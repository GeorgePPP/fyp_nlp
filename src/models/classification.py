from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

class SentimentClassifier:
    def __init__(self, classifier_type='svm'):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
        else:  # gaussian process
            kernel = 1.0 * RBF(1.0)
            self.classifier = GaussianProcessClassifier(
                kernel=kernel,
                random_state=42
            )

    def fit(self, features, labels):
        """Train the classifier"""
        X_scaled = self.scaler.fit_transform(features)
        y_encoded = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X_scaled, y_encoded)

    def predict(self, features):
        """Predict sentiment labels"""
        X_scaled = self.scaler.transform(features)
        y_pred = self.classifier.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def evaluate(self, features, labels, cv=5):
        """Evaluate classifier using cross-validation"""
        X_scaled = self.scaler.fit_transform(features)
        y_encoded = self.label_encoder.fit_transform(labels)
        scores = cross_val_score(
            self.classifier, X_scaled, y_encoded, cv=cv, scoring='f1_weighted'
        )
        return scores.mean(), scores.std()
