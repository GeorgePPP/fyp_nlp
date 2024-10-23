# config.py
# Feature extraction parameters
MAX_FEATURES = 1000
N_COMPONENTS_SVD = 100
N_TOPICS_LDA = 10

# Clustering parameters
N_CLUSTERS = 5

# Classification parameters
SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'probability': True,
    'random_state': 42
}

# Cross-validation parameters
CV_FOLDS = 5