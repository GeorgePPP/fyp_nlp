# config.py
TEXT_PREPROCESSING = {
}
# Feature extraction parameters
FEATURE_EXTRACTION = {
    'tfidf': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.9,
        'use_idf': True,
    },
    'svd': {
        'n_components': 750,
        'random_state': 42,
    },
    'lda': {
        'n_components': 30,
        'random_state': 42,
    },
    'linguistic': {
        'use_pos': True,
        'use_dep': True,
        'use_ner': True}
}

# Clustering parameters
CLUSTERING = {
    'hierarchical': {
        'n_clusters': 30,
        'metric': 'euclidean',
        'linkage': 'ward',
        'compute_distances': True
    }
}

# Classification parameters
CLASSIFICATION = {
    'svm': {
        'C': 1.0,
        'kernel': 'linear',
        'probability': True,
        'class_weight': 'balanced',
        'max_iter': 1000
    }
}

# Cross-validation parameters
CROSS_VALIDATION = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Output configuration
OUTPUT = {
    'save_models': True,
    'save_predictions': True,
    'save_metrics': True,
    'output_dir': 'outputs',
    'models_dir': 'outputs/models',
    'metrics_dir': 'outputs/metrics',
    'predictions_dir': 'outputs/predictions'
}