# config.py
# Text preprocessing parameters
TEXT_PREPROCESSING = {
    'min_word_length': 10,
    'max_word_length': 100,
    'min_df': 5,
    'max_df': 0.7,
}

# Feature extraction parameters
FEATURE_EXTRACTION = {
    'tfidf': {
        'max_features': 500,
        'ngram_range': (1, 2),
        'min_df': 5,
        'max_df': 0.7,
        'use_idf': True,
        'smooth_idf': True,
        'sublinear_tf': True
    },
    'svd': {
        'n_components': 100,
        'random_state': 42,
        'n_iter': 5,
        'algorithm': 'randomized'
    },
    'lda': {
        'n_components': 10,
        'max_iter': 50,
        'learning_offset': 50.,
        'random_state': 42,
        'doc_topic_prior': 0.1,
        'topic_word_prior': 0.1,
        'learning_method': 'online',
        'batch_size': 128
    }
}

# Clustering parameters
CLUSTERING = {
    'hierarchical': {
        'n_clusters': 5,
        'metric': 'euclidean',
        'linkage': 'ward',
        'compute_distances': True
    }
}

# Classification parameters
CLASSIFICATION = {
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'probability': True,
        'class_weight': 'balanced',
        'max_iter': 1000
    },
    'gaussian_process': {
        'max_iter_predict': 100,
        'random_state': 42
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