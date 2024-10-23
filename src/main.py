# main.py
import pandas as pd
from preprocessing.text_processor import TextProcessor
from features.nlp_features import FeatureExtractor
from models.clustering import TextClustering
from models.classification import SentimentClassifier

def main():
    # Load data
    df = pd.read_csv('dataset.csv')
    print("Dataset shape:", df.shape)

    # Initialize processors
    text_processor = TextProcessor()
    feature_extractor = FeatureExtractor()
    
    # Preprocess text
    print("Preprocessing texts...")
    df['processed_text'] = df['Text'].fillna('').apply(text_processor.preprocess)
    
    # Extract features
    print("Extracting features...")
    features = feature_extractor.fit_transform(df['processed_text'])
    
    # Perform clustering
    print("Performing clustering...")
    clustering = TextClustering(n_clusters=5)
    cluster_labels = clustering.fit_predict(features)
    df['cluster'] = cluster_labels
    
    # Train and evaluate sentiment classifiers
    print("Training classifiers...")
    # SVM Classifier
    svm_classifier = SentimentClassifier(classifier_type='svm')
    svm_score, svm_std = svm_classifier.evaluate(features, df['Sentiment'])
    print(f"SVM F1-Score: {svm_score:.3f} (+/- {svm_std:.3f})")
    
    # Gaussian Process Classifier
    gp_classifier = SentimentClassifier(classifier_type='gp')
    gp_score, gp_std = gp_classifier.evaluate(features, df['Sentiment'])
    print(f"Gaussian Process F1-Score: {gp_score:.3f} (+/- {gp_std:.3f})")

if __name__ == "__main__":
    main()