# main.py
import pandas as pd
import os
import numpy as np
import logging
from datetime import datetime
from preprocessing.text_processor import TextProcessor
from features.nlp_features import FeatureExtractor
from models.clustering import TextClustering
from models.classification import SentimentClassifier
from utils.logging_utils import LoggerSetup, LoggingDecorators
import config

# Setup logging
log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d'))
log_file = os.path.join(log_dir, 'nlp_analysis.log')
logger = LoggerSetup.setup_logger(log_file)

class NLPPipeline:
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor(config.TEXT_PREPROCESSING)
        self.feature_extractor = FeatureExtractor(config.FEATURE_EXTRACTION)
        self.logger = logging.getLogger('NLPAnalysis')

    @LoggingDecorators.log_step("Data Loading", logger)
    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    @LoggingDecorators.log_step("Text Preprocessing", logger)
    def preprocess_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Text'] = df['Text'].astype(str)
        df['processed_text'] = df['Text'].fillna('').apply(
            self.text_processor.preprocess
        )
        return df

    @LoggingDecorators.log_step("Feature Extraction", logger)
    def extract_features(self, texts: pd.Series) -> np.ndarray:
        return self.feature_extractor.fit_transform(texts)

    @LoggingDecorators.log_step("Clustering Analysis", logger)
    def perform_clustering(self, features: np.ndarray) -> np.ndarray:
        clustering = TextClustering(self.config)
        return clustering.fit_predict(features)

    @LoggingDecorators.log_step("Sentiment Classification", logger)
    def train_classifiers(self, features: np.ndarray, labels: pd.Series) -> dict:
        results = {}
        
        # Train and evaluate SVM
        svm_classifier = SentimentClassifier(
            self.config, classifier_type='svm'
        )
        results['svm'] = svm_classifier.evaluate(features, labels)
        
        return results

    @LoggingDecorators.log_function(logger)
    def run_pipeline(self, file_path: str) -> None:
        try:
            df = self.load_data(file_path)
            df = self.preprocess_texts(df)

            # Downsample Sentiment col with label 0 
            df = pd.concat([df[df['Sentiment'] != 0], df[df['Sentiment'] == 0].sample(n=500, random_state=42)])
            features = self.extract_features(df['processed_text'])
            df['cluster']  = self.perform_clustering(features)
            classification_results = self.train_classifiers(
                features, df['Sentiment']
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

def main():
    try:
        # Initialize pipeline
        pipeline = NLPPipeline(config)
        
        # Run pipeline
        pipeline.run_pipeline('../dataset.csv')
        
    except Exception as e:
        logger.error("Main execution failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()