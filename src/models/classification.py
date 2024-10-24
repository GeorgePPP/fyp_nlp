from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import time

class SentimentClassifier:
    """Simplified sentiment classifier with progress tracking"""
    
    def __init__(self, config, classifier_type='svm'):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier_type = classifier_type
        
        print(f"Initializing {classifier_type.upper()} classifier...")
        if classifier_type == 'svm':
            self.classifier = SVC(**config.CLASSIFICATION['svm'])
        else:  # gaussian process
            kernel = ConstantKernel(1.0) * RBF([1.0])
            gp_config = config.CLASSIFICATION['gaussian_process'].copy()
            gp_config['kernel'] = kernel
            self.classifier = GaussianProcessClassifier(**gp_config)

    def fit(self, features, labels):
        """Train the classifier with progress tracking"""
        print("\nStarting training process:")
        print("1. Scaling features...")
        start_time = time.time()
        X_scaled = self.scaler.fit_transform(features)
        
        print("2. Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"3. Training {self.classifier_type.upper()} classifier...")
        self.classifier.fit(X_scaled, y_encoded)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Number of samples processed: {len(features)}")

    def predict(self, features):
        """Predict sentiment labels with progress tracking"""
        print("\nStarting prediction process:")
        start_time = time.time()
        
        print("1. Scaling features...")
        X_scaled = self.scaler.transform(features)
        
        print("2. Making predictions...")
        y_pred = self.classifier.predict(X_scaled)
        y_prob = self.classifier.predict_proba(X_scaled)
        
        print("3. Converting predictions back to original labels...")
        final_predictions = self.label_encoder.inverse_transform(y_pred)
        
        prediction_time = time.time() - start_time
        print(f"\nPrediction completed in {prediction_time:.2f} seconds")
        print(f"Number of samples predicted: {len(features)}")
        
        return final_predictions, y_prob

    def evaluate(self, features, labels):
        """Model evaluation with progress tracking and detailed metrics"""
        print("\nStarting evaluation process:")
        start_time = time.time()
        
        print("1. Scaling features...")
        X_scaled = self.scaler.fit_transform(features)
        
        print("2. Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Cross-validation setup
        cv = StratifiedKFold(**self.config.CROSS_VALIDATION)
        n_splits = cv.get_n_splits()
        confusion_matrices = []
        metrics_per_fold = []
        
        print(f"\n3. Performing {n_splits}-fold cross-validation:")
        # Using tqdm for progress bar
        for fold, (train_idx, test_idx) in enumerate(
            tqdm(cv.split(X_scaled, y_encoded), 
                 total=n_splits, 
                 desc="Cross-validation progress"
            ), 1
        ):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            # Train model with increased max_iter for SVM
            if isinstance(self.classifier, SVC):
                self.classifier.set_params(max_iter=5000)
                
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate TP, FP, TN, FN for each class and sum them
            n_classes = cm.shape[0]
            tp = np.sum(np.diag(cm))  # Sum of diagonal elements
            fp = np.sum(cm) - tp  # Sum of all elements minus TP
            fn = fp  # In multi-class, FP and FN are the same
            tn = np.sum(cm) * (n_classes - 1) - fp  # All possible negative cases minus FP
            
            metrics_per_fold.append({
                'tp': tp, 
                'fp': fp, 
                'tn': tn, 
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Average confusion matrix and metrics across folds
        avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in metrics_per_fold])
            for metric in metrics_per_fold[0].keys()
        }
        
        eval_time = time.time() - start_time

        print(f"\nEvaluation completed in {eval_time:.2f} seconds")
        print(f"Number of samples processed: {len(features)}")
        print(f"Class labels: {labels.value_counts()}")
        print(f"Average confusion matrix shape: {avg_confusion_matrix.shape}")
        
        # Print averaged metrics
        print("\nAverage Metrics Across Folds:")
        print(f"True Positives (TP): {avg_metrics['tp']:.2f}")
        print(f"False Positives (FP): {avg_metrics['fp']:.2f}")
        print(f"True Negatives (TN): {avg_metrics['tn']:.2f}")
        print(f"False Negatives (FN): {avg_metrics['fn']:.2f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall: {avg_metrics['recall']:.4f}")
        print(f"F1 Score: {avg_metrics['f1']:.4f}")
        
        return {
            'confusion_matrix': avg_confusion_matrix,
            'metrics': avg_metrics
        }