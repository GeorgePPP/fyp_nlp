from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

class ModelMetrics:
    """Class to handle model evaluation and metrics visualization"""
    
    def __init__(self, config):
        self.config = config
        self._setup_output_dirs()

    def _setup_output_dirs(self):
        """Create output directories if they don't exist"""
        for dir_path in [
            self.config.OUTPUT['output_dir'],
            self.config.OUTPUT['models_dir'],
            self.config.OUTPUT['metrics_dir'],
            self.config.OUTPUT['predictions_dir'],
            self.config.VISUALIZATION['plots_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, y_prob, model_name):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        if self.config.METRICS['include_confusion_matrix']:
            cm = confusion_matrix(y_true, y_pred)
            self._plot_confusion_matrix(cm, model_name)
            metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        if self.config.METRICS['include_classification_report']:
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = report

        # ROC curves
        if self.config.METRICS['include_roc_curves']:
            self._plot_roc_curves(y_true, y_prob, model_name)

        # Precision-Recall curves
        if self.config.METRICS['include_precision_recall_curves']:
            self._plot_precision_recall_curves(y_true, y_prob, model_name)

        # Save metrics
        if self.config.OUTPUT['save_metrics']:
            self._save_metrics(metrics, model_name)

        return metrics

    def _plot_confusion_matrix(self, cm, model_name):
        """Plot and save confusion matrix"""
        plt.figure(figsize=self.config.VISUALIZATION['figsize'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        self._save_plot(f'confusion_matrix_{model_name}')

    def _plot_roc_curves(self, y_true, y_prob, model_name):
        """Plot and save ROC curves"""
        plt.figure(figsize=self.config.VISUALIZATION['figsize'])
        
        if y_prob.shape[1] > 2:  # Multi-class
            for i in range(y_prob.shape[1]):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
                plt.plot(fpr, tpr, label=f'Class {i}')
        else:  # Binary
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr)
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend()
        self._save_plot(f'roc_curves_{model_name}')

    def _plot_precision_recall_curves(self, y_true, y_prob, model_name):
        """Plot and save precision-recall curves"""
        plt.figure(figsize=self.config.VISUALIZATION['figsize'])
        
        if y_prob.shape[1] > 2:  # Multi-class
            for i in range(y_prob.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    (y_true == i).astype(int), y_prob[:, i]
                )
                plt.plot(recall, precision, label=f'Class {i}')
        else:  # Binary
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            plt.plot(recall, precision)
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {model_name}')
        plt.legend()
        self._save_plot(f'precision_recall_{model_name}')

    def _save_plot(self, name):
        """Save plot to file"""
        if self.config.VISUALIZATION['save_plots']:
            plt.savefig(
                os.path.join(
                    self.config.VISUALIZATION['plots_dir'],
                    f"{name}.{self.config.VISUALIZATION['plot_format']}"
                ),
                dpi=self.config.VISUALIZATION['dpi'],
                bbox_inches='tight'
            )
        plt.close()

    def _save_metrics(self, metrics, model_name):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.config.OUTPUT['metrics_dir'],
            f'metrics_{model_name}_{timestamp}.json'
        )
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value

        with open(filename, 'w') as f:
            json.dump(metrics_json, f, indent=4)