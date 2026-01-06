"""
Metrics Tracker for ML Model Monitoring
- Model validation metrics (from test/validation set)
- Production metrics (from live predictions)
"""

import numpy as np
from collections import deque
from prometheus_client import Gauge, Histogram, Info

# ========================================
# MODEL VALIDATION METRICS (STATIC)
# ========================================
# These are loaded once from the model's validation results

model_f1_score_validation = Gauge(
    'model_f1_score_validation',
    'F1 Score from validation/test set'
)

model_accuracy_validation = Gauge(
    'model_accuracy_validation',
    'Accuracy from validation/test set'
)

model_precision_validation = Gauge(
    'model_precision_validation',
    'Precision from validation/test set'
)

model_recall_validation = Gauge(
    'model_recall_validation',
    'Recall from validation/test set'
)

model_specificity_validation = Gauge(
    'model_specificity_validation',
    'Specificity (TNR) from validation/test set'
)

model_roc_auc_validation = Gauge(
    'model_roc_auc_validation',
    'ROC-AUC Score from validation/test set'
)

model_pr_auc_validation = Gauge(
    'model_pr_auc_validation',
    'Precision-Recall AUC from validation/test set'
)

# Confusion matrix components
model_true_positives = Gauge(
    'model_true_positives_validation',
    'True Positives from validation set'
)

model_true_negatives = Gauge(
    'model_true_negatives_validation',
    'True Negatives from validation set'
)

model_false_positives = Gauge(
    'model_false_positives_validation',
    'False Positives from validation set'
)

model_false_negatives = Gauge(
    'model_false_negatives_validation',
    'False Negatives from validation set'
)

# Model info
model_info = Info(
    'model_info',
    'Model metadata and version information'
)

# ========================================
# PRODUCTION METRICS (DYNAMIC)
# ========================================

# Drift detection
feature_mean_drift = Gauge(
    'feature_mean_drift',
    'Mean value drift from training baseline',
    ['feature_index']
)

prediction_drift_score = Gauge(
    'prediction_drift_score',
    'Malignant prediction rate in production (distribution drift)'
)

confidence_drift_score = Gauge(
    'confidence_drift_score',
    'Average prediction confidence in production'
)

# Confidence by diagnosis
prediction_confidence_by_diagnosis = Histogram(
    'prediction_confidence_by_diagnosis',
    'Confidence distribution by diagnosis in production',
    ['diagnosis'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

avg_confidence_benign = Gauge(
    'avg_confidence_benign_production',
    'Average confidence for Benign predictions in production'
)

avg_confidence_malignant = Gauge(
    'avg_confidence_malignant_production',
    'Average confidence for Malignant predictions in production'
)

# Production distribution
production_malignant_rate = Gauge(
    'production_malignant_rate',
    'Percentage of malignant predictions in production'
)

production_benign_rate = Gauge(
    'production_benign_rate',
    'Percentage of benign predictions in production'
)


class ModelMetricsTracker:
    """
    Tracks both validation and production metrics
    
    Validation metrics:  Loaded once from model evaluation
    Production metrics: Updated with each prediction
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # Production data
        self.predictions = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.feature_baseline = None
        
        # Counters
        self.malignant_count = 0
        self.benign_count = 0
        
        # Confidence tracking
        self.benign_confidences = deque(maxlen=window_size)
        self.malignant_confidences = deque(maxlen=window_size)
        
        print(f"📊 ModelMetricsTracker initialized (window_size={window_size})")
    
    # ========================================
    # VALIDATION METRICS (load once)
    # ========================================
    
    def load_validation_metrics(self, metrics_dict):
        """
        Load validation metrics from model evaluation
        
        Args:
            metrics_dict: Dictionary with validation metrics
        """
        try:
            # Helper function to safely set gauge
            def safe_set_gauge(gauge, value, name):
                try:
                    if value is not None:
                        float_value = float(value)
                        if not np.isnan(float_value) and not np.isinf(float_value):
                            gauge.set(float_value)
                            return True
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Could not set {name}:  {e}")
                return False
            
            # Performance metrics
            if 'f1_score' in metrics_dict:
                safe_set_gauge(model_f1_score_validation, metrics_dict['f1_score'], 'F1 Score')
            
            if 'accuracy' in metrics_dict:
                safe_set_gauge(model_accuracy_validation, metrics_dict['accuracy'], 'Accuracy')
            
            if 'precision' in metrics_dict:
                safe_set_gauge(model_precision_validation, metrics_dict['precision'], 'Precision')
            
            if 'recall' in metrics_dict:
                safe_set_gauge(model_recall_validation, metrics_dict['recall'], 'Recall')
            
            if 'specificity' in metrics_dict: 
                safe_set_gauge(model_specificity_validation, metrics_dict['specificity'], 'Specificity')
            
            if 'roc_auc' in metrics_dict:
                safe_set_gauge(model_roc_auc_validation, metrics_dict['roc_auc'], 'ROC-AUC')
            
            if 'pr_auc' in metrics_dict:
                safe_set_gauge(model_pr_auc_validation, metrics_dict['pr_auc'], 'PR-AUC')
            
            # Confusion matrix
            if 'confusion_matrix' in metrics_dict:
                cm = metrics_dict['confusion_matrix']
                model_true_positives.set(int(cm.get('tp', 0)))
                model_true_negatives.set(int(cm.get('tn', 0)))
                model_false_positives.set(int(cm.get('fp', 0)))
                model_false_negatives.set(int(cm.get('fn', 0)))
            
            # Model info - MUST convert all values to strings
            if 'model_info' in metrics_dict:
                try:
                    info_dict = {}
                    for k, v in metrics_dict['model_info'].items():
                        # Convert everything to string
                        if v is None:
                            info_dict[k] = 'unknown'
                        else:
                            info_dict[k] = str(v)
                    
                    model_info.info(info_dict)
                    print(f"✅ Model info set:  {list(info_dict.keys())}")
                
                except Exception as e:
                    print(f"⚠️ Could not set model_info:  {e}")
            
            print("✅ Validation metrics loaded successfully")
            print(f"   F1: {metrics_dict.get('f1_score', 'N/A')}")
            print(f"   Accuracy: {metrics_dict.get('accuracy', 'N/A')}")
            print(f"   ROC-AUC: {metrics_dict.get('roc_auc', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ Error loading validation metrics: {e}")
            import traceback
            print(traceback.format_exc())
    
    # ========================================
    # PRODUCTION METRICS (update continuously)
    # ========================================
    
    def set_feature_baseline(self, means, stds):
        """Set baseline feature statistics from training data"""
        self.feature_baseline = {
            'means': np.array(means),
            'stds': np.array(stds)
        }
        print(f"✅ Feature baseline set: {len(means)} features")
    
    def add_prediction(self, pred, prob, features=None):
        """Add a production prediction"""
        self.predictions.append(pred)
        self.probabilities.append(prob)
        
        # Track by diagnosis
        if pred == 1: 
            self.malignant_count += 1
            self.malignant_confidences.append(prob)
        else:
            self.benign_count += 1
            self.benign_confidences.append(prob)
        
        # Update drift
        if features is not None and self.feature_baseline is not None:
            self._update_drift_metrics(features)
    
    def _update_drift_metrics(self, features):
        """Calculate feature drift from baseline"""
        if self.feature_baseline is None:
            return
        
        features_array = np.array(features).reshape(1, -1)
        baseline_means = self.feature_baseline['means']
        baseline_stds = self.feature_baseline['stds']
        
        # Track first 10 features
        n_features = min(len(features), 10)
        
        for i in range(n_features):
            if i >= len(baseline_means):
                continue
            
            baseline_mean = baseline_means[i]
            baseline_std = baseline_stds[i]
            current_value = features_array[0, i]
            
            if baseline_std > 0:
                drift = abs(current_value - baseline_mean) / baseline_std
                feature_mean_drift.labels(feature_index=str(i)).set(float(drift))
    
    def calculate_metrics(self):
        """Calculate production metrics from rolling window"""
        if len(self.predictions) < 10:
            return
        
        preds = np.array(list(self.predictions))
        probs = np.array(list(self.probabilities))
        
        # Prediction distribution
        malignant_rate = float(np.mean(preds))
        benign_rate = 1.0 - malignant_rate
        
        prediction_drift_score.set(malignant_rate)
        production_malignant_rate.set(malignant_rate * 100)  # Percentage
        production_benign_rate.set(benign_rate * 100)
        
        # Confidence metrics
        avg_confidence = float(np.mean(probs))
        confidence_drift_score.set(avg_confidence)
        
        # Confidence by diagnosis
        if len(self.benign_confidences) > 0:
            avg_conf_benign = float(np.mean(list(self.benign_confidences)))
            avg_confidence_benign.set(avg_conf_benign)
        
        if len(self.malignant_confidences) > 0:
            avg_conf_malignant = float(np.mean(list(self.malignant_confidences)))
            avg_confidence_malignant.set(avg_conf_malignant)
    
    def get_stats(self):
        """Get current production statistics"""
        total = self.malignant_count + self.benign_count
        return {
            'total_predictions':  total,
            'malignant_count': self.malignant_count,
            'benign_count': self.benign_count,
            'malignant_rate': (self.malignant_count / total * 100) if total > 0 else 0,
            'benign_rate': (self.benign_count / total * 100) if total > 0 else 0,
            'window_size': len(self.predictions),
            'has_baseline': self.feature_baseline is not None,
            'avg_confidence':  float(np.mean(list(self.probabilities))) if len(self.probabilities) > 0 else 0
        }