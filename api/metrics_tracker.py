"""
Metrics Tracker for ML Model Monitoring
Tracks predictions, calculates performance metrics, and detects drift
"""

import numpy as np
from collections import deque
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from prometheus_client import Gauge, Histogram

# ========================================
# ML PERFORMANCE METRICS
# ========================================
model_f1_score = Gauge(
    'model_f1_score',
    'F1 Score of the model (rolling window)'
)

model_roc_auc = Gauge(
    'model_roc_auc_score',
    'ROC-AUC Score of the model (rolling window)'
)

model_pr_auc = Gauge(
    'model_pr_auc_score',
    'Precision-Recall AUC Score (rolling window)'
)

model_precision = Gauge(
    'model_precision',
    'Precision of the model (rolling window)'
)

model_recall = Gauge(
    'model_recall',
    'Recall of the model (rolling window)'
)

# ========================================
# DRIFT DETECTION METRICS
# ========================================
feature_mean_drift = Gauge(
    'feature_mean_drift',
    'Mean value drift from training baseline',
    ['feature_index']
)

feature_std_drift = Gauge(
    'feature_std_drift',
    'Standard deviation drift from training baseline',
    ['feature_index']
)

prediction_drift_score = Gauge(
    'prediction_drift_score',
    'Overall prediction distribution drift score'
)

# ========================================
# FAIRNESS METRICS
# ========================================
prediction_confidence_by_diagnosis = Histogram(
    'prediction_confidence_by_diagnosis',
    'Confidence distribution by diagnosis',
    ['diagnosis'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

false_positive_rate = Gauge(
    'false_positive_rate',
    'False positive rate (requires ground truth)'
)

false_negative_rate = Gauge(
    'false_negative_rate',
    'False negative rate (requires ground truth)'
)


class ModelMetricsTracker: 
    """
    Tracks model predictions and calculates performance metrics
    
    Features:
    - Rolling window for recent predictions
    - ML metrics (F1, ROC-AUC, PR-AUC, Precision, Recall)
    - Drift detection
    - Fairness monitoring
    """
    
    def __init__(self, window_size=100):
        """
        Initialize metrics tracker
        
        Args: 
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.true_labels = deque(maxlen=window_size)
        self.feature_baseline = None
        self.malignant_count = 0
        self.benign_count = 0
        
        print(f"📊 ModelMetricsTracker initialized with window_size={window_size}")
    
    def set_feature_baseline(self, means, stds):
        """
        Set baseline feature statistics from training data
        
        Args: 
            means: Array of mean values for each feature
            stds: Array of standard deviations for each feature
        """
        self.feature_baseline = {
            'means': np.array(means),
            'stds': np.array(stds)
        }
        print(f"✅ Feature baseline set:  {len(means)} features")
    
    def add_prediction(self, pred, prob, features=None):
        """
        Add a new prediction to the tracker
        
        Args:
            pred: Predicted class (0 or 1)
            prob: Prediction probability
            features: Feature vector (optional, for drift detection)
        """
        self.predictions.append(pred)
        self.probabilities.append(prob)
        
        # Update diagnosis counts
        if pred == 1:
            self.malignant_count += 1
        else:
            self.benign_count += 1
        
        # Update drift metrics if baseline exists
        if features is not None and self.feature_baseline is not None:
            self._update_drift_metrics(features)
    
    def add_true_label(self, true_label):
        """
        Add ground truth label (for validation/feedback)
        
        Args: 
            true_label: True class (0 or 1)
        """
        self.true_labels.append(true_label)
    
    def _update_drift_metrics(self, features):
        """
        Calculate feature drift from baseline
        
        Args: 
            features: Feature vector
        """
        if self.feature_baseline is None:
            return
        
        features_array = np.array(features).reshape(1, -1)
        baseline_means = self.feature_baseline['means']
        baseline_stds = self.feature_baseline['stds']
        
        # Track drift for first 10 features (to avoid metric explosion)
        n_features_to_track = min(len(features), 10)
        
        for i in range(n_features_to_track):
            if i >= len(baseline_means):
                continue
            
            baseline_mean = baseline_means[i]
            baseline_std = baseline_stds[i]
            current_value = features_array[0, i]
            
            # Calculate drift as standardized difference (z-score)
            if baseline_std > 0:
                drift = abs(current_value - baseline_mean) / baseline_std
                feature_mean_drift.labels(feature_index=str(i)).set(float(drift))
    
    def calculate_metrics(self):
        """
        Calculate all ML performance metrics from rolling window
        """
        if len(self.predictions) < 10:
            # Not enough data yet
            return
        
        preds = np.array(list(self.predictions))
        probs = np.array(list(self.probabilities))
        
        # ===== METRICS WITH GROUND TRUTH =====
        if len(self.true_labels) >= 10:
            y_true = np.array(list(self.true_labels))
            y_pred = preds[-len(y_true):]
            y_prob = probs[-len(y_true):]
            
            # F1 Score
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                model_f1_score.set(float(f1))
            except Exception as e:
                print(f"⚠️ Error calculating F1: {e}")
            
            # ROC-AUC & PR-AUC
            if len(np.unique(y_true)) > 1:
                try: 
                    roc_auc = roc_auc_score(y_true, y_prob)
                    model_roc_auc.set(float(roc_auc))
                    
                    # Precision-Recall AUC
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    pr_auc = auc(recall, precision)
                    model_pr_auc.set(float(pr_auc))
                except Exception as e:
                    print(f"⚠️ Error calculating AUC metrics: {e}")
            
            # Precision & Recall
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            model_precision.set(float(precision))
            model_recall.set(float(recall))
            
            # False rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            false_positive_rate. set(float(fpr))
            false_negative_rate.set(float(fnr))
        
        # ===== PREDICTION DRIFT =====
        # Track distribution of predictions over time
        current_malignant_rate = float(np.mean(preds))
        prediction_drift_score.set(current_malignant_rate)
    
    def get_stats(self):
        """
        Get current statistics
        
        Returns:
            dict: Current statistics
        """
        return {
            'total_predictions': self.malignant_count + self.benign_count,
            'malignant_count': self.malignant_count,
            'benign_count': self.benign_count,
            'ratio':  self.malignant_count / self.benign_count if self.benign_count > 0 else 0,
            'window_size': len(self.predictions),
            'has_baseline': self.feature_baseline is not None,
            'has_ground_truth': len(self.true_labels) > 0
        }
    
    def reset_counts(self):
        """Reset cumulative counts (predictions deque is not affected)"""
        self.malignant_count = 0
        self.benign_count = 0
        print("🔄 Cumulative counts reset")