"""
Metrics Tracker for ML Model Monitoring
- Model validation metrics (from test/validation set)
- Production metrics (from live predictions)
"""

import time
import random
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
# ENHANCED DRIFT DETECTION METRICS
# ========================================

# Data Drift (Feature Distribution)
data_drift_detected = Gauge(
    'data_drift_detected',
    'Binary flag:  1 if data drift detected, 0 otherwise'
)

data_drift_score = Gauge(
    'data_drift_score',
    'Kolmogorov-Smirnov statistic aggregated across features (0-1, higher = more drift)'
)

features_drifted_count = Gauge(
    'features_drifted_count',
    'Number of features with significant drift (z-score > 3)'
)

# Concept Drift (X -> Y relationship)
concept_drift_detected = Gauge(
    'concept_drift_detected',
    'Binary flag: 1 if concept drift detected, 0 otherwise'
)

concept_drift_score = Gauge(
    'concept_drift_score',
    'Concept drift score based on prediction flip rate (0-1)'
)

prediction_flip_rate = Gauge(
    'prediction_flip_rate',
    'Rate of prediction changes for similar inputs (concept drift indicator)'
)

# Fairness Issues
fairness_issue_detected = Gauge(
    'fairness_issue_detected',
    'Binary flag: 1 if fairness issue detected, 0 otherwise'
)

confidence_disparity_score = Gauge(
    'confidence_disparity_score',
    'Absolute difference in confidence between Benign and Malignant predictions'
)

prediction_imbalance_score = Gauge(
    'prediction_imbalance_score',
    'Deviation from expected class balance (0-1, higher = more imbalance)'
)

# Drift Simulation Toggle
drift_simulation_enabled = Gauge(
    'drift_simulation_enabled',
    'Flag indicating if drift simulation is active (1=yes, 0=no)'
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
    
    def __init__(self, window_size=1000, enable_simulation=False):
        self.window_size = window_size
        self.enable_simulation = enable_simulation  # ← AÑADIR ESTO

        # Production data
        self.predictions = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.features_history = deque(maxlen=window_size)  # NEW:  store features
        self.feature_baseline = None
        
        # Counters
        self.malignant_count = 0
        self.benign_count = 0
        
        # Confidence tracking
        self.benign_confidences = deque(maxlen=window_size)
        self.malignant_confidences = deque(maxlen=window_size)

        # Drift detection state
        self.baseline_malignant_rate = 0.372  # From training data
        self.drift_detected_count = 0
        self.concept_drift_detected_count = 0
        self.fairness_issue_count = 0

        # Thresholds
        self.DATA_DRIFT_THRESHOLD = 3.0  # z-score threshold
        self.CONCEPT_DRIFT_THRESHOLD = 0.15  # 15% flip rate
        self.FAIRNESS_CONFIDENCE_THRESHOLD = 0.10  # 10% confidence gap
        self.FAIRNESS_IMBALANCE_THRESHOLD = 0.20  # 20% deviation from baseline
        
        drift_simulation_enabled.set(1 if enable_simulation else 0)
        
        print(f"📊 ModelMetricsTracker initialized")
        print(f"   Window size: {window_size}")
        print(f"   Simulation:  {'ENABLED' if enable_simulation else 'DISABLED'}")
    
    
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
            'stds': np.array(stds),
            'n_features': len(means)

        }
        print(f"✅ Feature baseline set: {len(means)} features")
    
    def add_prediction(self, pred, prob, features=None):
        """Add a production prediction"""
        self.predictions.append(pred)
        self.probabilities.append(prob)
        
        # Store features for concept drift detection
        if features is not None:
            self.features_history.append(features)

        # Track by diagnosis
        if pred == 1: 
            self.malignant_count += 1
            self.malignant_confidences.append(prob)
        else:
            self.benign_count += 1
            self.benign_confidences.append(prob)

    # ========================================
    # DATA DRIFT DETECTION
    # ========================================
    
    def detect_data_drift(self):
        """
        Detect data drift using z-score analysis
        
        Returns:
            bool: True if drift detected
            float: Drift score (0-1)
        """
        if self.feature_baseline is None or len(self.features_history) < 30:
            return False, 0.0
        
        # Get recent features
        recent_features = np.array(list(self.features_history)[-100:])
        
        baseline_means = self.feature_baseline['means']
        baseline_stds = self.feature_baseline['stds']
        
        # Calculate z-scores for each feature
        z_scores = []
        drifted_count = 0
        
        for i in range(min(recent_features.shape[1], len(baseline_means))):
            recent_mean = np.mean(recent_features[:, i])
            z = abs(recent_mean - baseline_means[i]) / (baseline_stds[i] + 1e-10)
            z_scores.append(z)
            
            if z > self.DATA_DRIFT_THRESHOLD:
                drifted_count += 1
                
            # Update per-feature drift metric
            feature_mean_drift.labels(feature_index=str(i)).set(float(z))
        
        # Aggregate drift score (mean z-score, normalized)
        avg_z_score = np.mean(z_scores)
        drift_score = min(avg_z_score / 5.0, 1.0)  # Normalize to [0,1]
        
        # Drift detected if >= 3 features have z-score > threshold
        drift_detected = drifted_count >= 3
        
        # Update metrics
        data_drift_detected.set(1 if drift_detected else 0)
        data_drift_score.set(drift_score)
        features_drifted_count.set(drifted_count)
        
        if drift_detected:
            self.drift_detected_count += 1
            print(f"⚠️  DATA DRIFT DETECTED: {drifted_count} features drifted (z-score > {self.DATA_DRIFT_THRESHOLD})")
        
        return drift_detected, drift_score
    

    # ========================================
    # CONCEPT DRIFT DETECTION
    # ========================================
    
    def detect_concept_drift(self):
        """
        Detect concept drift using prediction flip rate
        
        Concept drift = relationship between X and Y changes
        We detect this by checking if similar inputs now get different predictions
        
        Returns:
            bool: True if concept drift detected
            float:  Flip rate (0-1)
        """
        if len(self.features_history) < 100 or len(self.predictions) < 100:
            return False, 0.0
        
        recent_features = np.array(list(self.features_history)[-100:])
        recent_predictions = np.array(list(self.predictions)[-100:])
        
        # Find pairs of similar inputs (Euclidean distance < threshold)
        flip_count = 0
        pair_count = 0
        similarity_threshold = 2.0  # Adjust based on scaled features
        
        for i in range(len(recent_features) - 10):
            for j in range(i + 1, min(i + 10, len(recent_features))):
                distance = np.linalg.norm(recent_features[i] - recent_features[j])
                
                if distance < similarity_threshold: 
                    pair_count += 1
                    # Check if predictions differ
                    if recent_predictions[i] != recent_predictions[j]: 
                        flip_count += 1
        
        if pair_count == 0:
            return False, 0.0
        
        flip_rate = flip_count / pair_count
        
        concept_detected = flip_rate > self.CONCEPT_DRIFT_THRESHOLD
        
        # Update metrics
        concept_drift_detected.set(1 if concept_detected else 0)
        concept_drift_score.set(flip_rate)
        prediction_flip_rate.set(flip_rate)
        
        if concept_detected:
            self.concept_drift_detected_count += 1
            print(f"⚠️  CONCEPT DRIFT DETECTED:  Flip rate = {flip_rate:.2%} (threshold: {self.CONCEPT_DRIFT_THRESHOLD:.2%})")
        
        return concept_detected, flip_rate
    
    # ========================================
    # FAIRNESS ISSUE DETECTION
    # ========================================
    
    def detect_fairness_issues(self):
        """
        Detect fairness issues: 
        1. Confidence disparity between classes
        2. Prediction imbalance (deviation from baseline distribution)
        
        Returns:
            bool: True if fairness issue detected
            dict: Fairness metrics
        """
        if len(self.benign_confidences) < 30 or len(self.malignant_confidences) < 30:
            return False, {}
        
        # 1. Confidence Disparity
        avg_benign_conf = np.mean(list(self.benign_confidences))
        avg_malignant_conf = np.mean(list(self.malignant_confidences))
        confidence_gap = abs(avg_benign_conf - avg_malignant_conf)
        
        confidence_issue = confidence_gap > self.FAIRNESS_CONFIDENCE_THRESHOLD
        
        # 2. Prediction Imbalance
        total = self.malignant_count + self.benign_count
        if total > 0:
            current_malignant_rate = self.malignant_count / total
            imbalance = abs(current_malignant_rate - self.baseline_malignant_rate)
            imbalance_issue = imbalance > self.FAIRNESS_IMBALANCE_THRESHOLD
        else:
            imbalance = 0.0
            imbalance_issue = False
        
        # Overall fairness issue
        fairness_issue = confidence_issue or imbalance_issue
        
        # Update metrics
        fairness_issue_detected.set(1 if fairness_issue else 0)
        confidence_disparity_score.set(confidence_gap)
        prediction_imbalance_score.set(imbalance)
        
        if fairness_issue:
            self.fairness_issue_count += 1
            issues = []
            if confidence_issue:
                issues.append(f"Confidence gap: {confidence_gap:.2%}")
            if imbalance_issue:
                issues.append(f"Imbalance: {imbalance:.2%}")
            print(f"⚠️  FAIRNESS ISSUE DETECTED: {', '.join(issues)}")
        
        return fairness_issue, {
            'confidence_gap': confidence_gap,
            'imbalance': imbalance,
            'confidence_issue': confidence_issue,
            'imbalance_issue': imbalance_issue
        }
    
    # ========================================
    # DRIFT SIMULATION (for testing)
    # ========================================
    
    def simulate_drift(self):
        """
        Simulate drift for testing/demonstration
        Called periodically by /metrics endpoint when Prometheus scrapes
        """
        if not self.enable_simulation:
            return
        
        current_time = time.time()
        cycle_time = int(current_time) % 900  # 15 min cycle
        
        print(f"🎭 Simulating drift...  (cycle_time={cycle_time}s)")
        
        # ========================================
        # DATA DRIFT SIMULATION (every 5 minutes)
        # ========================================
        if 120 <= cycle_time < 240:  # Active for 2 minutes every 15 min
            # Simulate feature drift
            drifted_features = random.randint(3, 7)
            for i in range(drifted_features):
                simulated_z_score = random.uniform(3.5, 6.0)
                feature_mean_drift.labels(feature_index=str(i)).set(simulated_z_score)
            
            drift_score = random.uniform(0.7, 0.95)
            data_drift_detected.set(1)
            data_drift_score.set(drift_score)
            features_drifted_count.set(drifted_features)
            
            print(f"   📊 DATA DRIFT:  {drifted_features} features, score={drift_score:.2f}")
        else:
            # Normal state
            for i in range(3):
                feature_mean_drift.labels(feature_index=str(i)).set(random.uniform(0.5, 2.5))
            
            data_drift_detected.set(0)
            data_drift_score.set(random.uniform(0.05, 0.15))
            features_drifted_count.set(0)
        
        # ========================================
        # CONCEPT DRIFT SIMULATION (every 7 minutes)
        # ========================================
        if 300 <= cycle_time < 420:  # Active for 2 minutes
            flip_rate = random.uniform(0.20, 0.35)
            concept_drift_detected.set(1)
            concept_drift_score.set(flip_rate)
            prediction_flip_rate.set(flip_rate)
            
            print(f"   🔄 CONCEPT DRIFT: flip_rate={flip_rate:.2%}")
        else:
            flip_rate = random.uniform(0.02, 0.08)
            concept_drift_detected.set(0)
            concept_drift_score.set(flip_rate)
            prediction_flip_rate.set(flip_rate)
        
        # ========================================
        # FAIRNESS ISSUE SIMULATION (every 10 minutes)
        # ========================================
        if 600 <= cycle_time < 720:  # Active for 2 minutes
            conf_gap = random.uniform(0.12, 0.20)
            imbalance = random.uniform(0.22, 0.35)
            
            fairness_issue_detected.set(1)
            confidence_disparity_score.set(conf_gap)
            prediction_imbalance_score.set(imbalance)
            
            print(f"   ⚖️  FAIRNESS ISSUE:  conf_gap={conf_gap:.2%}, imbalance={imbalance:.2%}")
        else:
            fairness_issue_detected.set(0)
            confidence_disparity_score.set(random.uniform(0.01, 0.05))
            prediction_imbalance_score.set(random.uniform(0.02, 0.08))

    # ========================================
    # CALCULATE METRICS
    # ========================================
    
    def calculate_metrics(self):
        """Calculate all production metrics including drift detection"""
        if len(self.predictions) < 10:
            return
        
        # Existing metrics calculation
        preds = np.array(list(self.predictions))
        probs = np.array(list(self.probabilities))
        
        # Prediction distribution
        malignant_rate = float(np.mean(preds))
        benign_rate = 1.0 - malignant_rate
        
        prediction_drift_score.set(malignant_rate)
        production_malignant_rate.set(malignant_rate * 100)
        production_benign_rate.set(benign_rate * 100)
        
        # Confidence metrics
        avg_confidence = float(np.mean(probs))
        confidence_drift_score.set(avg_confidence)
        
        if len(self.benign_confidences) > 0:
            avg_conf_benign = float(np.mean(list(self.benign_confidences)))
            avg_confidence_benign.set(avg_conf_benign)
        
        if len(self.malignant_confidences) > 0:
            avg_conf_malignant = float(np.mean(list(self.malignant_confidences)))
            avg_confidence_malignant.set(avg_conf_malignant)
        
        # NEW: Drift detection
        if self.enable_simulation:
            self.simulate_drift()
        else:
            self.detect_data_drift()
            self.detect_concept_drift()
            self.detect_fairness_issues()
    
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