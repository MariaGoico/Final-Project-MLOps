import os
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from optuna.integration.mlflow import MLflowCallback
import matplotlib.pyplot as plt
import seaborn as sns
from logic.data_module import StrokeDataProcessor
from logic.utilities import DataValidator


class XGBoostStrokeClassifier:
    """
    XGBoost classifier for stroke prediction with Optuna optimization and MLflow tracking
    """
    
    def __init__(self, data_path, tracking_uri=None):
        self.data_path = data_path
        self.processor = StrokeDataProcessor()
        self.validator = DataValidator()
        self.best_model = None
        self.best_threshold = 0.5
        
        # Set MLflow tracking URI
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("stroke_prediction")
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        X, y = self.processor.load_data(self.data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = self.validator.split_train_test(X, y)
        
        # Preprocess
        X_train_processed = self.processor.fit_transform(X_train)
        X_test_processed = self.processor.transform(X_test)
        
        # Calculate scale_pos_weight for imbalanced data
        self.scale_pos_weight = self.validator.get_scale_pos_weight(y_train)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def objective(self, trial, X_train, y_train, cv):
        """
        Optuna objective function for hyperparameter optimization
        Uses threshold-independent metrics (ROC-AUC, PR-AUC) for optimization
        """
        # Suggest hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Cross-validation scores
        roc_auc_scores = []
        pr_auc_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Predict probabilities
            y_pred_proba = model.predict(dval)
            
            # Calculate threshold-independent metrics
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            
            roc_auc_scores.append(roc_auc)
            pr_auc_scores.append(pr_auc)
        
        # Average scores across folds
        mean_roc_auc = np.mean(roc_auc_scores)
        mean_pr_auc = np.mean(pr_auc_scores)
        
        # Log metrics to MLflow (child run)
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", mean_roc_auc)
            mlflow.log_metric("cv_pr_auc", mean_pr_auc)
            mlflow.log_metric("cv_roc_auc_std", np.std(roc_auc_scores))
            mlflow.log_metric("cv_pr_auc_std", np.std(pr_auc_scores))
        
        # Return metric to optimize (maximize ROC-AUC)
        return mean_roc_auc
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """
        Find optimal threshold using threshold-dependent metrics
        Optimizes F1 score for production use
        """
        y_pred_proba = model.predict(xgb.DMatrix(X_val))
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_f1, thresholds, f1_scores
    
    def plot_feature_importance(self, model, save_path='feature_importance.png'):
        """Plot and save feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model, ax=ax, max_num_features=15)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def plot_threshold_analysis(self, thresholds, f1_scores, save_path='threshold_analysis.png'):
        """Plot threshold vs F1 score"""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores)
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('Threshold vs F1 Score')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def train_and_optimize(self, n_trials=100):
        """
        Main training loop with Optuna optimization and MLflow tracking
        """
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # Prepare cross-validation
        cv = self.validator.get_stratified_kfold(n_splits=5)
        
        # Start MLflow parent run
        with mlflow.start_run(run_name="xgboost_optimization"):
            
            # Log dataset information
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_0_count", np.sum(y_train == 0))
            mlflow.log_param("class_1_count", np.sum(y_train == 1))
            mlflow.log_param("scale_pos_weight", self.scale_pos_weight)
            
            # Optuna optimization
            study = optuna.create_study(
                direction='maximize',
                study_name='xgboost_stroke_optimization'
            )
            
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, cv),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            # Log best trial
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_cv_roc_auc", study.best_value)
            
            # Train final model with best parameters
            best_params = study.best_params
            best_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': self.scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1
            })
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            self.best_model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtest, 'test')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Find optimal threshold
            self.best_threshold, best_f1, thresholds, f1_scores = \
                self.find_optimal_threshold(self.best_model, X_test, y_test)
            
            mlflow.log_param("optimal_threshold", self.best_threshold)
            
            # Final predictions
            y_pred_proba = self.best_model.predict(dtest)
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)
            
            # Log final metrics
            mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba))
            mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_pred_proba))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
            
            # Generate and log artifacts
            # 1. Feature importance
            feat_imp_path = self.plot_feature_importance(self.best_model)
            mlflow.log_artifact(feat_imp_path)
            
            # 2. Threshold analysis
            threshold_path = self.plot_threshold_analysis(thresholds, f1_scores)
            mlflow.log_artifact(threshold_path)
            
            # 3. Confusion matrix
            cm_path = self.plot_confusion_matrix(y_test, y_pred)
            mlflow.log_artifact(cm_path)
            
            # 4. Classification report
            report = classification_report(y_test, y_pred)
            with open('classification_report.txt', 'w') as f:
                f.write(report)
            mlflow.log_artifact('classification_report.txt')
            
            # Log model
            mlflow.xgboost.log_model(self.best_model, "model")
            
            # Save preprocessor
            self.processor.save('models/preprocessor.pkl')
            mlflow.log_artifact('models/preprocessor.pkl')
            
            print(f"\nTraining complete!")
            print(f"Best ROC-AUC: {study.best_value:.4f}")
            print(f"Optimal threshold: {self.best_threshold:.4f}")
            print(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")
            
        return self.best_model, self.best_threshold


if __name__ == "__main__":
    classifier = XGBoostStrokeClassifier(
        data_path='stroke_data.csv'
    )
    model, threshold = classifier.train_and_optimize(n_trials=50)