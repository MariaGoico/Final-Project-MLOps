import os
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score
)
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import json
import shap
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from logic.data_module import CancerDataProcessor
from logic.utilities import DataValidator
from logic.utilities import set_seed



class XGBoostBreastCancerClassifier:
    """
    XGBoost classifier for breast cancer prediction with Optuna optimization and MLflow tracking.
    Optimized for CI/CD pipeline integration.
    """
    
    def __init__(self, data_path, tracking_uri=None):
        self.data_path = data_path
        self.processor = CancerDataProcessor()
        self.validator = DataValidator()
        self.best_model = None
        self.best_threshold = 0.5
        self.n_features = None  
        
        # Set MLflow tracking URI (configured for CI/CD)
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("breast_cancer_predicition")
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        X, y = self.processor.load_data(self.data_path)
        
        # Split data
        X_train,X_val,X_test,y_train,y_val,y_test = self.validator.split_train_val_test(X, y)
        
        # Preprocess
        X_train = self.processor.fit_transform(X_train)
        X_val = self.processor.transform(X_val)
        X_test = self.processor.transform(X_test)
        
        # Calculate scale_pos_weight for imbalanced data
        self.scale_pos_weight = self.validator.get_scale_pos_weight(y_train)
        self.n_features = X_train.shape[1]  

        return X_train, X_val, X_test, y_train, y_val, y_test

    def plot_confusion_matrix(self, y_true, y_pred, save_path="plots/confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix (Test Set)")
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, save_path="plots/roc_curve.png"):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test Set)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()
    
    def objective(self, trial, X_train, y_train, cv):
        """
        Optuna objective function for hyperparameter optimization.
        Uses threshold-independent metrics (ROC-AUC) for optimization.
        Logs each trial as a child run in MLflow.
        """
        # Suggest hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
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
        
        # Log each trial as a child run in MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", mean_roc_auc)
            mlflow.log_metric("cv_pr_auc", mean_pr_auc)
        
        # Return threshold-independent metric to optimize (ROC-AUC)
        return mean_roc_auc
    
    def find_optimal_threshold(self, model, X_val, y_val, plot=True, save_path="plots/threshold_f1.png"):
        """
        Find optimal threshold using threshold-dependent metrics.
        Optimizes F1 score for production use.
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
        f1_scores = np.array(f1_scores)

        max_f1 = f1_scores.max()
        best_idx = np.where(f1_scores == max_f1)[0][-1]

        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        if plot:
            plt.figure()
            plt.plot(thresholds, f1_scores)
            plt.axvline(best_threshold, linestyle="--")
            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.title("Threshold vs F1 Score")
            plt.tight_layout()

            plt.savefig(save_path)
            plt.close()
        
        return best_threshold, best_f1

    def log_shap_interpretability(self, model, X_train, feature_names):
        """
        Generates and logs SHAP global interpretability artifacts to MLflow.
        """
        # Create Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # 1. Save Summary Plot (Global Importance)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
        summary_plot_path = "plots/shap_summary.png"
        plt.tight_layout()
        plt.savefig(summary_plot_path)
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(summary_plot_path)
        
        # 2. Optional: Log SHAP values as a pickle for raw data access
        # Save aggregated SHAP importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = dict(zip(feature_names, mean_abs_shap.tolist()))

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/shap_global.json", "w") as f:
            json.dump(shap_summary, f, indent=2)

        mlflow.log_artifact("artifacts/shap_global.json")

    def train_and_optimize(self, n_trials=100):
        """
        Main training loop with Optuna optimization and MLflow tracking.
        Optimized for CI/CD pipeline.
        """
        # Load data
        X_train,X_val,X_test,y_train,y_val,y_test = self.load_and_prepare_data()
        
        # Prepare cross-validation
        cv = self.validator.get_stratified_kfold(n_splits=5)

        # Features for shap values 
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Start MLflow parent run
        with mlflow.start_run(run_name="xgboost_optimization"):
            
            # Log dataset information
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_0_count", int(np.sum(y_train == 0)))
            mlflow.log_param("class_1_count", int(np.sum(y_train == 1)))
            mlflow.log_param("scale_pos_weight", float(self.scale_pos_weight))
            
            # Optuna optimization
            study = optuna.create_study(
                direction='maximize',
                study_name='xgboost_breast_cancer_optimization'
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, cv),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            # Log best hyperparameters
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_cv_roc_auc", study.best_value)
            
            # Train final model with best parameters
            best_params = study.best_params.copy()
            best_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': self.scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1
            })
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            self.best_model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            print("Generating SHAP global interpretability artifacts...")
            self.log_shap_interpretability(self.best_model, X_train, feature_names)
            
            # Find optimal threshold using threshold-dependent metrics
            self.best_threshold, _ = self.find_optimal_threshold(
                self.best_model, X_val, y_val
            )
            
            mlflow.log_param("optimal_threshold", float(self.best_threshold))
            
            # Final evaluation on TEST
            y_test_proba = self.best_model.predict(dtest)
            y_test_pred = (y_test_proba >= self.best_threshold).astype(int)


            mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))
            mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_test_proba))
            mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))

           # SERIALIZE THE MODEL 
            os.makedirs("artifacts", exist_ok=True)

            # Save XGBoost booster
            self.best_model.save_model("artifacts/model.json")

            # Save threshold
            with open("artifacts/threshold.json", "w") as f:
                json.dump({"threshold": float(self.best_threshold)}, f)

            # Save preprocessing pipeline
            with open("artifacts/preprocessor.pkl", "wb") as f:
                pickle.dump(self.processor, f)# NOT COMPLETELY SURE ABOUT THIS 

            # Optional metadata
            with open("artifacts/metadata.json", "w") as f:
                json.dump({
                    "n_features": self.n_features,
                    "model_type": "xgboost",
                    "objective": "binary:logistic"
                }, f)

            # Log feature importance as artifact for global interpretability
            importance_dict = self.best_model.get_score(importance_type='weight')
            importance_file = 'feature_importance.txt'
            with open(importance_file, 'w') as f:
                for feat, score in sorted(importance_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
                    f.write(f"{feat}: {score}\n")
            mlflow.log_artifact(importance_file)
            os.remove(importance_file)
            
            # Log the best model
            mlflow.xgboost.log_model(self.best_model, "model")
            
            # Save threshold
            os.makedirs('models', exist_ok=True)
            threshold_dict = {'threshold': self.best_threshold}
            with open('models/threshold.pkl', 'wb') as f:
                pickle.dump(threshold_dict, f)
                print('Saved threshold')
            mlflow.log_artifact('models/threshold.pkl')

            self.plot_roc_curve(y_test, y_test_proba)
            mlflow.log_artifact("plots/roc_curve.png")

            self.plot_confusion_matrix(y_test, y_test_pred)
            mlflow.log_artifact("plots/confusion_matrix.png")
            
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Best CV ROC-AUC: {study.best_value:.4f}")
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
            print(f"Test PR-AUC: {average_precision_score(y_test, y_test_proba):.4f}")
            print(f"Optimal Threshold: {self.best_threshold:.4f}")
            print(f"Test F1 Score: {f1_score(y_test, y_test_pred):.4f}")
            print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
            print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
            print(f"{'='*60}\n")
            
        return self.best_model, self.best_threshold


if __name__ == "__main__":
    set_seed() #set a fixed seed   
    classifier = XGBoostBreastCancerClassifier(
        data_path='data/data.csv'
    )
    model, threshold = classifier.train_and_optimize(n_trials=5)