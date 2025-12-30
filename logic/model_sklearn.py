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
from xgboost.callback import EarlyStopping

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from logic.data_module import CancerDataProcessor
from logic.utilities import DataValidator
from logic.utilities import set_seed


class XGBoostBreastCancerClassifier:
    """
    XGBoost classifier for breast cancer prediction with Optuna optimization and MLflow tracking.
    Uses sklearn-compatible XGBClassifier for ONNX export.
    """

    def __init__(self, data_path, tracking_uri=None):
        self.data_path = data_path
        self.processor = CancerDataProcessor()
        self.validator = DataValidator()
        self.best_model = None
        self.best_threshold = 0.5
        self.n_features = None

        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("breast_cancer_predicition")

    def load_and_prepare_data(self):
        X, y = self.processor.load_data(self.data_path)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.validator.split_train_val_test(X, y)

        X_train = self.processor.fit_transform(X_train)
        X_val = self.processor.transform(X_val)
        X_test = self.processor.transform(X_test)

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
        """Optuna objective using sklearn XGBClassifier"""

        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "reg_lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        roc_auc_scores = []
        pr_auc_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(**params)

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            y_proba = model.predict_proba(X_va)[:, 1]

            roc_auc_scores.append(roc_auc_score(y_va, y_proba))
            pr_auc_scores.append(average_precision_score(y_va, y_proba))

        mean_roc_auc = float(np.mean(roc_auc_scores))
        mean_pr_auc = float(np.mean(pr_auc_scores))

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_roc_auc", mean_roc_auc)
            mlflow.log_metric("cv_pr_auc", mean_pr_auc)

        return mean_roc_auc

    def find_optimal_threshold(self, model, X_val, y_val, plot=True, save_path="plots/threshold_f1.png"):
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []

        for t in thresholds:
            preds = (y_pred_proba >= t).astype(int)
            f1_scores.append(f1_score(y_val, preds))

        f1_scores = np.array(f1_scores)
        best_idx = np.argmax(f1_scores)

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

    def save_onnx_model(self):
        """Export sklearn XGBClassifier to ONNX"""
        initial_type = [("input", FloatTensorType([None, self.n_features]))]

        onnx_model = convert_sklearn(
            self.best_model,
            initial_types=initial_type
        )

        os.makedirs("models", exist_ok=True)
        onnx_path = "models/model.onnx"

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        return onnx_path

    def train_and_optimize(self, n_trials=100):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()

        cv = self.validator.get_stratified_kfold(n_splits=5)

        with mlflow.start_run(run_name="xgboost_optimization"):

            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_0_count", int(np.sum(y_train == 0)))
            mlflow.log_param("class_1_count", int(np.sum(y_train == 1)))
            mlflow.log_param("scale_pos_weight", float(self.scale_pos_weight))

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda t: self.objective(t, X_train, y_train, cv),
                n_trials=n_trials,
                show_progress_bar=True
            )

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_cv_roc_auc", study.best_value)

            best_params = study.best_params.copy()
            best_params.update({
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": self.scale_pos_weight,
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
            })

            self.best_model = XGBClassifier(**best_params)

            self.best_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # ===== ONNX EXPORT =====
            onnx_path = self.save_onnx_model()
            mlflow.log_artifact(onnx_path)

            # Threshold optimization
            self.best_threshold, _ = self.find_optimal_threshold(
                self.best_model, X_val, y_val
            )

            mlflow.log_param("optimal_threshold", float(self.best_threshold))

            y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= self.best_threshold).astype(int)

            mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))
            mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_test_proba))
            mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))

            # Feature importance
            importance_file = "feature_importance.txt"
            with open(importance_file, "w") as f:
                for i, score in enumerate(self.best_model.feature_importances_):
                    f.write(f"f{i}: {score}\n")

            mlflow.log_artifact(importance_file)
            os.remove(importance_file)

            mlflow.xgboost.log_model(self.best_model, "model")

            os.makedirs("models", exist_ok=True)
            with open("models/threshold.pkl", "wb") as f:
                pickle.dump({"threshold": self.best_threshold}, f)
            mlflow.log_artifact("models/threshold.pkl")

            self.plot_roc_curve(y_test, y_test_proba)
            mlflow.log_artifact("plots/roc_curve.png")

            self.plot_confusion_matrix(y_test, y_test_pred)
            mlflow.log_artifact("plots/confusion_matrix.png")

            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print(f"Best CV ROC-AUC: {study.best_value:.4f}")
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
            print(f"Test PR-AUC: {average_precision_score(y_test, y_test_proba):.4f}")
            print(f"Optimal Threshold: {self.best_threshold:.4f}")
            print(f"Test F1 Score: {f1_score(y_test, y_test_pred):.4f}")
            print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
            print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
            print("=" * 60)

        return self.best_model, self.best_threshold


if __name__ == "__main__":
    set_seed()
    classifier = XGBoostBreastCancerClassifier(data_path="data/data.csv")
    model, threshold = classifier.train_and_optimize(n_trials=5)
