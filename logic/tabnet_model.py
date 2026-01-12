import os
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
    precision_recall_curve, auc as sklearn_auc
)
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import json
from datetime import datetime
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from logic.data_module import CancerDataProcessorTabNet
from logic.utilities import DataValidator
from logic.utilities import set_seed


class TabNetBreastCancerClassifier:
    """
    TabNet classifier for breast cancer prediction with Optuna optimization and MLflow tracking.
    Serialization with ONNX for production deployment.
    """
    
    def __init__(self, data_path, tracking_uri=None):
        self.data_path = data_path
        self.processor = CancerDataProcessorTabNet()
        self.validator = DataValidator()
        self.best_model = None
        self.best_threshold = 0.5
        self.n_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set MLflow tracking URI
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("breast_cancer_prediction_tabnet")
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        X, y = self.processor.load_data(self.data_path)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.validator.split_train_val_test(X, y)
        
        # Preprocess
        X_train = self.processor.fit_transform(X_train)
        X_val = self.processor.transform(X_val)
        X_test = self.processor.transform(X_test)
        
        # Calculate scale_pos_weight for imbalanced data
        self.scale_pos_weight = self.validator.get_scale_pos_weight(y_train)
        self.n_features = X_train.shape[1]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def plot_confusion_matrix(self, y_true, y_pred, save_path="plots/tabnet/confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, save_path="plots/tabnet/roc_curve.png"):
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
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna objective function for TabNet hyperparameter optimization.
        Uses threshold-independent metrics (ROC-AUC) for optimization.
        """
        # Suggest hyperparameters
        params = {
            'n_d': trial.suggest_int('n_d', 8, 64),
            'n_a': trial.suggest_int('n_a', 8, 64),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        }
        
        # Create TabNet model
        model = TabNetClassifier(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            lambda_sparse=params['lambda_sparse'],
            optimizer_params=params['optimizer_params'],
            mask_type=params['mask_type'],
            device_name=str(self.device),
            seed=42,
            verbose=0
        )
        
        # Train model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=30,
            patience=1,
            batch_size=32,
            virtual_batch_size=16,
        )
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate threshold-independent metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        # Log trial to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_roc_auc", roc_auc)
            mlflow.log_metric("val_pr_auc", pr_auc)
        
        return roc_auc
    
    def find_optimal_threshold(self, model, X_val, y_val, plot=True, save_path="plots/tabnet/threshold_f1.png"):
        """
        Find optimal threshold using threshold-dependent metrics.
        Optimizes F1 score for production use.
        """
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
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

    def log_feature_importance(self, model, feature_names):
        """
        Log TabNet feature importance to MLflow.
        """
        feature_importance = model.feature_importances_
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, feature_importance))
        
        # Save to file
        os.makedirs("artifacts/tabnet", exist_ok=True)
        with open("artifacts/tabnet/feature_importance.json", "w") as f:
            json.dump(importance_dict, f, indent=2)
        
        mlflow.log_artifact("artifacts/tabnet/feature_importance.json")
        os.makedirs("plots/tabnet", exist_ok=True)
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)[::-1][:20]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig("plots/tabnet/feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("plots/tabnet/feature_importance.png")

    def save_validation_metrics(self, y_test, y_test_pred, y_test_proba, X_train):
        """
        Save validation metrics for production monitoring.
        """
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        # Calculate PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
        pr_auc = sklearn_auc(recall_curve, precision_curve)
        
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        validation_metrics = {
            'f1_score': float(f1_score(y_test, y_test_pred)),
            'accuracy': float(accuracy_score(y_test, y_test_pred)),
            'precision': float(precision_score(y_test, y_test_pred)),
            'recall': float(recall_score(y_test, y_test_pred)),
            'specificity': specificity,
            'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
            'pr_auc': float(pr_auc),
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            },
            'model_info': {
                'algorithm': 'TabNet',
                'version': '1.0.0',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': int(self.n_features),
                'train_samples': int(X_train.shape[0]),
                'test_samples': int(len(y_test)),
                'threshold': float(self.best_threshold)
            }
        }
        
        os.makedirs("artifacts/tabnet", exist_ok=True)
        with open('artifacts/tabnet/validation_metrics.json', 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ Validation metrics saved to artifacts/tabnet/validation_metrics.json")
        print("="*60)
        print(json.dumps(validation_metrics, indent=2))
        print("="*60 + "\n")
        
        mlflow.log_artifact('artifacts/tabnet/validation_metrics.json')
        
        return validation_metrics

    def save_feature_baseline(self, X_train):
        """
        Save feature statistics for drift detection in production.
        """
        feature_means = X_train.mean(axis=0)
        feature_stds = X_train.std(axis=0)
        
        os.makedirs("artifacts/tabnet", exist_ok=True)
        np.savez(
            'artifacts/tabnet/feature_baseline.npz',
            means=feature_means,
            stds=feature_stds
        )
        
        print("="*60)
        print(f"✅ Feature baseline saved: {len(feature_means)} features")
        print(f"   Mean range: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
        print(f"   Std range: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")
        print("="*60 + "\n")
        
        mlflow.log_artifact('artifacts/tabnet/feature_baseline.npz')

    def export_to_onnx(self, model, X_sample, save_path="artifacts/tabnet/tabnet.onnx"):
        print("\n" + "=" * 60)
        print("📦 Exporting TabNet model to ONNX format...")
        print("=" * 60)

        model.network.eval()

        X_tensor = torch.tensor(
            X_sample[:1],
            dtype=torch.float32,
            device=self.device
        )

        torch.onnx.export(
            model.network,
            X_tensor,
            save_path,
            export_params=True,
            opset_version=11,            # 🔒 safest for TabNet
            do_constant_folding=False,   # still required
            dynamo=False,                # 🔥 THIS is the missing piece
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        print("✅ Raw ONNX export completed")

        # Optional but recommended validation
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(save_path)
        ort_inputs = {
            ort_session.get_inputs()[0].name: X_sample[:1].astype("float32")
        }
        ort_outs = ort_session.run(None, ort_inputs)

        print("✅ ONNX Runtime inference successful")
        print(f"   Output shape: {ort_outs[0].shape}")
        print("=" * 60 + "\n")

        return save_path


    def train_and_optimize(self, n_trials=50):
        """
        Main training loop with Optuna optimization and MLflow tracking.
        Exports model to ONNX format.
        """
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
        
        # Feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Start MLflow parent run
        if mlflow.active_run() is not None:
            mlflow.end_run()
            
        with mlflow.start_run(run_name="tabnet_optimization"):
            
            # Log dataset information
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("class_0_count", int(np.sum(y_train == 0)))
            mlflow.log_param("class_1_count", int(np.sum(y_train == 1)))
            mlflow.log_param("scale_pos_weight", float(self.scale_pos_weight))
            mlflow.log_param("device", str(self.device))
            
            # Optuna optimization
            study = optuna.create_study(
                direction='maximize',
                study_name='tabnet_breast_cancer_optimization'
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            # Log best hyperparameters
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_val_roc_auc", study.best_value)
            
            # Train final model with best parameters
            best_params = study.best_params.copy()
            
            self.best_model = TabNetClassifier(
                n_d=best_params['n_d'],
                n_a=best_params['n_a'],
                n_steps=best_params['n_steps'],
                gamma=best_params['gamma'],
                lambda_sparse=best_params['lambda_sparse'],
                optimizer_params=dict(lr=best_params['lr']),
                mask_type=best_params['mask_type'],
                device_name=str(self.device),
                seed=42,
                verbose=1
            )
            
            # Train final model
            self.best_model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                max_epochs=30,
                patience=1,
                batch_size=32,
                virtual_batch_size=16,
            )
            
            # Log feature importance
            print("Logging feature importance...")
            self.log_feature_importance(self.best_model, feature_names)
            
            # Find optimal threshold
            self.best_threshold, _ = self.find_optimal_threshold(
                self.best_model, X_val, y_val
            )
            mlflow.log_param("optimal_threshold", float(self.best_threshold))
            mlflow.log_artifact("plots/tabnet/threshold_f1.png")
            
            # Final evaluation on TEST
            y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= self.best_threshold).astype(int)

            # Log test metrics
            mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))
            mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_test_proba))
            mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))

            # Save validation metrics
            print("\n" + "="*60)
            print("📊 Saving validation metrics for production monitoring...")
            print("="*60)
            
            self.save_validation_metrics(y_test, y_test_pred, y_test_proba, X_train)
            self.save_feature_baseline(X_train)

            # ===== SERIALIZE WITH ONNX =====
            os.makedirs("artifacts/tabnet", exist_ok=True)
            
            # Export to ONNX
            self.export_to_onnx(self.best_model, X_test)
            
            # Save TabNet model (PyTorch checkpoint as backup)
            self.best_model.save_model("artifacts/tabnet/tabnet_model")
            
            # Save threshold
            with open("artifacts/tabnet/threshold.json", "w") as f:
                json.dump({"threshold": float(self.best_threshold)}, f)
            mlflow.log_artifact("artifacts/tabnet/threshold.json")
            
            # Save preprocessing pipeline
            with open("artifacts/tabnet/preprocessor.pkl", "wb") as f:
                pickle.dump(self.processor, f)
            mlflow.log_artifact("artifacts/tabnet/preprocessor.pkl")
            
            # Save metadata
            with open("artifacts/tabnet/metadata.json", "w") as f:
                json.dump({
                    "n_features": self.n_features,
                    "model_type": "tabnet",
                    "framework": "pytorch",
                    "onnx_version": onnx.__version__,
                    "device": str(self.device)
                }, f)
            mlflow.log_artifact("artifacts/tabnet/metadata.json")

            # Plot visualizations
            self.plot_roc_curve(y_test, y_test_proba)
            mlflow.log_artifact("plots/tabnet/roc_curve.png")

            self.plot_confusion_matrix(y_test, y_test_pred)
            mlflow.log_artifact("plots/tabnet/confusion_matrix.png")
            
            # Log PyTorch model to MLflow
            mlflow.pytorch.log_model(self.best_model.network, "pytorch_model")
            
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Best Val ROC-AUC: {study.best_value:.4f}")
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
            print(f"Test PR-AUC: {average_precision_score(y_test, y_test_proba):.4f}")
            print(f"Optimal Threshold: {self.best_threshold:.4f}")
            print(f"Test F1 Score: {f1_score(y_test, y_test_pred):.4f}")
            print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
            print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
            print(f"{'='*60}\n")
            
        return self.best_model, self.best_threshold


if __name__ == "__main__":
    set_seed()  # Set a fixed seed
    classifier = TabNetBreastCancerClassifier(
        data_path='data/data.csv'
    )
    model, threshold = classifier.train_and_optimize(n_trials=30)