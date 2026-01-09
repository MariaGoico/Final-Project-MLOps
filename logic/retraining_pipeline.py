"""
Automated Retraining Pipeline
Triggered when drift is detected
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
import sys

# Import your existing predictor
from logic.breast_cancer_predictor import BreastCancerPredictor

class RetrainingPipeline:
    """
    Handles complete retraining workflow
    """
    
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.backup_dir = Path("artifacts_backup")
        self.data_dir = Path("data")
        
    def run(self, trigger_reason:  str):
        """
        Execute full retraining pipeline
        
        Steps:
        1. Backup current model
        2. Fetch new training data
        3. Train new model
        4. Evaluate new model
        5. Compare with current model
        6. Deploy if better
        
        Returns:
            dict:  Retraining results
        """
        results = {
            'success': False,
            'trigger_reason': trigger_reason,
            'timestamp': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # ===== STEP 1: BACKUP CURRENT MODEL =====
            print("\n" + "="*60)
            print("📦 STEP 1: Backing up current model")
            print("="*60)
            
            self.backup_current_model()
            results['steps']['backup'] = 'success'
            
            # ===== STEP 2: FETCH NEW DATA =====
            print("\n" + "="*60)
            print("📥 STEP 2: Fetching new training data")
            print("="*60)
            
            # TODO: Replace with actual data fetching logic
            # For now, use existing data with augmentation
            train_data = self.fetch_training_data()
            results['steps']['data_fetch'] = {
                'status': 'success',
                'rows': len(train_data)
            }
            
            # ===== STEP 3: TRAIN NEW MODEL =====
            print("\n" + "="*60)
            print("🏋️ STEP 3: Training new model")
            print("="*60)
            
            new_model_metrics = self.train_new_model(train_data)
            results['steps']['training'] = {
                'status':  'success',
                'metrics': new_model_metrics
            }
            
            # ===== STEP 4: EVALUATE NEW MODEL =====
            print("\n" + "="*60)
            print("📊 STEP 4: Evaluating new model")
            print("="*60)
            
            evaluation = self.evaluate_model()
            results['steps']['evaluation'] = evaluation
            
            # ===== STEP 5: COMPARE WITH CURRENT =====
            print("\n" + "="*60)
            print("⚖️ STEP 5: Comparing with current model")
            print("="*60)
            
            should_deploy = self.should_deploy_new_model(evaluation)
            results['should_deploy'] = should_deploy
            
            # ===== STEP 6: DEPLOY IF BETTER =====
            if should_deploy:
                print("\n" + "="*60)
                print("🚀 STEP 6: Deploying new model")
                print("="*60)
                
                self.deploy_new_model()
                results['deployed'] = True
                results['success'] = True
                print("✅ New model deployed successfully")
            else:
                print("\n" + "="*60)
                print("🔄 STEP 6: Reverting to previous model")
                print("="*60)
                
                self.restore_backup()
                results['deployed'] = False
                results['success'] = True
                print("⚠️ New model not better, kept previous model")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Restore backup on failure
            try:
                self.restore_backup()
            except:
                pass
            
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def backup_current_model(self):
        """Backup current model artifacts"""
        if self.artifacts_dir.exists():
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            shutil.copytree(self.artifacts_dir, self.backup_dir)
            print(f"✅ Backed up to {self.backup_dir}")
        else:
            print("⚠️ No existing model to backup")
    
    def fetch_training_data(self):
        """
        Fetch new training data
        
        TODO: Implement actual data fetching logic
        Options:
        - Fetch from database
        - Fetch from API
        - Fetch from production logs
        - Use data versioning tool (DVC)
        """
        # For now, use existing data
        data_path = self.data_dir / "data.csv"
        
        if not data_path.exists():
            # Try alternative path
            data_path = Path("data/data.csv")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"📊 Loaded {len(df)} samples from {data_path}")
        
        # TODO: Add logic to incorporate recent production data
        # For example, fetch predictions from last 30 days with ground truth
        
        return df
    
    def train_new_model(self, data):
        """
        Train new model with updated data
        
        TODO: Call your actual training script
        """
        # For now, simulate training by calling your training script
        # You should adapt this to your actual training code
        
        print("🔄 Running training script...")
        
        # Option 1: Call training script
        import subprocess
        result = subprocess.run(
            [sys.executable, "model.py"],  # Adjust path
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")
        
        print("✅ Training completed")
        
        # Load metrics
        metrics_path = self.artifacts_dir / "validation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        
        return {}
    
    def evaluate_model(self):
        """Evaluate newly trained model"""
        # Load validation metrics
        metrics_path = self.artifacts_dir / "validation_metrics.json"
        
        if not metrics_path.exists():
            raise FileNotFoundError("validation_metrics.json not found")
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print(f"📊 New model metrics:")
        print(f"   F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        return metrics
    
    def should_deploy_new_model(self, new_metrics):
        """
        Compare new model with current model
        
        Returns True if new model is better
        """
        # Load previous metrics
        backup_metrics_path = self.backup_dir / "validation_metrics.json"
        
        if not backup_metrics_path.exists():
            print("⚠️ No previous metrics found, deploying new model")
            return True
        
        with open(backup_metrics_path) as f:
            old_metrics = json.load(f)
        
        # Compare F1 scores
        old_f1 = old_metrics.get('f1_score', 0)
        new_f1 = new_metrics.get('f1_score', 0)
        
        improvement = new_f1 - old_f1
        threshold = 0.01  # Require at least 1% improvement
        
        print(f"\n📊 Model Comparison:")
        print(f"   Previous F1: {old_f1:.4f}")
        print(f"   New F1:      {new_f1:.4f}")
        print(f"   Improvement: {improvement: +.4f}")
        
        if new_f1 > old_f1 + threshold:
            print(f"   ✅ New model is better (+{improvement:.4f})")
            return True
        else: 
            print(f"   ❌ New model not significantly better")
            return False
    
    def deploy_new_model(self):
        """
        Deploy new model by committing artifacts to GitHub and triggering CI/CD
        """
        import subprocess
        import os
        
        print("📊 Preparing deployment...")
        
        # ===== STEP 1: Regenerate baseline (ya lo tienes) =====
        print("📊 Regenerating feature baseline...")
        # Ya se generó en train_new_model()
        
        # ===== STEP 2: Commit artifacts to GitHub =====
        print("📦 Committing artifacts to GitHub...")
        
        try:
            # Configure git
            subprocess.run(['git', 'config', 'user.email', 'retraining-bot@mlops.com'], check=True)
            subprocess.run(['git', 'config', 'user.name', 'Retraining Bot'], check=True)
            
            # Add artifacts
            subprocess.run(['git', 'add', 'artifacts/'], check=True)
            
            # Commit
            commit_message = f"🤖 Auto-retrain:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Push (requires GITHUB_TOKEN)
            github_token = os.environ.get('GITHUB_TOKEN')
            if not github_token:
                raise Exception("GITHUB_TOKEN not found in environment variables")
            
            # Get repo info from environment or config
            repo_owner = os.environ.get('GITHUB_REPO_OWNER', 'MariaGoico')
            repo_name = os.environ.get('GITHUB_REPO_NAME', 'Final-Project-MLOps')
            
            remote_url = f"https://{github_token}@github.com/{repo_owner}/{repo_name}.git"
            
            subprocess.run(['git', 'remote', 'set-url', 'origin', remote_url], check=True)
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            
            print("✅ Artifacts committed and pushed to GitHub")
            print(f"   This will trigger CI/CD pipeline automatically")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return False
        
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
        
    def restore_backup(self):
        """Restore previous model from backup"""
        if self.backup_dir.exists():
            if self.artifacts_dir.exists():
                shutil.rmtree(self.artifacts_dir)
            
            shutil.copytree(self.backup_dir, self.artifacts_dir)
            print(f"✅ Restored from {self.backup_dir}")
        else:
            print("⚠️ No backup found to restore")