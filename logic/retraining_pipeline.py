"""
Simplified Automated Retraining Pipeline
Always trains and deploys without comparison
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
import sys
import subprocess
import os

class RetrainingPipeline:
    """
    Simplified retraining workflow - always deploys
    """
    
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.backup_dir = Path("artifacts_backup")
        self.data_dir = Path("data")
        
    def run(self, trigger_reason:  str):
        """
        Simplified retraining pipeline: 
        1. Backup current model
        2. Train with original data
        3. ALWAYS deploy (no comparison)
        
        Returns: 
            dict:  Retraining results
        """
        results = {
            'success': False,
            'trigger_reason': trigger_reason,
            'timestamp': datetime.now().isoformat(),
            'steps':  {}
        }
        
        try:
            # ===== STEP 1: BACKUP =====
            print("\n" + "="*60)
            print("📦 STEP 1: Backing up current model")
            print("="*60)
            
            self.backup_current_model()
            results['steps']['backup'] = 'success'
            
            # ===== STEP 2: TRAIN =====
            print("\n" + "="*60)
            print("🏋️ STEP 2: Training new model with original data")
            print("="*60)
            
            new_model_metrics = self.train_new_model()
            results['steps']['training'] = {
                'status': 'success',
                'metrics': new_model_metrics
            }
            
            # ===== STEP 3: DEPLOY (ALWAYS) =====
            print("\n" + "="*60)
            print("🚀 STEP 3: Deploying new model (no comparison)")
            print("="*60)
            
            deploy_success = self.deploy_new_model()
            
            if deploy_success:
                results['deployed'] = True
                results['success'] = True
                print("✅ New model deployed successfully via CI/CD")
            else:
                results['deployed'] = False
                results['success'] = False
                results['error'] = "Deployment failed"
            
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
    
    def train_new_model(self):
        """
        Train new model with ORIGINAL data (data/data.csv)
        """
        print("🔄 Running training script with original data...")
        print(f"   Using: data/data.csv")
        
        # Call training script
        result = subprocess.run(
            [sys.executable, "-m", "logic.model"],
            capture_output=True,
            text=True,
            cwd=str(Path.cwd())
        )
        
        if result.returncode != 0:
            print(f"❌ Training output: {result.stdout}")
            print(f"❌ Training error: {result.stderr}")
            raise Exception(f"Training failed: {result.stderr}")
        
        print("✅ Training completed successfully")
        
        # Show last lines of output
        if result.stdout:
            output_lines = result.stdout.split('\n')
            print("\n📤 Training output (last 10 lines):")
            for line in output_lines[-10:]: 
                if line.strip():
                    print(f"   {line}")
        
        # Load metrics
        metrics_path = self.artifacts_dir / "validation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                print(f"\n📊 New model metrics:")
                print(f"   F1: {metrics.get('f1_score', 0):.4f}")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                return metrics
        
        return {}
    
    def deploy_new_model(self):
        """
        Deploy new model via CI/CD (commit + push to trigger GitHub Actions)
        Uses environment variables instead of git config to avoid permission issues
        """
        print("📦 Deploying artifacts to GitHub...")
        
        try:
            # ========================================
            # CONFIGURE GIT USING ENVIRONMENT VARIABLES
            # ========================================
            env = os.environ.copy()
            env['GIT_AUTHOR_NAME'] = 'Retraining Bot'
            env['GIT_AUTHOR_EMAIL'] = 'retraining-bot@render.com'
            env['GIT_COMMITTER_NAME'] = 'Retraining Bot'
            env['GIT_COMMITTER_EMAIL'] = 'retraining-bot@render.com'
            
            print("   ✅ Git identity configured via environment variables")
            
            # ========================================
            # ADD ARTIFACTS
            # ========================================
            print("   📦 Staging artifacts...")
            result = subprocess.run(
                ['git', 'add', 'artifacts/'], 
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            print("   ✅ Artifacts staged")
            
            # ========================================
            # COMMIT
            # ========================================
            commit_message = f"🤖 Auto-retrain:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            print(f"   💬 Committing: {commit_message}")
            
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message], 
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                if "nothing to commit" in result.stdout.lower():
                    print("   ⚠️ No changes to commit (artifacts unchanged)")
                    return True  # Not an error
                else:
                    print(f"   ❌ Commit failed: {result.stderr}")
                    if result.stdout:
                        print(f"   stdout: {result.stdout}")
                    return False
            
            print(f"   ✅ Committed successfully")
            
            # ========================================
            # CONFIGURE REMOTE
            # ========================================
            github_token = os.environ.get('GITHUB_TOKEN')
            if not github_token:
                print("   ❌ GITHUB_TOKEN not found in environment")
                print("      Deployment via Git push not available")
                print("      Model trained but not pushed to GitHub")
                return False
            
            repo_owner = os.environ.get('GITHUB_REPO_OWNER', 'MariaGoico')
            repo_name = os.environ.get('GITHUB_REPO_NAME', 'Final-Project-MLOps')
            
            remote_url = f"https://{github_token}@github.com/{repo_owner}/{repo_name}.git"
            
            print(f"   🔗 Configuring remote: {repo_owner}/{repo_name}")
            
            # Try to set remote URL (may fail if remote doesn't exist)
            subprocess.run(
                ['git', 'remote', 'set-url', 'origin', remote_url], 
                capture_output=True,
                env=env
            )
            
            # If set-url failed, try to add remote
            subprocess.run(
                ['git', 'remote', 'add', 'origin', remote_url],
                capture_output=True,
                env=env
            )
            
            # ========================================
            # PUSH
            # ========================================
            print("   🚀 Pushing to GitHub...")
            
            push_result = subprocess.run(
                ['git', 'push', 'origin', 'main'], 
                capture_output=True,
                text=True,
                env=env,
                timeout=60  # 60 second timeout
            )
            
            if push_result.returncode != 0:
                print(f"   ❌ Push failed: {push_result.stderr}")
                if push_result.stdout:
                    print(f"   stdout:  {push_result.stdout}")
                return False
            
            print("   ✅ Artifacts pushed to GitHub successfully")
            print(f"   📝 Commit: {commit_message}")
            print(f"   🔄 CI/CD pipeline will trigger automatically")
            
            return True
            
        except subprocess.CalledProcessError as e: 
            print(f"   ❌ Git operation failed: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"      stderr: {e.stderr}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"      stdout: {e.stdout}")
            return False
        
        except subprocess.TimeoutExpired:
            print(f"   ❌ Git push timed out (>60s)")
            return False
        
        except Exception as e: 
            print(f"   ❌ Deployment error: {e}")
            import traceback
            traceback.print_exc()
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