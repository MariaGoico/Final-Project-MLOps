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
        Deploy new model via GitHub API (without requiring . git directory)
        """
        print("📦 Deploying artifacts to GitHub via API...")
        
        try:
            import requests
            import base64
            
            # ========================================
            # GET GITHUB CREDENTIALS
            # ========================================
            github_token = os.environ.get('GITHUB_TOKEN')
            if not github_token:
                print("   ❌ GITHUB_TOKEN not found in environment")
                return False
            
            repo_owner = os.environ.get('GITHUB_REPO_OWNER', 'MariaGoico')
            repo_name = os.environ.get('GITHUB_REPO_NAME', 'Final-Project-MLOps')
            branch = 'main'
            
            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            print(f"   🔗 Target:  {repo_owner}/{repo_name} (branch: {branch})")
            
            # ========================================
            # UPLOAD EACH ARTIFACT FILE
            # ========================================
            if not self.artifacts_dir.exists():
                print("   ❌ Artifacts directory not found")
                return False
            
            artifact_files = list(self.artifacts_dir.glob('*'))
            if not artifact_files:
                print("   ⚠️ No artifact files to upload")
                return False
            
            print(f"   📦 Found {len(artifact_files)} files to upload")
            
            uploaded = []
            failed = []
            
            for artifact_file in artifact_files: 
                if not artifact_file.is_file():
                    continue
                    
                print(f"   📤 Uploading {artifact_file.name}.. .", end=" ")
                
                try:
                    # Read file content and encode to base64
                    with open(artifact_file, 'rb') as f:
                        content = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Get current file SHA (if file exists)
                    file_path = f'artifacts/{artifact_file.name}'
                    get_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}'
                    
                    get_response = requests.get(
                        get_url,
                        headers=headers,
                        params={'ref': branch}
                    )
                    
                    file_sha = None
                    if get_response.status_code == 200:
                        file_sha = get_response.json().get('sha')
                    
                    # Prepare commit message
                    commit_message = f"Auto-retrain: Update {artifact_file.name}"
                    
                    # Create/update file
                    payload = {
                        'message':  commit_message,
                        'content': content,
                        'branch': branch
                    }
                    
                    if file_sha:
                        payload['sha'] = file_sha  # Required for updates
                    
                    put_response = requests.put(
                        get_url,
                        headers=headers,
                        json=payload
                    )
                    
                    if put_response.status_code in [200, 201]:
                        print("✅")
                        uploaded.append(artifact_file.name)
                    else:
                        print(f"❌ ({put_response.status_code})")
                        error_msg = put_response.json().get('message', 'Unknown error')
                        print(f"      Error: {error_msg}")
                        failed.append(artifact_file.name)
                    
                except Exception as e:
                    print(f"❌ ({str(e)})")
                    failed.append(artifact_file.name)
            
            # ========================================
            # SUMMARY
            # ========================================
            print()
            print(f"   📊 Upload Summary:")
            print(f"      ✅ Uploaded: {len(uploaded)} files")
            if uploaded:
                for name in uploaded:
                    print(f"         - {name}")
            
            if failed:
                print(f"      ❌ Failed: {len(failed)} files")
                for name in failed:
                    print(f"         - {name}")
            
            if uploaded and not failed:
                print()
                print("   ✅ All artifacts pushed to GitHub successfully")
                print("   🔄 CI/CD pipeline will trigger automatically")
                return True
            elif uploaded and failed:
                print()
                print("   ⚠️ Some artifacts uploaded, but some failed")
                return False
            else:
                print()
                print("   ❌ No artifacts uploaded successfully")
                return False
                
        except ImportError:
            print("   ❌ 'requests' library not available")
            print("      Add 'requests' to pyproject.toml dependencies")
            return False
        
        except Exception as e:
            print(f"   ❌ GitHub API error: {e}")
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