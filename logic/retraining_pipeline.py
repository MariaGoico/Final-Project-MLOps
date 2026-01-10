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
        Deploy new model via GitHub API in a SINGLE commit
        Uses Tree API + Commit API for atomic updates
        """
        print("📦 Deploying artifacts to GitHub via API...")
        
        try:
            import requests
            import base64
            from datetime import datetime, timezone
            
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
                'Accept':  'application/vnd.github.v3+json'
            }
            
            base_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}'
            
            print(f"   🔗 Target: {repo_owner}/{repo_name} (branch: {branch})")
            
            # ========================================
            # COLLECT ARTIFACT FILES
            # ========================================
            if not self.artifacts_dir.exists():
                print("   ❌ Artifacts directory not found")
                return False
            
            artifact_files = [f for f in self.artifacts_dir.glob('*') if f.is_file()]
            if not artifact_files:
                print("   ⚠️ No artifact files to upload")
                return False
            
            print(f"   📦 Found {len(artifact_files)} files to upload")
            
            # ========================================
            # GET CURRENT COMMIT SHA
            # ========================================
            print("   🔍 Getting current commit SHA...")
            ref_url = f'{base_url}/git/refs/heads/{branch}'
            ref_response = requests.get(ref_url, headers=headers)
            
            if ref_response.status_code != 200:
                print(f"   ❌ Failed to get branch ref: {ref_response.json()}")
                return False
            
            current_commit_sha = ref_response.json()['object']['sha']
            print(f"   ✅ Current commit:  {current_commit_sha[: 7]}")
            
            # ========================================
            # GET BASE TREE SHA
            # ========================================
            commit_url = f'{base_url}/git/commits/{current_commit_sha}'
            commit_response = requests.get(commit_url, headers=headers)
            
            if commit_response.status_code != 200:
                print(f"   ❌ Failed to get commit: {commit_response.json()}")
                return False
            
            base_tree_sha = commit_response.json()['tree']['sha']
            
            # ========================================
            # CREATE BLOBS FOR EACH FILE
            # ========================================
            print("   📤 Creating blobs for artifacts...")
            tree_items = []
            
            for artifact_file in artifact_files: 
                print(f"      - {artifact_file.name}...", end=" ")
                
                try:
                    # Read and encode file
                    with open(artifact_file, 'rb') as f:
                        content = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Create blob
                    blob_url = f'{base_url}/git/blobs'
                    blob_payload = {
                        'content':  content,
                        'encoding':  'base64'
                    }
                    
                    blob_response = requests.post(blob_url, headers=headers, json=blob_payload)
                    
                    if blob_response.status_code != 201:
                        print(f"❌ ({blob_response.status_code})")
                        continue
                    
                    blob_sha = blob_response.json()['sha']
                    
                    # Add to tree
                    tree_items.append({
                        'path': f'artifacts/{artifact_file.name}',
                        'mode': '100644',
                        'type': 'blob',
                        'sha': blob_sha
                    })
                    
                    print("✅")
                    
                except Exception as e: 
                    print(f"❌ ({str(e)})")
            
            if not tree_items: 
                print("   ❌ No blobs created successfully")
                return False
            
            # ========================================
            # CREATE NEW TREE
            # ========================================
            print(f"   🌳 Creating tree with {len(tree_items)} files...")
            tree_url = f'{base_url}/git/trees'
            tree_payload = {
                'base_tree': base_tree_sha,
                'tree':  tree_items
            }
            
            tree_response = requests.post(tree_url, headers=headers, json=tree_payload)
            
            if tree_response.status_code != 201:
                print(f"   ❌ Failed to create tree:  {tree_response.json()}")
                return False
            
            new_tree_sha = tree_response.json()['sha']
            print(f"   ✅ Tree created: {new_tree_sha[:7]}")
            
            # ========================================
            # CREATE COMMIT
            # ========================================
            commit_message = f"Auto-retrain:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            print(f"   💬 Creating commit: {commit_message}")
            
            # ⭐ FORMAT DATE FOR GITHUB API (ISO 8601 with UTC timezone)
            commit_date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            commit_url = f'{base_url}/git/commits'
            commit_payload = {
                'message':  commit_message,
                'tree': new_tree_sha,
                'parents': [current_commit_sha],
                'author':  {
                    'name': 'Retraining Bot',
                    'email': 'retraining-bot@render.com',
                    'date': commit_date
                },
                'committer':  {
                    'name': 'Retraining Bot',
                    'email': 'retraining-bot@render.com',
                    'date': commit_date
                }
            }
            
            commit_response = requests.post(commit_url, headers=headers, json=commit_payload)
            
            if commit_response.status_code != 201:
                print(f"   ❌ Failed to create commit: {commit_response.json()}")
                return False
            
            new_commit_sha = commit_response.json()['sha']
            print(f"   ✅ Commit created: {new_commit_sha[:7]}")
            
            # ========================================
            # UPDATE BRANCH REFERENCE
            # ========================================
            print(f"   🔄 Updating {branch} branch...")
            update_ref_url = f'{base_url}/git/refs/heads/{branch}'
            update_payload = {
                'sha': new_commit_sha,
                'force': False
            }
            
            update_response = requests.patch(update_ref_url, headers=headers, json=update_payload)
            
            if update_response.status_code != 200:
                print(f"   ❌ Failed to update branch: {update_response.json()}")
                return False
            
            print(f"   ✅ Branch updated successfully")
            
            # ========================================
            # SUMMARY
            # ========================================
            print()
            print(f"   📊 Deployment Summary:")
            print(f"      ✅ Files deployed: {len(tree_items)}")
            for item in tree_items:
                print(f"         - {item['path']}")
            print(f"      📝 Commit:  {commit_message}")
            print(f"      🔗 SHA: {new_commit_sha[:7]}")
            print()
            print("   ✅ All artifacts pushed to GitHub in a SINGLE commit")
            print("   🔄 CI/CD pipeline will trigger automatically")
            
            return True
            
        except ImportError:
            print("   ❌ 'requests' library not available")
            return False
        
        except Exception as e:
            print(f"   ❌ GitHub API error:  {e}")
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