import os
import requests
from datetime import datetime

class RetrainingPipeline:
    
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.repo_owner = os.environ.get('GITHUB_REPO_OWNER')
        self.repo_name = os.environ.get('GITHUB_REPO_NAME')
        self.workflow_id = 'cicd.yml'
        self.ref = 'main'
        
    def run(self, trigger_reason: str):
        """
        Triggers the CI/CD pipeline on GitHub Actions via API.
        Instead of training locally (which requires heavy libs), we offload
        the heavy lifting to GitHub Runners.
        """
        print(f"\n🚀 Triggering Remote Retraining on GitHub Actions...")
        print(f"   Reason: {trigger_reason}")
        
        if not self.github_token:
            print("❌ Error: GITHUB_TOKEN not found in environment variables.")
            return {'success': False, 'error': 'Missing GITHUB_TOKEN'}

        # Endpoint de GitHub para disparar workflows manualmente
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{self.workflow_id}/dispatches"
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "ref": self.ref,
            "inputs": {
                "reason": trigger_reason  # Podrías usar esto en el yaml si quisieras
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 204:
                print("✅ Successfully triggered GitHub Action!")
                print("   The new model will be trained and deployed automatically.")
                return {
                    'success': True, 
                    'message': 'GitHub Action triggered successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"❌ Failed to trigger GitHub Action. Status: {response.status_code}")
                print(f"   Response: {response.text}")
                return {
                    'success': False, 
                    'error': f"GitHub API Error: {response.text}"
                }
                
        except Exception as e:
            print(f"❌ Exception calling GitHub API: {e}")
            return {'success': False, 'error': str(e)}

# Para pruebas locales
if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    pipeline.run("Manual Test")