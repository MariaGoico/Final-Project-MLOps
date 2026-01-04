import gradio as gr
import requests
import pandas as pd
from io import StringIO
import os

# API URL on Render
API_URL = "https://mlops-finalproject-latest.onrender.com"

def get_api_info():
    """Gets model information from the API"""
    try:
        response = requests.get(f"{API_URL}/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except: 
        return None

def check_api_health():
    """Checks the API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = "🟢 Online" if data.get("model_loaded") else "🟡 Online (model not loaded)"
            return status
        return "🔴 Offline"
    except:
        return "🔴 Cannot connect"

def predict_csv(file_path):
    """Performs prediction from a CSV file"""
    if not file_path: 
        return "❌ Please upload a CSV file", None
    
    try:
        # Read CSV for preview
        df_preview = pd.read_csv(file_path)
        
        # Send to API
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            files = {"file": (filename, f, "text/csv")}
            
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
            if response.status_code != 200:
                try:
                    detail = response.json().get('detail', 'Unknown error')
                except:
                    detail = response.text
                return f"❌ API Error ({response.status_code}): {detail}", None
            
            result = response.json()
            
            # Format results
            predictions = result.get('predictions', [])
            summary = result.get('summary', {})
            
            # Create results table
            results_data = []
            for pred in predictions:
                if "error" not in pred:
                    results_data.append({
                        "Row": pred['row'],
                        "Diagnosis": pred['diagnosis'],
                        "Probability": pred['probability'],
                        "Confidence": pred['confidence']
                    })
                else:
                    results_data.append({
                        "Row": pred['row'],
                        "Diagnosis": "ERROR",
                        "Probability":  "-",
                        "Confidence":  pred['error']
                    })
            
            results_df = pd.DataFrame(results_data)
            
            # Create comprehensive summary with stats
            malignant_count = sum(1 for p in predictions if p.get('prediction') == 1)
            benign_count = sum(1 for p in predictions if p.get('prediction') == 0)
            
            summary_text = f"""
## 📊 Prediction Summary

**File Information:**
- File: `{result.get('file')}`
- Total rows: **{summary.get('total_rows', 0)}**
- Successful predictions: **{summary.get('successful', 0)}**
- Errors: **{summary.get('errors', 0)}**
- Features used: **{summary.get('features_used', 0)}**

**Diagnosis Distribution:**
- 🔴 Malignant (M): **{malignant_count}** ({malignant_count/len(predictions)*100:.1f}%)
- 🟢 Benign (B): **{benign_count}** ({benign_count/len(predictions)*100:.1f}%)

---

### 📋 Detailed Results

{results_df.to_markdown(index=False)}

---

**CSV Preview (first 5 rows):**

{df_preview.head(5).to_markdown(index=False)}
"""
            
            return summary_text, results_df
            
    except requests.exceptions.Timeout:
        return "❌ Timeout:  The API took too long to respond", None
    except requests.exceptions.ConnectionError:
        return "❌ Connection error: Could not connect to the API", None
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", None

def predict_manual(features_text):
    """Performs prediction from manual input (30 features separated by commas)"""
    if not features_text:
        return "❌ Please enter 30 values separated by commas"
    
    try:
        # Parse values
        values = [float(x.strip()) for x in features_text.split(',')]
        
        if len(values) != 30:
            return f"❌ Expected 30 values, but received {len(values)}"
        
        # Create temporary CSV
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            # Header
            header = [
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
                "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
                "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
            ]
            f.write(','.join(header) + '\n')
            f.write(','.join(map(str, values)) + '\n')
            temp_path = f.name
        
        # Send to API
        with open(temp_path, "rb") as f:
            files = {"file": ("manual_input.csv", f, "text/csv")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if response.status_code != 200:
            try: 
                detail = response.json().get('detail', 'Unknown error')
            except:
                detail = response.text
            return f"❌ API Error ({response.status_code}): {detail}"
        
        result = response.json()
        pred = result['predictions'][0]
        
        if "error" in pred:
            return f"❌ Prediction error: {pred['error']}"
        
        # Format result
        diagnosis = pred['diagnosis']
        probability = pred['probability']
        confidence = pred['confidence']
        
        emoji = "🔴" if pred['prediction'] == 1 else "🟢"
        
        return f"""
## {emoji} Prediction Result

### Diagnosis: **{diagnosis}**

- **Probability**: `{probability}`
- **Confidence**: `{confidence}`

---

{"### ⚠️ Warning\nPotentially **malignant** case detected.  Medical consultation is recommended." if pred['prediction'] == 1 else "### ✅ Good News\nThe case appears to be **benign** (non-cancerous)."}

**Note:** This prediction is for informational purposes only and should not replace professional medical diagnosis. 
"""
        
    except ValueError:
        return "❌ Error: Make sure to enter only numbers separated by commas"
    except Exception as e: 
        return f"❌ Unexpected error: {str(e)}"

# Custom CSS
custom_css = """
.output-markdown {
    font-size: 14px;
}
.output-markdown table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 13px;
}
.output-markdown th {
    background-color: #4CAF50;
    color:  white;
    padding: 10px;
    text-align:  left;
}
.output-markdown td {
    padding: 8px;
    border-bottom:  1px solid #ddd;
}
.output-markdown tr:hover {
    background-color: #f5f5f5;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Cancer Prediction") as demo:
    gr.Markdown("""
    # 🏥 Breast Cancer Prediction API
    
    **Machine Learning-powered breast cancer diagnosis prediction** using XGBoost trained on the Wisconsin Breast Cancer dataset.
    """)
    
    # API status bar
    with gr.Row():
        with gr.Column(scale=4):
            api_status = gr.Textbox(
                label="API Status", 
                value=check_api_health(), 
                interactive=False,
                container=True
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            refresh_btn.click(check_api_health, outputs=api_status)
    
    # Tabs
    with gr.Tabs():
        # Tab 1: CSV Prediction
        with gr.Tab("📁 CSV Prediction"):
            gr.Markdown("""
            ### How to use:
            1. Upload a CSV file containing **30 features** from the Wisconsin Breast Cancer dataset
            2. The file may or may not include column headers
            3. Each row represents one patient case to be analyzed
            
            **Supported format:** `.csv` files with 30 numeric columns
            """)
            
            csv_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            
            predict_csv_btn = gr.Button(
                "🔍 Analyze CSV", 
                variant="primary", 
                size="lg"
            )
            
            with gr.Accordion("📊 Results", open=True):
                summary_output = gr.Markdown(label="Prediction Summary")
            
            with gr.Accordion("📈 Results Table (Interactive)", open=False):
                results_df_output = gr.Dataframe(
                    label="Detailed Results",
                    interactive=False,
                    wrap=True
                )
            
            predict_csv_btn.click(
                predict_csv,
                inputs=csv_input,
                outputs=[summary_output, results_df_output]
            )
        
        # Tab 2: Manual Prediction
        with gr.Tab("✍️ Manual Input"):
            gr.Markdown("""
            ### How to use: 
            Enter **30 feature values** separated by commas. 
            
            **Features (in order):**
            
            **Mean values (1-10):**  
            radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension
            
            **Standard Error (11-20):**  
            radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se
            
            **Worst values (21-30):**  
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
            """)
            
            features_input = gr.Textbox(
                label="Feature Values (30 comma-separated numbers)",
                placeholder="17.99,10.38,122.8,1001,0.1184,... ",
                lines=4
            )
            
            predict_manual_btn = gr.Button(
                "🔍 Predict", 
                variant="primary", 
                size="lg"
            )
            
            manual_output = gr.Markdown(label="Prediction Result")
            
            predict_manual_btn.click(
                predict_manual,
                inputs=features_input,
                outputs=manual_output
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189"],
                    ["20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902"]
                ],
                inputs=features_input,
                label="📝 Example Cases (click to use)"
            )
        
        # Tab 3: Model Information
        with gr.Tab("ℹ️ About"):
            gr.Markdown("### Model Information")
            
            info_btn = gr.Button("📊 Fetch Model Details", size="lg")
            info_output = gr.JSON(label="Model Configuration")
            
            def get_info_formatted():
                info = get_api_info()
                if info:
                    return info
                return {"error": "Could not retrieve model information"}
            
            info_btn.click(get_info_formatted, outputs=info_output)
            
            gr.Markdown("""
            ---
            
            ### 📚 Wisconsin Breast Cancer Dataset
            
            This system uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, containing features computed 
            from digitized images of fine needle aspirate (FNA) of breast masses. 
            
            **Dataset Composition:**
            - 10 real-valued features computed for each cell nucleus
            - 3 statistical measures for each feature:  mean, standard error, and "worst" (largest value)
            - **Total: 30 features**
            
            **Target Classes:**
            - **M (Malignant)**: Cancerous tumor
            - **B (Benign)**: Non-cancerous tumor
            
            ### 🔬 Technology Stack
            
            - **Algorithm**: XGBoost Classifier
            - **Optimization**: Optuna (automated hyperparameter tuning)
            - **Experiment Tracking**: MLflow
            - **Explainability**: SHAP values
            - **Deployment**: Docker + FastAPI + Render
            
            ### ⚠️ Disclaimer
            
            This tool is for **educational and research purposes only**.  Predictions should not be used as a 
            substitute for professional medical diagnosis.  Always consult qualified healthcare providers.
            """)
    
    # Footer
    gr.Markdown("""
    ---
    **🔗 Links:**  
    [API Endpoint](https://mlops-finalproject-latest.onrender.com) • 
    [API Documentation](https://mlops-finalproject-latest.onrender.com/docs) • 
    [GitHub Repository](https://github.com/MariaGoico/Final-Project-MLOps)
    
    *Built with ❤️ using FastAPI, XGBoost, and Gradio*
    """)

if __name__ == "__main__":
    demo.launch(share=True)