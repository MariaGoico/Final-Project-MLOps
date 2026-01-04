import gradio as gr
import requests
import pandas as pd
from io import StringIO
import os

# API URL on Render
API_URL = "https://mlops-finalproject-latest. onrender.com"

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
            status = "🟢 Online" if data. get("model_loaded") else "🟡 Online (model not loaded)"
            return status
        return "🔴 Offline"
    except:
        return "🔴 Cannot connect"

def predict_csv(file_path):
    """Performs prediction from a CSV file"""
    if not file_path:  
        return "❌ Please upload a CSV file", None, None
    
    try:  
        # Read and show CSV preview
        df_preview = pd.read_csv(file_path)
        preview_html = df_preview.head(5).to_html(index=False, classes="dataframe")
        
        # Send to API
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            files = {"file": (filename, f, "text/csv")}
            
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
            if response. status_code != 200:
                try:  
                    detail = response.json().get('detail', 'Unknown error')
                except:
                    detail = response. text
                return f"❌ API Error ({response.status_code}): {detail}", None, preview_html
            
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
            results_html = results_df.to_html(index=False, classes="dataframe")
            
            # Summary
            summary_text = f"""
            ## 📊 Prediction Summary
            
            - **File**: {result.get('file')}
            - **Total rows**: {summary.get('total_rows', 0)}
            - **Successful predictions**: {summary.get('successful', 0)}
            - **Errors**: {summary.get('errors', 0)}
            - **Features used**: {summary.get('features_used', 0)}
            
            ### Diagnosis Distribution:  
            - **Malignant (M)**: {sum(1 for p in predictions if p. get('prediction') == 1)}
            - **Benign (B)**: {sum(1 for p in predictions if p.get('prediction') == 0)}
            """
            
            return summary_text, results_html, preview_html
            
    except requests.exceptions. Timeout:
        return "❌ Timeout:  The API took too long to respond", None, None
    except requests.exceptions.ConnectionError:
        return "❌ Connection error: Could not connect to the API", None, None
    except Exception as e: 
        return f"❌ Unexpected error: {str(e)}", None, None

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
        
        result = response. json()
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
        
        - **Diagnosis**: {diagnosis}
        - **Probability**: {probability}
        - **Confidence**: {confidence}
        
        {"⚠️ **Warning**:  Potentially malignant case detected" if pred['prediction'] == 1 else "✅ The case appears to be benign"}
        """
        
    except ValueError:
        return "❌ Error:  Make sure to enter only numbers separated by commas"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# Custom CSS
custom_css = """
.dataframe {
    width: 100%;
    border-collapse: collapse;
    margin:  20px 0;
}
.dataframe th {
    background-color: #4CAF50;
    color:  white;
    padding: 12px;
    text-align:  left;
}
.dataframe td {
    padding: 10px;
    border-bottom:  1px solid #ddd;
}
. dataframe tr:hover {
    background-color: #f5f5f5;
}
"""

# Create Gradio interface
with gr. Blocks(css=custom_css, theme=gr.themes. Soft()) as demo:
    gr.Markdown("""
    # 🏥 Breast Cancer Prediction API - Gradio Interface
    
    Breast cancer prediction system using XGBoost trained on the Wisconsin Breast Cancer dataset.
    """)
    
    # API status
    with gr.Row():
        api_status = gr.Textbox(label="API Status", value=check_api_health(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(check_api_health, outputs=api_status)
    
    # Tabs
    with gr.Tabs():
        # Tab 1: CSV Prediction
        with gr.Tab("📁 CSV Prediction"):
            gr.Markdown("""
            ### Instructions: 
            1. Upload a CSV file with 30 features from the Wisconsin Breast Cancer dataset
            2. The file can have headers (column names) or not
            3. Each row represents a case to predict
            """)
            
            with gr. Row():
                with gr.Column(scale=1):
                    csv_input = gr.File(
                        label="Upload CSV file",
                        file_types=[".csv"],
                        type="filepath"
                    )
                    predict_csv_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    summary_output = gr.Markdown(label="Summary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📄 CSV Preview")
                    csv_preview = gr.HTML(label="Preview")
                
                with gr. Column():
                    gr. Markdown("### 📊 Prediction Results")
                    results_table = gr.HTML(label="Results")
            
            predict_csv_btn.click(
                predict_csv,
                inputs=csv_input,
                outputs=[summary_output, results_table, csv_preview]
            )
        
        # Tab 2: Manual Prediction
        with gr.Tab("✍️ Manual Prediction"):
            gr.Markdown("""
            ### Instructions:
            Enter the 30 feature values separated by commas. 
            
            **Example:**
            ```
            17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
            ```
            
            **Features (in order):**
            1-10: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean  
            11-20: radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se  
            21-30: radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
            """)
            
            features_input = gr.Textbox(
                label="Feature Values (30 values separated by commas)",
                placeholder="17.99,10.38,122.8,1001,0.1184,... ",
                lines=3
            )
            
            predict_manual_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
            manual_output = gr.Markdown(label="Result")
            
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
                label="Example cases"
            )
        
        # Tab 3: Model Information
        with gr.Tab("ℹ️ Model Info"):
            gr.Markdown("### Model Information")
            
            info_btn = gr.Button("📊 Get Model Information", size="lg")
            info_output = gr.JSON(label="Model Details")
            
            def get_info_formatted():
                info = get_api_info()
                if info:
                    return info
                return {"error": "Could not retrieve model information"}
            
            info_btn.click(get_info_formatted, outputs=info_output)
            
            gr.Markdown("""
            ### 📚 About the Wisconsin Breast Cancer Dataset
            
            The model has been trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, 
            which contains features computed from digitized images of fine needle aspirate (FNA) 
            of breast masses. 
            
            **Features (30 in total):**
            - 10 characteristics are calculated for each cell nucleus
            - For each characteristic, 3 values are computed:  mean, standard error, and worst (largest value)
            - Total: 10 × 3 = 30 features
            
            **Classes:**
            - **M (Malignant)**: Malignant tumor (cancer)
            - **B (Benign)**: Benign tumor (non-cancerous)
            
            **Technology:**
            - Model: XGBoost Classifier
            - Optimization: Optuna (hyperparameter tuning)
            - Tracking: MLflow
            - Interpretability: SHAP values
            """)
    
    gr.Markdown("""
    ---
    **API URL**: https://mlops-finalproject-latest.onrender.com  
    **Documentation**: [FastAPI Docs](https://mlops-finalproject-latest.onrender.com/docs)  
    **Repository**: [GitHub](https://github.com/MariaGoico/Final-Project-MLOps)
    """)

if __name__ == "__main__": 
    demo.launch(share=True)