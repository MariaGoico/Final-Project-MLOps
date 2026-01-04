import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from logic. breast_cancer_predictor import BreastCancerPredictor
from io import StringIO
import traceback

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for breast cancer prediction using XGBoost",
    version="1.0.0"
)

try:
    predictor = BreastCancerPredictor("artifacts")
    EXPECTED_FEATURES = predictor.model.num_features()
    print(f"✅ Model loaded.  Expects {EXPECTED_FEATURES} features")
except Exception as e:  
    print(f"❌ Error loading model: {e}")
    predictor = None
    EXPECTED_FEATURES = 30

def clean_dataframe(df):
    """
    Cleans the dataframe by removing empty columns and validating data
    """
    original_shape = df.shape
    
    # Remove 'Unnamed' columns (created by extra commas)
    unnamed_cols = [col for col in df. columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"🧹 Removing columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    # Remove completely empty columns (all NaN)
    df = df.dropna(axis=1, how='all')
    
    # Remove columns without name or only spaces
    empty_name_cols = [col for col in df.columns if str(col).strip() == '']
    if empty_name_cols:
        df = df.drop(columns=empty_name_cols)
    
    print(f"📊 Shape:  {original_shape} → {df.shape}")
    
    # Verify all columns are numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to convert
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing values (after cleaning)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_info = missing[missing > 0].to_dict()
        raise ValueError(f"CSV contains missing values: {missing_info}")
    
    # Adjust number of features
    current_features = df.shape[1]
    
    if current_features < EXPECTED_FEATURES:
        # Add dummy columns with 0
        n_missing = EXPECTED_FEATURES - current_features
        print(f"⚠️ Missing {n_missing} features, adding dummy columns...")
        for i in range(n_missing):
            df[f'dummy_{i}'] = 0.0
    
    elif current_features > EXPECTED_FEATURES:
        # Take only the first N features
        print(f"⚠️ {current_features - EXPECTED_FEATURES} extra features, taking first {EXPECTED_FEATURES}...")
        df = df.iloc[:, :EXPECTED_FEATURES]
    
    print(f"✅ Final shape: {df.shape}")
    return df


@app.get("/")
async def home():
    """Main endpoint with API information"""
    return {
        "message": "Breast Cancer Prediction API",
        "status": "ready" if predictor else "error:  model not loaded",
        "model_features": EXPECTED_FEATURES,
        "endpoints": {
            "POST /predict": "Upload CSV for prediction",
            "GET /health": "API health status",
            "GET /info": "Model information"
        }
    }


@app.get("/health")
async def health():
    """Checks API health"""
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded":  predictor is not None,
        "expected_features": EXPECTED_FEATURES
    }


@app.get("/info")
async def info():
    """Detailed model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "expected_features": EXPECTED_FEATURES,
        "threshold": float(predictor.threshold),
        "classes": {
            "0": "Benign (B)",
            "1": "Malignant (M)"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts breast cancer diagnosis from a CSV file
    
    The CSV must contain 30 columns with features from the Wisconsin Breast Cancer dataset. 
    'Unnamed' columns and empty values are automatically removed.
    """
    if not predictor:  
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.  Please check server logs."
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV (. csv extension)"
        )
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        print(f"\n📁 File received: {file.filename}")
        print(f"📊 Original shape: {df.shape}")
        
        # Clean and validate data
        df = clean_dataframe(df)
        
        # Make predictions
        predictions = []
        for i, row in df.iterrows():
            try:
                pred, prob = predictor.predict_with_confidence(row. values)
                predictions.append({
                    "row": int(i),
                    "prediction": int(pred),
                    "diagnosis": "Malignant (M)" if pred == 1 else "Benign (B)",
                    "probability": round(float(prob), 4),
                    "confidence": f"{float(prob) * 100:.2f}%"
                })
            except Exception as e:  
                predictions.append({
                    "row": int(i),
                    "error": str(e)
                })
        
        # Count results
        success_count = sum(1 for p in predictions if "error" not in p)
        error_count = len(predictions) - success_count
        
        return {
            "success": True,
            "file":  file.filename,
            "predictions": predictions,
            "summary": {
                "total_rows": len(predictions),
                "successful": success_count,
                "errors": error_count,
                "features_used":  EXPECTED_FEATURES
            }
        }
    
    except ValueError as e:  
        raise HTTPException(status_code=400, detail=str(e))
    
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(e)}"
        )
    
    except Exception as e:
        print(f"❌ Unexpected error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )