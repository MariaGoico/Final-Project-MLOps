import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from logic.breast_cancer_predictor import BreastCancerPredictor
from io import StringIO
import traceback

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API para predicción de cáncer de mama usando XGBoost",
    version="1.0.0"
)

try:
    predictor = BreastCancerPredictor("artifacts")
    EXPECTED_FEATURES = predictor.model.num_features()
    print(f"✅ Modelo cargado.  Espera {EXPECTED_FEATURES} features")
except Exception as e: 
    print(f"❌ Error cargando modelo: {e}")
    predictor = None
    EXPECTED_FEATURES = 30

def clean_dataframe(df):
    """
    Limpia el dataframe eliminando columnas vacías y validando datos
    """
    original_shape = df.shape
    
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"🧹 Eliminando columnas:  {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    df = df.dropna(axis=1, how='all')
    
    empty_name_cols = [col for col in df.columns if str(col).strip() == '']
    if empty_name_cols:
        df = df.drop(columns=empty_name_cols)
    
    print(f"📊 Shape:  {original_shape} → {df.shape}")
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    missing = df.isnull().sum()
    if missing. sum() > 0:
        missing_info = missing[missing > 0].to_dict()
        raise ValueError(f"CSV contiene valores faltantes: {missing_info}")
    
    current_features = df.shape[1]
    
    if current_features < EXPECTED_FEATURES:
        n_missing = EXPECTED_FEATURES - current_features
        print(f"⚠️ Faltan {n_missing} features, agregando columnas dummy...")
        for i in range(n_missing):
            df[f'dummy_{i}'] = 0.0
    
    elif current_features > EXPECTED_FEATURES:
        print(f"⚠️ Sobran {current_features - EXPECTED_FEATURES} features, tomando las primeras {EXPECTED_FEATURES}...")
        df = df.iloc[:, :EXPECTED_FEATURES]
    
    print(f"✅ Shape final: {df.shape}")
    return df


@app.get("/")
async def home():
    """Endpoint principal con información de la API"""
    return {
        "message": "Breast Cancer Prediction API",
        "status":  "ready" if predictor else "error:  model not loaded",
        "model_features": EXPECTED_FEATURES,
        "endpoints": {
            "POST /predict": "Subir CSV para predicción",
            "GET /health": "Estado de salud de la API",
            "GET /info": "Información del modelo"
        }
    }

@app.get("/info")
async def info():
    """Información detallada del modelo"""
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


@app. post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predice diagnóstico de cáncer de mama desde un CSV
    
    El CSV debe contener 30 columnas con las features del dataset Wisconsin Breast Cancer. 
    Las columnas 'Unnamed' y valores vacíos se eliminan automáticamente.
    """
    if not predictor: 
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if not file. filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV (. csv extension)"
        )
    
    try:
        contents = await file. read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        print(f"\n📁 Archivo recibido: {file.filename}")
        print(f"📊 Shape original: {df.shape}")
        
        df = clean_dataframe(df)
        
        predictions = []
        for i, row in df.iterrows():
            try:
                pred, prob = predictor.predict_with_confidence(row.values)
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
                "features_used": EXPECTED_FEATURES
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
        print(f"❌ Error inesperado: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
