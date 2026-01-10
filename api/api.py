import sys
import os
from pathlib import Path
import threading
from typing import Optional

# Añadir directorio raíz al path para encontrar logic/
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, Header, UploadFile, HTTPException, Request
from logic.breast_cancer_predictor import BreastCancerPredictor
from api.metrics_tracker import ModelMetricsTracker, prediction_confidence_by_diagnosis
from io import StringIO
import traceback
import time
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CREATE APP
# ========================================
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for breast cancer prediction using XGBoost",
    version="1.0.0"
)

# ============================================
# RETRAINING STATE (Global Variables)
# ============================================
retraining_lock = threading.Lock()
retraining_in_progress = False
retraining_status = {
    "status": "idle",
    "last_triggered": None,
    "last_completed": None,
    "trigger_reason": None,
    "deployed":  None,
    "error": None
}

# ============================================
# RETRAINING FUNCTION
# ============================================
def run_retraining(trigger_reason:  str):
    """
    Execute model retraining pipeline
    """
    global retraining_in_progress, retraining_status, predictor
    
    with retraining_lock:
        if retraining_in_progress: 
            logger.warning("⚠️ Retraining already in progress, skipping")
            return
        retraining_in_progress = True
    
    retraining_status['status'] = 'in_progress'
    retraining_status['error'] = None
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 STARTING RETRAINING PIPELINE")
        logger.info(f"   Trigger:  {trigger_reason}")
        logger.info(f"   Timestamp: {datetime.now()}")
        logger.info(f"{'='*60}\n")
        
        # Import and run retraining pipeline
        from logic.retraining_pipeline import RetrainingPipeline
        
        pipeline = RetrainingPipeline(artifacts_dir="artifacts")
        results = pipeline.run(trigger_reason)
        
        if results['success']:
            
            # Check if new model was deployed
            if results.get('deployed', False):
                logger.info("✅ New model deployed via CI/CD")
                logger.info("   Waiting for Render to redeploy with new artifacts...")
                
                # The container will be replaced by Render's CI/CD
                # This instance will be shut down
                retraining_status['status'] = 'completed'
                retraining_status['deployed'] = True
                retraining_status['deployment_method'] = 'cicd_triggered'
                retraining_status['last_completed'] = datetime.now().isoformat()
                
            else:
                # Model not better, no deployment
                logger.info("⚠️ New model not better, keeping current model")
                
                retraining_status['status'] = 'completed'
                retraining_status['deployed'] = False
                retraining_status['last_completed'] = datetime.now().isoformat()
            
            retraining_status['results'] = results
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ RETRAINING SUCCESSFUL")
            logger.info(f"   Deployed: {results.get('deployed', False)}")
            logger.info(f"{'='*60}\n")
        else:
            raise Exception(f"Retraining failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e: 
        logger.error(f"\n{'='*60}")
        logger.error(f"❌ RETRAINING FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"{'='*60}\n")
        logger.error(traceback.format_exc())
        
        retraining_status['status'] = 'failed'
        retraining_status['error'] = str(e)
    
    finally:
        retraining_in_progress = False

# ========================================
# PROMETHEUS METRICS
# ========================================

# ===== BASIC METRICS =====
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status']
)

predictions_by_diagnosis_total = Counter(
    'predictions_by_diagnosis_total',
    'Total predictions by diagnosis type',
    ['diagnosis']
)

model_loaded = Gauge(
    'model_loaded',
    'Whether the model is loaded (1=loaded, 0=not loaded)'
)

expected_features_gauge = Gauge(
    'expected_features',
    'Number of features expected by the model'
)

rows_processed_total = Counter(
    'rows_processed_total',
    'Total number of rows processed',
    ['status']
)

# ===== ADVANCED BUSINESS METRICS =====
prediction_confidence_histogram = Histogram(
    'prediction_confidence_score',
    'Distribution of prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent making predictions',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

request_latency = Histogram(
    'request_latency_seconds',
    'Total request processing time',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

csv_file_size_bytes = Histogram(
    'csv_file_size_bytes',
    'Size of uploaded CSV files',
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000]
)

prediction_errors_total = Counter(
    'prediction_errors_total',
    'Prediction errors by type',
    ['error_type']
)

predictions_by_hour = Counter(
    'predictions_by_hour_total',
    'Predictions by hour of day',
    ['hour']
)

diagnosis_ratio = Gauge(
    'malignant_benign_ratio',
    'Ratio of malignant to benign predictions'
)

rows_throughput_total = Counter(
    'rows_throughput_total',
    'Total rows processed for throughput calculation'
)

api_start_time = Gauge(
    'api_start_timestamp_seconds',
    'Timestamp when the API started'
)

api_health_score = Gauge(
    'api_health_score',
    'Overall API health score (0-100)'
)

endpoint_requests_total = Counter(
    'endpoint_requests_total',
    'Requests by endpoint',
    ['endpoint', 'method', 'status_code']
)

missing_values_total = Counter(
    'missing_values_total',
    'Total missing values encountered'
)

# ========================================
# INITIALIZE METRICS TRACKER
# ========================================
ENABLE_DRIFT_SIMULATION = os.environ.get("ENABLE_DRIFT_SIMULATION", "false").lower() == "true"

metrics_tracker = ModelMetricsTracker(
    window_size=1000,
    enable_simulation=ENABLE_DRIFT_SIMULATION
)

print(f"🔍 Drift simulation: {'ENABLED' if ENABLE_DRIFT_SIMULATION else 'DISABLED'}")

# ========================================
# MODEL LOADING
# ========================================
try:
    predictor = BreastCancerPredictor("artifacts")
    EXPECTED_FEATURES = predictor.model.num_features()
    print(f"✅ Model loaded.  Expects {EXPECTED_FEATURES} features")
    model_loaded.set(1)
    expected_features_gauge.set(EXPECTED_FEATURES)
    api_start_time.set(time.time())
    
    # ===== LOAD VALIDATION METRICS =====
    print("\n" + "="*60)
    print("📊 Loading validation metrics for monitoring...")
    print("="*60)
    
    try:
        validation_metrics_path = Path("artifacts/validation_metrics.json")
        
        if validation_metrics_path.exists():
            with open(validation_metrics_path, 'r') as f:
                validation_metrics = json.load(f)
            
            metrics_tracker.load_validation_metrics(validation_metrics)
            print("✅ Validation metrics loaded from artifacts/validation_metrics.json")
        else:
            print("⚠️  validation_metrics.json not found.  Using placeholder metrics.")
            metrics_tracker.load_validation_metrics({
                'f1_score': 0.0,
                'accuracy': 0.0,
                'precision':  0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                'confusion_matrix':  {'tp': 0, 'tn': 0, 'fp': 0, 'fn':  0},
                'model_info': {
                    'algorithm': 'XGBoost',
                    'version': 'unknown',
                    'training_date': 'unknown',
                    'features': str(EXPECTED_FEATURES)
                }
            })
    
    except Exception as e:
        print(f"⚠️  Error loading validation metrics: {e}")
        print("   Using placeholder metrics.")
    
    # ===== LOAD FEATURE BASELINE FOR DRIFT DETECTION =====
    print("\n" + "="*60)
    print("🔍 Loading feature baseline for drift detection...")
    print("="*60)
    
    try:
        baseline_path = Path("artifacts/feature_baseline.npz")
        
        if baseline_path.exists():
            baseline = np.load(baseline_path)
            metrics_tracker.set_feature_baseline(
                means=baseline['means'],
                stds=baseline['stds']
            )
            print("✅ Feature baseline loaded successfully")
        else:
            print("⚠️  feature_baseline.npz not found.")
            print("   Drift detection will be disabled.")
    
    except Exception as e: 
        print(f"⚠️  Error loading feature baseline: {e}")
        print("   Drift detection will be disabled.")
    
    print("="*60 + "\n")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())
    predictor = None
    EXPECTED_FEATURES = 30
    model_loaded.set(0)
    expected_features_gauge.set(EXPECTED_FEATURES)

# ========================================
# MIDDLEWARE
# ========================================
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all HTTP requests"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    endpoint_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code
    ).inc()
    
    if request.url.path == "/predict":
        request_latency.observe(duration)
    
    return response

# ========================================
# HELPER FUNCTIONS
# ========================================
def calculate_health_score():
    """Calculate overall API health score (0-100)"""
    score = 100
    
    if not predictor:
        score -= 50
    
    api_health_score.set(score)
    return score

def clean_dataframe(df):
    """Clean and validate dataframe"""
    original_shape = df.shape
    
    # Remove unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"🧹 Removing columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    # Remove all-empty columns
    df = df.dropna(axis=1, how='all')
    
    # Remove columns with empty names
    empty_name_cols = [col for col in df.columns if str(col).strip() == '']
    if empty_name_cols:
        df = df.drop(columns=empty_name_cols)
    
    print(f"📊 Shape:  {original_shape} → {df.shape}")
    
    # Convert to numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        missing_values_total.inc(missing)
        missing_info = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        raise ValueError(f"CSV contains missing values: {missing_info}")
    
    # Adjust features
    current_features = df.shape[1]
    
    if current_features < EXPECTED_FEATURES:
        n_missing = EXPECTED_FEATURES - current_features
        print(f"⚠️ Missing {n_missing} features, adding dummy columns...")
        for i in range(n_missing):
            df[f'dummy_{i}'] = 0.0
    elif current_features > EXPECTED_FEATURES:
        print(f"⚠️ {current_features - EXPECTED_FEATURES} extra features, taking first {EXPECTED_FEATURES}...")
        df = df.iloc[:, :EXPECTED_FEATURES]
    
    print(f"✅ Final shape: {df.shape}")
    return df

# ========================================
# ENDPOINTS
# ========================================
@app.get("/")
async def home():
    """Main endpoint with API information"""
    return {
        "message": "Breast Cancer Prediction API",
        "status": "ready" if predictor else "error:  model not loaded",
        "model_features": EXPECTED_FEATURES,
        "health_score": calculate_health_score(),
        "monitoring": {
            "validation_metrics_loaded": metrics_tracker.get_stats().get('has_baseline', False),
            "drift_detection_enabled": metrics_tracker.feature_baseline is not None,
            "drift_simulation":  ENABLE_DRIFT_SIMULATION
        },
        "endpoints": {
            "POST /predict": "Upload CSV for prediction",
            "GET /health": "API health status",
            "GET /info": "Model information",
            "GET /metrics": "Prometheus metrics",
            "GET /stats": "Current statistics",
            "POST /webhook/retrain": "Webhook for automatic retraining",
            "GET /retrain/status": "Retraining status",
            "POST /retrain/manual": "Manual retraining trigger"
        }
    }

@app.get("/health")
async def health():
    """API health check"""
    health_score = calculate_health_score()
    return {
        "status": "healthy" if predictor and health_score > 50 else "unhealthy",
        "model_loaded": predictor is not None,
        "expected_features": EXPECTED_FEATURES,
        "health_score":  health_score,
        "retraining_in_progress": retraining_in_progress
    }

@app.get("/info")
async def info():
    """Model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "expected_features": EXPECTED_FEATURES,
        "threshold": float(predictor.threshold),
        "classes": {
            "0": "Benign (B)",
            "1": "Malignant (M)"
        },
        "monitoring": {
            "validation_metrics_available": True,
            "drift_detection_enabled": metrics_tracker.feature_baseline is not None,
            "drift_simulation": ENABLE_DRIFT_SIMULATION
        }
    }

@app.get("/stats")
async def stats():
    """Current prediction statistics"""
    return metrics_tracker.get_stats()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        logger.info("="*60)
        logger.info("📊 /metrics CALLED")
        logger.info(f"   ENABLE_DRIFT_SIMULATION = {ENABLE_DRIFT_SIMULATION}")
        logger.info(f"   metrics_tracker.enable_simulation = {metrics_tracker.enable_simulation}")
        logger.info("="*60)
        # Calculate health and metrics
        calculate_health_score()
        logger.info("🔄 About to call calculate_metrics()")
        metrics_tracker.calculate_metrics()
        logger.info("✅ calculate_metrics() returned")

        content = generate_latest()
        content_str = content.decode('utf-8')
        for line in content_str.split('\n'):
            if 'data_drift_detected' in line and not line.startswith('#'):
                logger.info(f"   DRIFT METRIC: {line}")
        
        # Generate Prometheus format
        return Response(
            content=content,
            media_type=CONTENT_TYPE_LATEST
        )
    
    except Exception as e:
        logger.error(f"❌ Error generating metrics: {e}")
        logger.error(traceback.format_exc())
        
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )

@app.post("/predict")
async def predict(file: UploadFile = File(... )):
    """Predict breast cancer diagnosis from CSV file"""
    request_start_time = time.time()
    
    if not predictor:
        prediction_requests_total.labels(status='error').inc()
        prediction_errors_total.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        prediction_requests_total.labels(status='error').inc()
        prediction_errors_total.labels(error_type='invalid_file_type').inc()
        raise HTTPException(status_code=400, detail="File must be CSV")
    
    try:
        # Read file
        contents = await file.read()
        file_size = len(contents)
        csv_file_size_bytes.observe(file_size)
        
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logger.info(f"\n📁 File: {file.filename}, Size: {file_size} bytes, Shape: {df.shape}")
        
        df = clean_dataframe(df)
        
        current_hour = datetime.now().hour
        predictions = []
        batch_start_time = time.time()
        
        # Process each row
        for i, row in df.iterrows():
            try:
                # Predict with timing
                pred_start = time.time()
                pred, prob = predictor.predict_with_confidence(row.values)
                pred_duration = time.time() - pred_start
                
                # Update metrics
                prediction_latency.observe(pred_duration)
                diagnosis = "Malignant" if pred == 1 else "Benign"
                predictions_by_diagnosis_total.labels(diagnosis=diagnosis).inc()
                rows_processed_total.labels(status='success').inc()
                rows_throughput_total.inc()
                
                # Confidence metrics
                prediction_confidence_histogram.observe(float(prob))
                prediction_confidence_by_diagnosis.labels(diagnosis=diagnosis).observe(float(prob))
                
                # Hour tracking
                predictions_by_hour.labels(hour=str(current_hour)).inc()
                
                # Add to tracker (for ML metrics and drift)
                metrics_tracker.add_prediction(pred, prob, row.values)
                
                predictions.append({
                    "row": int(i),
                    "prediction": int(pred),
                    "diagnosis": "Malignant (M)" if pred == 1 else "Benign (B)",
                    "probability": round(float(prob), 4),
                    "confidence": f"{float(prob) * 100:.2f}%",
                    "latency_ms": round(pred_duration * 1000, 2)
                })
                
            except Exception as e:
                logger.error(f"❌ Prediction error on row {i}: {e}")
                rows_processed_total.labels(status='error').inc()
                prediction_errors_total.labels(error_type='prediction_error').inc()
                predictions.append({"row": int(i), "error": str(e)})
        
        batch_duration = time.time() - batch_start_time
        success_count = sum(1 for p in predictions if "error" not in p)
        error_count = len(predictions) - success_count
        
        # Update ratio gauge
        if metrics_tracker.benign_count > 0:
            diagnosis_ratio.set(metrics_tracker.malignant_count / metrics_tracker.benign_count)
        
        # Calculate metrics
        metrics_tracker.calculate_metrics()
        
        # Success
        prediction_requests_total.labels(status='success').inc()
        request_duration = time.time() - request_start_time
        
        return {
            "success":  True,
            "file":  file.filename,
            "predictions": predictions,
            "summary": {
                "total_rows": len(predictions),
                "successful": success_count,
                "errors": error_count,
                "features_used":  EXPECTED_FEATURES,
                "processing_time_seconds": round(batch_duration, 3),
                "request_time_seconds": round(request_duration, 3),
                "throughput_rows_per_second": round(len(predictions) / batch_duration, 2) if batch_duration > 0 else 0,
                "file_size_bytes": file_size
            },
            "statistics": {
                "malignant_count": sum(1 for p in predictions if p.get('prediction') == 1),
                "benign_count": sum(1 for p in predictions if p.get('prediction') == 0),
                "avg_confidence": round(np.mean([p.get('probability', 0) for p in predictions if 'probability' in p]), 4),
                "current_hour": current_hour
            }
        }
    
    except ValueError as e:
        prediction_requests_total.labels(status='error').inc()
        prediction_errors_total.labels(error_type='value_error').inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    except pd.errors.ParserError as e:
        prediction_requests_total.labels(status='error').inc()
        prediction_errors_total.labels(error_type='parser_error').inc()
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    
    except Exception as e:
        prediction_requests_total.labels(status='error').inc()
        prediction_errors_total.labels(error_type='unknown').inc()
        print(f"❌ Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ========================================
# RETRAINING ENDPOINTS
# ========================================

@app.post("/webhook/retrain")
async def webhook_retrain(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Webhook triggered by Grafana Cloud Alerting for automatic retraining
    """
    global retraining_in_progress, retraining_status
    
    # ===== AUTHENTICATION =====
    import base64
    
    # Get secret from environment (más seguro)
    webhook_secret = os.environ.get("WEBHOOK_SECRET", "your-webhook-secret")
    expected_auth = "Basic " + base64.b64encode(f"admin:{webhook_secret}".encode()).decode()
    
    if authorization != expected_auth:
        logger.warning(f"⚠️ Unauthorized webhook attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        payload = await request.json()
        logger.info(f"📥 Webhook payload received: {json.dumps(payload, indent=2)}")
        
        # ===== EXTRACT ALERT INFO =====
        # Grafana sends alerts in 'alerts' array
        alerts = payload.get('alerts', [])
        
        if not alerts:
            logger.info("No alerts in payload, ignoring")
            return {"status":  "no_alerts", "message": "No alerts to process"}
        
        # Process first firing alert
        firing_alerts = [a for a in alerts if a.get('status') == 'firing']
        
        if not firing_alerts: 
            logger.info(f"No firing alerts (found {len(alerts)} alerts with other status)")
            return {"status": "ignored", "reason": "No firing alerts"}
        
        alert = firing_alerts[0]
        alert_name = alert.get('labels', {}).get('alertname', 'Unknown Alert')
        alert_value = alert.get('values', {}).get('A', 'N/A')
        
        logger.info(f"🚨 ALERT FIRING: {alert_name} (value={alert_value})")
        
        # ===== CHECK IF ALREADY RETRAINING =====
        if retraining_in_progress: 
            logger.warning("⚠️ Retraining already in progress, skipping")
            return {
                "status": "already_in_progress",
                "message": "Retraining already running",
                "current_status": retraining_status
            }
        
        # ===== TRIGGER RETRAINING =====
        retraining_status['status'] = 'triggered'
        retraining_status['last_triggered'] = datetime.now().isoformat()
        retraining_status['trigger_reason'] = alert_name
        retraining_status['alert_value'] = str(alert_value)
        
        # Start retraining in background thread
        thread = threading.Thread(
            target=run_retraining,
            args=(alert_name,),
            daemon=True
        )
        thread.start()
        
        logger.info(f"✅ Retraining triggered by {alert_name}")
        
        return {
            "status":  "retraining_triggered",
            "alert":  alert_name,
            "timestamp": retraining_status['last_triggered'],
            "message": f"Automatic retraining started due to {alert_name}"
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    except Exception as e:
        logger.error(f"❌ Error processing webhook: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrain/status")
async def retrain_status():
    """Get current retraining status"""
    return {
        "in_progress": retraining_in_progress,
        **retraining_status
    }

@app.post("/retrain/manual")
async def manual_retrain():
    """Manually trigger retraining (for testing)"""
    global retraining_in_progress
    
    if retraining_in_progress: 
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    retraining_status['status'] = 'manual_trigger'
    retraining_status['last_triggered'] = datetime.now().isoformat()
    retraining_status['trigger_reason'] = 'Manual trigger'
    
    thread = threading.Thread(target=run_retraining, args=("ManualTrigger",))
    thread.daemon = True
    thread.start()
    
    return {
        "status": "retraining_started",
        "timestamp": retraining_status['last_triggered']
    }

# ========================================
# STARTUP
# ========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)