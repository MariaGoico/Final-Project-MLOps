# Breast Cancer Prediction — MLOps Final Project
[![CICD](https://github.com/MariaGoico/Final-Project-MLOps/actions/workflows/CICD.yml/badge.svg)](https://github.com/MariaGoico/Final-Project-MLOps/actions/workflows/CICD.yml)

**Authors:** Maria Goicoechea, Paula Pina & Joaquín Orradre

This project demonstrates the end-to-end development and deployment of a machine learning system for breast cancer diagnosis prediction. Leveraging the Wisconsin Breast Cancer dataset, the goal is to provide accurate predictions, robust monitoring, automated retraining, and a seamless user experience through modern MLOps practices.

## Features

- **Automated ML Pipeline:** Full training pipeline using Optuna for hyperparameter optimization.
- **Model Selection:** Comparison between XGBoost and TabNet, with XGBoost selected for deployment based on superior validation and test performance.
- **API Deployment:** Production-grade REST API built with FastAPI and deployed on Render.
- **Monitoring & Drift Detection:** Real-time metrics exposed via Prometheus, with dashboards and alerts in Grafana. Includes drift and fairness detection.
- **Automated Retraining:** Remote retraining triggered by drift or alert events via GitHub Actions, ensuring continuous model improvement without manual intervention.
- **User Interface:** Interactive, no-code Gradio UI deployed on Hugging Face Spaces for batch and manual predictions.
  
## Project Structure

```
├── .github/workflows/              
│   ├── CICD.yml                    # Main workflow for continuous integration and deployment

├── api/                            # FastAPI app and monitoring logic
│   ├── __init__.py                 
│   ├── api.py                      # Main API server (FastAPI)
│   ├── metrics_tracker.py          # Tracks and serves application/model metrics

├── artifacts/                      # Model artifacts and baseline data for serving/monitoring
│   ├── feature_baseline.json       
│   ├── feature_baseline.npz       
│   ├── metadata.json              
│   ├── model.json                 
│   ├── preprocessor.pkl 
│   ├── shap_global.json    
│   ├── threshold.json 
│   ├── validation_metrics.json 

├── data/
│   ├── data.csv                    # Main source data (Wisconsin Breast Cancer)

├── logic/                          # Core ML logic: training, evaluation, selection
│   ├── __init__.py                 
│   ├── breast_cancer_predictor.py  # Model loading, prediction code
│   ├── data_module.py              # Data processing routines and helpers
│   ├── evaluate.py                 # Model evaluation and comparison
│   ├── generate_baseline.py        
│   ├── retraining_pipeline.py      # Automated retraining triggering logic
│   ├── tabnet_model.py             # TabNet training pipeline
│   ├── utilities.py                
│   ├── xgboost_model.py            # XGBoost training pipeline

├── monitoring/
│   ├── .dockerignore               
│   ├── Dockerfile                  

├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis notebook

├── templates/
│   ├── home.html                   # HTML template used by API

├── tests/                          # Automated test suite
│   ├── __init__.py                 
│   ├── test_data_module.py         
│   ├── test_model.py               
│   ├── test_predictor.py           
│   ├── test_utilities.py           

├── Dockerfile                      # Main project Docker config (for the API/app)
├── Makefile                        # Automation: install, format, lint, test, all                     
```
## Setup and Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast package management and virtual environments.  
To get started, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/MariaGoico/Final-Project-MLOps.git
cd Final-Project-MLOps
```

### 2. Install Dependencies

Install all project dependencies and set up the virtual environment:

```bash
make install
```

### 3. Activate the Virtual Environment

- **On Windows (PowerShell):**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
- **On macOS/Linux (Bash):**
    ```bash
    source .venv/Scripts/activate
    ```

---

## Usage

### Quality Checks (Makefile)

The `Makefile` contains commands for running code quality and testing checks:

- **Auto-format all code with black:**  
  ```bash
  make format
  ```

- **Lint all code with pylint:**  
  ```bash
  make lint
  ```

- **Run all tests with pytest and generate a coverage report:**  
  ```bash
  make test
  ```

- **Run all local checks (format, lint, test):**  
  ```bash
  make all
  ```
---

## Local Training, Evaluation, and API Usage

### 1. Train and Select the Best Model (XGBoost vs. TabNet)

To reproduce the training pipeline and automatically select the best algorithm, run the following commands in your project root:

```bash
uv run python -m logic.tabnet_model

uv run python -m logic.xgboost_model

uv run python -m logic.evaluate
```

- This sequence will train both TabNet and XGBoost using Optuna hyperparameter search (30 runs each) and automatically select and save the best-performing model artifacts for production use.

### 2. Visualizing Optuna Runs in MLflow

After training, all Optuna optimization runs and metrics can be explored locally with [MLflow](https://mlflow.org/):

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to see detailed experiment/trial histories for each algorithm.

### 3. Running the Prediction API Locally

You can launch the FastAPI-based backend for prediction and metrics on your local machine:

```bash
uv run python -m api.api
```
- The API will expose `/predict` for predictions and `/metrics` for Prometheus-compatible monitoring.
- All endpoints and interactive docs are available at `http://localhost:8000/docs`.

### 4. Production Endpoints

- **Production API:**  
  [https://mlops-finalproject-latest.onrender.com](https://mlops-finalproject-latest.onrender.com)
- **Interactive Gradio UI:**  
  [https://huggingface.co/spaces/joaquinorradre/mlops-finalproject](https://huggingface.co/spaces/joaquinorradre/mlops-finalproject)
- **Prometheus Monitoring (scraping every 30s):**  
  [https://mlops-finalproject-prometheus.onrender.com](https://mlops-finalproject-prometheus.onrender.com)


## Monitoring and Model Management

- All evaluation and production metrics are published at `/metrics` (compatible with Prometheus).
- Prometheus automatically scrapes these metrics every 30 seconds and pushes them to Grafana Cloud for real-time visualization and automated drift/fairness alerting.
- Automated retraining can be triggered remotely via API/webhook or through Grafana alerts, thanks to the MLOps integration.

---

This project is for educational and research purposes only. Not intended for clinical use.

---

**Author:** [Your Name or Group]  
**GitHub:** [MariaGoico/Final-Project-MLOps](https://github.com/MariaGoico/Final-Project-MLOps)
