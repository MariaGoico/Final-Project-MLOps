import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="API Connection Test",
    description="Minimal API to test endpoint connections",
    version="1.0.0",
)

# Home Endpoint
@app.get("/")
async def home():
    print("Endpoint reached: /")
    return {"message": "Home endpoint reached successfully"}

# Predict Endpoint
@app.get("/predict")
async def predict():
    print("Endpoint reached: /predict")
    return {"message": "Predict endpoint reached successfully"}

if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
