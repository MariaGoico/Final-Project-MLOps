import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi import Request
from pathlib import Path
import io
from PIL import Image
from PIL import UnidentifiedImageError

from logic.utilities import predict, resize


app = FastAPI(
    title="API of the Lab 1 using FastAPI",
    description="API to perform preprocessing on images",
    version="1.0.0",
)

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML homepage.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ---------------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_class(
    file: UploadFile = File(...),
):
    """
    Predict the class for an uploaded pet image using the trained ONNX model.

    Args:
        file: The image file to classify

    Returns:
        dict: The predicted class and confidence (if available)
    """
    try:
        # Read the uploaded file to validate it's an image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Predict using the ONNX model
        result, confidence = predict(image)

        response = {"predicted_class": result, "filename": file.filename}

        # Add confidence if available
        if confidence is not None:
            response["confidence"] = round(confidence, 4)

        return response

    except UnidentifiedImageError:
        return {"error": "Uploaded file is not a valid image."}

    except OSError as e:# pragma: no cover
        return {"error": f"Failed to read the image: {str(e)}"}

