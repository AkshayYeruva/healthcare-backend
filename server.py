from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import io
import os
from PIL import Image

from symptom_predict import predict_disease
from image_predict import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    symptoms: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/symptoms")
def symptoms_endpoint(req: SymptomRequest):
    if not req.symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms cannot be empty")
    return predict_disease(req.symptoms)


@app.post("/predict/image")
async def image_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return predict_image(image)


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)