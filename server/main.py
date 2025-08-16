from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import threading
from inference import (
    TranslationInference,
)
import torch
import uvicorn
from contextlib import asynccontextmanager


translator = None
translator_lock = threading.Lock()

REPO_ID = "DrDrunkenstein22/mbart-kn-en-finetune"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator
    try:
        print("hf hub loading...")
        translator = TranslationInference(
            repo_id=REPO_ID,
            device=DEVICE,
        )
        print("translation model loaded")
    except Exception as e:
        print(f"error loading model: {e}")
        translator = None

    yield  # This separates startup from shutdown

    # Shutdown - Clean up resources if needed
    print("Shutting down translation service...")
    # Add any cleanup code here if needed


app = FastAPI(title="Translation api", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextTranslationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    success: bool
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Translation api", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": translator is not None,
        "device": DEVICE,
    }


@app.post("/translate/text", response_model=TranslationResponse)
async def translate_text(request: TextTranslationRequest):
    """translate text"""

    if translator is None:
        raise HTTPException(status_code=503, detail="Translation model not loaded")

    print(f"translate text: '{request.text}'")

    with translator_lock:
        translated = translator.translate(
            text=request.text, max_length=request.max_length
        )

    print(f"translation result: '{translated}'")

    return TranslationResponse(
        original_text=request.text, translated_text=translated, success=True
    )


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
