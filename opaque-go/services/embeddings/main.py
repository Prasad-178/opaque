#!/usr/bin/env python3
"""
Local Embedding Service using ONNX Runtime.
Fast, private embeddings without external API calls.
"""
import os
import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer

app = FastAPI(title="Opaque Embedding Service", version="1.0.0")

# Model configuration
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_PATH = os.getenv("MODEL_PATH", None)  # Optional: path to local ONNX model

# Global model state
tokenizer = None
session = None
model_loaded = False


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    latency_ms: float


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling to get sentence embeddings."""
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


@app.on_event("startup")
async def load_model():
    """Load the ONNX model on startup."""
    global tokenizer, session, model_loaded

    print(f"Loading model: {MODEL_NAME}")
    start = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load ONNX model
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"Loading ONNX model from: {MODEL_PATH}")
        session = ort.InferenceSession(MODEL_PATH)
    else:
        # Export model to ONNX if not provided
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        print("Loading model with optimum ONNX export...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        session = model.model

    model_loaded = True
    print(f"Model loaded in {time.time() - start:.2f}s")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for input texts."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    start = time.time()

    # Tokenize
    encoded = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
    )

    # Pool and normalize
    embeddings = mean_pooling(outputs[0], encoded["attention_mask"])

    if request.normalize:
        embeddings = normalize_embeddings(embeddings)

    latency_ms = (time.time() - start) * 1000

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        model=MODEL_NAME,
        latency_ms=latency_ms
    )


@app.post("/embed/single")
async def embed_single(text: str, normalize: bool = True):
    """Convenience endpoint for single text embedding."""
    response = await embed(EmbedRequest(texts=[text], normalize=normalize))
    return {
        "embedding": response.embeddings[0],
        "dimension": response.dimension,
        "latency_ms": response.latency_ms
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
