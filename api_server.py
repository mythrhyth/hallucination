from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import inspect

# Existing imports
from app.api.router import api_router  
from injectionModel.injection_heuristics_pipeline import HeuristicInjectionPipeline
from injectionModel.adverserial_pipeline import AdversarialInjectionPipeline
from injectionModel.prompting_pipeline import PromptInjectionPipeline
from injectionModel.sentiment_pipeline import SentimentInjectionPipeline

# NEW IMPORT for Recursive Hybrid Detection
from app.api.detetction.recursive_hybrid import RecursiveHybridModel 

app = FastAPI(
    title="Hallucination Detection & Injection API",
    description=(
        "Unified API for both LSTM-based and Recursive Hybrid hallucination detection "
        "and injection pipelines (heuristic, adversarial, prompting, sentiment-based)."
    ),
    version="2.1.0"
)

# ---------------- MIDDLEWARE ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTERS ----------------
# Existing LSTM-based detection router
app.include_router(api_router, prefix="/api/detection", tags=["Detection"])

# Initialize Injection Pipelines
heuristic_pipeline = HeuristicInjectionPipeline()
adversarial_pipeline = AdversarialInjectionPipeline()
prompt_pipeline = PromptInjectionPipeline()
sentiment_pipeline = SentimentInjectionPipeline()

# Initialize Recursive Hybrid Detection Model
recursive_model = RecursiveHybridModel()  # ✅ path to your saved model

# ---------- Request Models ----------
class DetectionRequest(BaseModel):
    context: str
    question: str
    reasoning_steps: List[str]

class InjectionData(BaseModel):
    context: str
    question: str
    reasoning_steps: List[str]

class HeuristicRequest(BaseModel):
    data: InjectionData
    mode: str = "single"
    pure: bool = False

class AdversarialRequest(BaseModel):
    data: InjectionData
    mode: str = "single"
    strategy: str = "random"
    max_edits: int = 1
    num_variants: int = 5
    pure: bool = False

class PromptRequest(BaseModel):
    data: InjectionData
    mode: str = "single"
    pure: bool = False

class SentimentRequest(BaseModel):
    data: InjectionData
    mode: str = "single"
    pure: bool = False

# ---------- ROOT ----------
@app.get("/")
def root():
    return {
        "message": "Hallucination Detection & Injection API is running!",
        "available_routes": {
            "Detection": {
                "LSTM": "/api/detection/*",
                "RecursiveHybrid": "/api/detection/recursive_hybrid/predict",
            },
            "Injection": {
                "Heuristic": "/inject/heuristic",
                "Adversarial": "/inject/adversarial",
                "Sentiment": "/inject/sentiment",
                "Prompting": "/inject/prompting",
            }
        },
    }

# ---------------- Recursive Hybrid Detection ----------------
@app.post("/api/detection/recursive_hybrid/predict", tags=["Detection"])
def detect_recursive_hybrid(request: DetectionRequest):
    """
    Detect hallucinations using the Recursive Hybrid Model.
    """
    try:
        data = {
            "context": request.context,
            "question": request.question,
            "reasoning_steps": request.reasoning_steps
        }

        # Run the recursive hybrid model prediction
        results = recursive_model.predict(data)

        return {
            "status": "success",
            "type": "recursive_hybrid",
            "context": request.context,
            "question": request.question,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------- Injection Endpoints ----------------
@app.post("/inject/heuristic", tags=["Injection"])
def inject_heuristic(request: HeuristicRequest):
    """Inject hallucinations using heuristic rules."""
    try:
        data_dict = {
            "context": request.data.context,
            "question": request.data.question,
            "reasoning_steps": request.data.reasoning_steps
        }
        result = heuristic_pipeline.inject(data=data_dict, mode=request.mode, pure=request.pure)
        return {"status": "success", "type": "heuristic", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/inject/adversarial", tags=["Injection"])
def inject_adversarial(request: AdversarialRequest):
    """Inject hallucinations using adversarial transformations."""
    try:
        data_dict = {
            "context": request.data.context,
            "question": request.data.question,
            "reasoning_steps": request.data.reasoning_steps
        }
        result = adversarial_pipeline.inject(
            data=data_dict,
            mode=request.mode,
            strategy=request.strategy,
            max_edits=request.max_edits,
            num_variants=request.num_variants,
            pure=request.pure
        )
        return {"status": "success", "type": "adversarial", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/inject/sentiment", tags=["Injection"])
def inject_sentiment(request: SentimentRequest):
    """Inject hallucinations using sentiment-based flips."""
    try:
        data_dict = {
            "context": request.data.context,
            "question": request.data.question,
            "reasoning_steps": request.data.reasoning_steps
        }
        inject_fn = sentiment_pipeline.inject
        sig = inspect.signature(inject_fn)
        kwargs = {"data": data_dict, "pure": request.pure}
        if "num_steps_to_alter" in sig.parameters:
            kwargs["num_steps_to_alter"] = 1
        elif "num_flips" in sig.parameters:
            kwargs["num_flips"] = 1
        elif "mode" in sig.parameters:
            kwargs["mode"] = request.mode

        result = inject_fn(**kwargs)
        return {"status": "success", "type": "sentiment", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/inject/prompting", tags=["Injection"])
def inject_prompt(request: PromptRequest):
    """Inject hallucinations using prompt-based techniques."""
    try:
        data_dict = {
            "context": request.data.context,
            "question": request.data.question,
            "reasoning_steps": request.data.reasoning_steps
        }
        result = prompt_pipeline.inject(data=data_dict, mode=request.mode, pure=request.pure)
        return {"status": "success", "type": "prompt", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------- Run Server ----------------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
