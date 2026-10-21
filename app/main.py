"""
Polyglot-120 API: FastAPI service with confidence-based fallback routing.

POST /identify  {"text": "...", "top_k": 3}
  -> primary transformer prediction; if max confidence < CONF_THRESHOLD the
     request is routed to a fallback path (fastText lid.176 if available,
     otherwise "und" with a low-confidence flag). Every response reports which
     path answered and the latency in milliseconds.

GET /health

Environment:
  MODEL_ID        HF Hub id or local path of the fine-tuned model (required)
  CONF_THRESHOLD  float, default 0.70 (pick from reports/REPORT.md)
  FASTTEXT_PATH   path to lid.176.ftz (optional)
"""
import os
import time
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "model")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))
FASTTEXT_PATH = os.environ.get("FASTTEXT_PATH", "lid.176.ftz")
MAX_LENGTH = 64

app = FastAPI(title="Polyglot-120", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()


def _num_threads() -> int:
    """Threads for CPU inference. Containers often report the host's CPU count,
    so prefer an explicit TORCH_NUM_THREADS, then the CPU affinity mask."""
    v = os.environ.get("TORCH_NUM_THREADS")
    if v:
        return max(1, int(v))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


torch.set_num_threads(_num_threads())
# config keys may be str or int depending on the transformers version
ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}

_ft = None
if os.path.exists(FASTTEXT_PATH):
    try:
        import fasttext  # type: ignore
        _ft = fasttext.load_model(FASTTEXT_PATH)
        print(f"[polyglot] fastText fallback loaded from {FASTTEXT_PATH}")
    except Exception as e:  # fallback stays disabled
        print("[polyglot] WARNING: fastText fallback failed to load:", e)
else:
    print(f"[polyglot] WARNING: no fastText model at {FASTTEXT_PATH}; low-confidence inputs will return 'und'")


def fasttext_top1(text: str):
    """Top-1 fastText label and probability, or (None, None) if unavailable.

    fastText 0.9.2's Python `predict()` ends with `np.array(probs, copy=False)`,
    which raises on NumPy 2.x. Call the C++ binding directly (same arguments
    `predict()` would pass), and fall back to the Python wrapper for other builds.
    """
    if _ft is None:
        return None, None
    line = text.replace("\n", " ").strip() + "\n"
    try:
        preds = _ft.f.predict(line, 1, 0.0, "strict")       # list of (prob, "__label__xx")
        if preds:
            prob, lab = preds[0]
            return lab.replace("__label__", ""), float(prob)
        return None, None
    except Exception:
        pass
    try:
        labs, probs = _ft.predict(line.strip(), k=1)
        if len(labs):
            return labs[0].replace("__label__", ""), float(probs[0])
    except Exception as e:
        print("[polyglot] WARNING: fastText predict failed:", e)
    return None, None


class IdentifyIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=10)


class Candidate(BaseModel):
    language: str
    confidence: float


class IdentifyOut(BaseModel):
    language: str
    confidence: float
    candidates: List[Candidate]
    routed_to_fallback: bool
    fallback_language: Optional[str] = None
    fallback_confidence: Optional[float] = None
    threshold: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device,
            "threshold": CONF_THRESHOLD, "fallback": "fasttext" if _ft else "und"}


@app.post("/identify", response_model=IdentifyOut)
@torch.no_grad()
def identify(inp: IdentifyIn):
    t0 = time.perf_counter()
    text = inp.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    probs = torch.softmax(model(**enc).logits[0], dim=-1)
    top = torch.topk(probs, k=min(inp.top_k, probs.numel()))
    cands = [Candidate(language=ID2LABEL[int(i)], confidence=float(p))
             for p, i in zip(top.values, top.indices)]
    best = cands[0]

    routed, fb_lang, fb_conf = False, None, None
    if best.confidence < CONF_THRESHOLD:
        routed = True
        # The fallback must never fail the request: any problem degrades to "und".
        fb_lang, fb_conf = fasttext_top1(text)
        if not fb_lang:
            fb_lang = "und"

    return IdentifyOut(
        language=fb_lang if routed else best.language,
        confidence=best.confidence,
        candidates=cands,
        routed_to_fallback=routed,
        fallback_language=fb_lang,
        fallback_confidence=fb_conf,
        threshold=CONF_THRESHOLD,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
