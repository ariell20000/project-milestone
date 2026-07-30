from __future__ import annotations

from functools import lru_cache

import torch
from transformers import pipeline

from score_model import THEME_COLS

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3/...) GPU backend - torch.cuda is never
        # available on Mac, so without this check the classifier would
        # silently fall back to slow CPU-only inference on every Mac.
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_classifier():
    device = _select_device()
    if device in ("cuda", "mps"):
        # Half precision roughly doubles GPU throughput and is well
        # supported on Apple Silicon's MPS backend; fall back to default
        # precision if a given GPU/driver combination rejects it.
        try:
            return pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=device,
                torch_dtype=torch.float16,
            )
        except Exception:
            pass
    return pipeline("zero-shot-classification", model=MODEL_NAME, device=device)


def infer_theme_scores(lyrics: str) -> dict[str, float]:
    text = lyrics.strip() if lyrics else ""
    if not text:
        text = "no lyrics"
    result = get_classifier()(text, candidate_labels=THEME_COLS, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))
