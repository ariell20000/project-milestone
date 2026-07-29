from __future__ import annotations

from functools import lru_cache

import torch
from transformers import pipeline

from score_model import THEME_COLS

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@lru_cache(maxsize=1)
def get_classifier():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=MODEL_NAME, device=device)


def infer_theme_scores(lyrics: str) -> dict[str, float]:
    text = lyrics.strip() if lyrics else ""
    if not text:
        text = "no lyrics"
    result = get_classifier()(text, candidate_labels=THEME_COLS, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))
