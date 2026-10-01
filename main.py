"""Local, stateless sentiment inference service for the Chrome extension."""
from __future__ import annotations

import asyncio
import os
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

LABELS = ("negative", "neutral", "positive")


class Comment(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text cannot be blank")
        return value


class AnalyseRequest(BaseModel):
    comments: list[Comment] = Field(min_length=1, max_length=2000)


class Probabilities(BaseModel):
    negative: float
    neutral: float
    positive: float


class SentimentResult(BaseModel):
    id: str
    label: Literal["negative", "neutral", "positive"]
    confidence: float
    probabilities: Probabilities
    # An explanatory heuristic for neutral comments; this is not a fourth model class.
    neutral_context: Literal["question", "spam_or_bot_like", "informational"] | None = None


class AnalyseResponse(BaseModel):
    model: str
    total: int
    distribution: dict[str, int]
    mean_probabilities: Probabilities
    neutral_contexts: dict[str, int]
    results: list[SentimentResult]


@dataclass
class SentimentEngine:
    model_name: str
    tokenizer: object | None = None
    model: object | None = None

    @property
    def transformer_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        path = os.getenv("MODEL_PATH", "./model")
        if not os.path.isdir(path):
            self.model_name = "lexicon-fallback"
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
            self.model.eval()
            self.model_name = f"DistilBERT ({path})"
        except Exception as exc:  # do not prevent local extension use on a bad model export
            print(f"Could not load transformer: {exc}")
            self.model_name = "lexicon-fallback"
            self.tokenizer = self.model = None

    def analyse(self, comments: list[Comment]) -> list[SentimentResult]:
        if self.transformer_ready:
            return self._transformer(comments)
        return [self._fallback(comment) for comment in comments]

    def _transformer(self, comments: list[Comment]) -> list[SentimentResult]:
        import torch

        output: list[SentimentResult] = []
        for start in range(0, len(comments), 64):
            batch = comments[start : start + 64]
            encoded = self.tokenizer([item.text for item in batch], truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                probabilities = torch.softmax(self.model(**encoded).logits, dim=1).cpu().numpy()
            for comment, row in zip(batch, probabilities):
                # Training labels are fixed as 0=negative, 1=neutral, 2=positive.
                probs = Probabilities(**dict(zip(LABELS, map(float, row))))
                label = LABELS[int(np.argmax(row))]
                output.append(SentimentResult(id=comment.id, label=label, confidence=round(float(max(row)), 4), probabilities=probs, neutral_context=self._neutral_context(comment.text) if label == "neutral" else None))
        return output

    def _fallback(self, comment: Comment) -> SentimentResult:
        words = {word.strip(".,!?;:'\"()[]").lower() for word in comment.text.split()}
        positive = len(words & {"good", "great", "love", "awesome", "amazing", "excellent", "thanks", "best", "helpful", "wow"})
        negative = len(words & {"bad", "hate", "worst", "awful", "boring", "fake", "waste", "terrible", "dislike", "poor"})
        # Softmax keeps the response schema identical to transformer output.
        row = np.exp(np.array([negative, 0.35, positive], dtype=float))
        row /= row.sum()
        probs = Probabilities(**dict(zip(LABELS, map(float, row))))
        label = LABELS[int(np.argmax(row))]
        return SentimentResult(id=comment.id, label=label, confidence=round(float(max(row)), 4), probabilities=probs, neutral_context=self._neutral_context(comment.text) if label == "neutral" else None)

    @staticmethod
    def _neutral_context(text: str) -> Literal["question", "spam_or_bot_like", "informational"]:
        """Provide explainable neutral grouping without changing the model's polarity class."""
        lower, words = text.lower(), text.lower().split()
        if "?" in text or any(word in {"how", "what", "when", "where", "why", "can", "does"} for word in words):
            return "question"
        if len(words) <= 3 or any(term in lower for term in ("subscribe", "check my", "http", "www.", "telegram", "giveaway")):
            return "spam_or_bot_like"
        return "informational"


engine = SentimentEngine("starting")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(engine.load)
    yield


app = FastAPI(title="YouTube Sentiment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # A packed extension has a stable but installation-specific origin.
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": engine.model_name, "transformer_ready": engine.transformer_ready}


@app.post("/v1/sentiment/analyze", response_model=AnalyseResponse)
async def analyse(payload: AnalyseRequest) -> AnalyseResponse:
    # Avoid event-loop starvation while model inference runs.
    results = await asyncio.to_thread(engine.analyse, payload.comments)
    if len(results) != len(payload.comments):
        raise HTTPException(500, "inference returned an incomplete result set")
    counts = Counter(item.label for item in results)
    contexts = Counter(item.neutral_context for item in results if item.neutral_context)
    means = {label: round(float(np.mean([getattr(item.probabilities, label) for item in results])), 4) for label in LABELS}
    return AnalyseResponse(
        model=engine.model_name,
        total=len(results),
        distribution={label: counts[label] for label in LABELS},
        mean_probabilities=Probabilities(**means),
        neutral_contexts={context: contexts[context] for context in ("question", "spam_or_bot_like", "informational")},
        results=results,
    )
