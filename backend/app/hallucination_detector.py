"""
Hallucination detection for the medical report summarizer.

Two-tier check applied to every summary sentence:

  Tier 1 - Entailment (NLI): retrieve the most relevant sentences from the
  source report via TF-IDF cosine similarity, then call a hosted NLI model
  through the Hugging Face Inference API to check whether that source
  evidence entails the summary sentence. Using a hosted model (instead of
  loading transformers/torch in-process) is what keeps this backend small
  enough to run as a Vercel serverless function.

  Tier 2 - Entity verification: extract high-stakes entities (dosages, lab
  values, vitals, dates) from the summary sentence and confirm each one
  appears verbatim (or near-verbatim) in the source. Numbers are the most
  dangerous class of hallucination in a medical setting, so they get a
  dedicated, stricter check rather than relying on the NLI model alone.

The two signals are combined into a single label: supported / partial /
unsupported.
"""

import os
import time
from typing import List

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import extract_entities, split_sentences
from .schemas import ClaimVerdict

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
NLI_MODEL = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{NLI_MODEL}"

ENTAILMENT_SUPPORTED = 0.60
ENTAILMENT_PARTIAL = 0.35
TOP_K_EVIDENCE = 3
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15


def _retrieve_evidence(summary_sentence: str, source_sentences: List[str], k: int = TOP_K_EVIDENCE):
    if not source_sentences:
        return []
    vectorizer = TfidfVectorizer().fit(source_sentences + [summary_sentence])
    source_vecs = vectorizer.transform(source_sentences)
    query_vec = vectorizer.transform([summary_sentence])
    sims = cosine_similarity(query_vec, source_vecs)[0]
    ranked = sorted(zip(source_sentences, sims), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:k]]


def _entailment_score(premise: str, hypothesis: str) -> float:
    """Calls the Hugging Face Inference API for the NLI model. Degrades to
    0.0 (relies entirely on the entity check) if no token is configured or
    the API is unreachable, rather than failing the whole request."""
    if not HF_API_TOKEN:
        return 0.0

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": {"text": premise, "text_pair": hypothesis}}

    for _ in range(MAX_RETRIES):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            return 0.0

        if resp.status_code == 200:
            data = resp.json()
            scores_list = data[0] if (isinstance(data, list) and data and isinstance(data[0], list)) else data
            if not isinstance(scores_list, list):
                return 0.0
            scores = {item.get("label", "").lower(): item.get("score", 0.0) for item in scores_list}
            return scores.get("entailment", 0.0)

        if resp.status_code == 503:
            # Free-tier model is cold-starting on HF's side; wait and retry.
            wait_seconds = 5
            try:
                wait_seconds = min(resp.json().get("estimated_time", 5), 10)
            except ValueError:
                pass
            time.sleep(wait_seconds)
            continue

        return 0.0  # any other error: don't block the whole report on it

    return 0.0


def _check_entities(sentence: str, source_text: str) -> List[str]:
    """Return entities mentioned in the summary sentence that do NOT appear
    (verbatim, whitespace-normalized) anywhere in the source report."""
    entities = extract_entities(sentence)
    normalized_source = " ".join(source_text.split()).lower()
    missing = []
    for ent in entities:
        norm_ent = " ".join(ent.split()).lower()
        if norm_ent not in normalized_source:
            missing.append(ent)
    return missing


def verify_summary(summary_sentences: List[str], source_text: str) -> List[ClaimVerdict]:
    source_sentences = split_sentences(source_text)
    verdicts = []

    for sentence in summary_sentences:
        evidence_candidates = _retrieve_evidence(sentence, source_sentences)
        best_evidence, best_score = "", 0.0
        for evidence in evidence_candidates:
            score = _entailment_score(evidence, sentence)
            if score > best_score:
                best_score, best_evidence = score, evidence

        missing_entities = _check_entities(sentence, source_text)

        # A missing high-stakes entity (wrong/invented number, dose, date)
        # overrides the entailment score -- it's flagged regardless.
        if missing_entities:
            label = "unsupported"
        elif best_score >= ENTAILMENT_SUPPORTED:
            label = "supported"
        elif best_score >= ENTAILMENT_PARTIAL:
            label = "partial"
        else:
            label = "unsupported"

        verdicts.append(
            ClaimVerdict(
                sentence=sentence,
                label=label,
                entailment_score=round(float(best_score), 3),
                best_evidence=best_evidence,
                flagged_entities=missing_entities,
            )
        )

    return verdicts


def compute_confidence(verdicts: List[ClaimVerdict]) -> float:
    if not verdicts:
        return 0.0
    weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    total = sum(weights[v.label] for v in verdicts)
    return round(total / len(verdicts), 3)
