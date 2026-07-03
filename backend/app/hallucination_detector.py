"""
Hallucination detection for the medical report summarizer.

Two-tier check applied to every summary sentence:

  Tier 1 - Entailment check: retrieve the most relevant sentences from the
  source report via TF-IDF cosine similarity, then ask a small, fast LLM
  (via Groq) a narrow, structured question -- does this specific evidence
  sentence support this specific summary sentence? This call is deliberately
  isolated from the original summarization prompt/context, so it can't just
  rubber-stamp its own prior output.

  Tier 2 - Entity verification: extract high-stakes entities (dosages, lab
  values, vitals, dates) from the summary sentence and confirm each one
  appears verbatim (or near-verbatim) in the source. Numbers are the most
  dangerous class of hallucination in a medical setting, so they get a
  dedicated, stricter, model-free check rather than relying on the LLM call
  alone.

The two signals are combined into a single label: supported / partial /
unsupported.
"""

import os
import json
from typing import List

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import extract_entities, split_sentences
from .schemas import ClaimVerdict

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "llama-3.1-8b-instant")

ENTAILMENT_SUPPORTED = 0.60
ENTAILMENT_PARTIAL = 0.35
TOP_K_EVIDENCE = 3
REQUEST_TIMEOUT = 20

VERIFIER_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given
a SOURCE sentence and a CLAIM sentence. Decide whether the SOURCE sentence
entails (fully supports) the CLAIM.

Return ONLY valid JSON, no preamble, no markdown fences, in this exact shape:
{"entailment_score": <float between 0.0 and 1.0>}

Scoring guide:
- 1.0: the SOURCE directly and fully supports the CLAIM
- 0.5: the SOURCE partially supports the CLAIM, or supports it with a
  different emphasis/wording that changes meaning slightly
- 0.0: the SOURCE does not support the CLAIM, or the CLAIM contains
  information not present in the SOURCE
"""


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
    """Calls a small Groq model to independently judge entailment. Degrades
    to 0.0 (relies entirely on the entity check) on any failure, rather than
    failing the whole request."""
    if not GROQ_API_KEY:
        return 0.0

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VERIFIER_MODEL,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE: {premise}\n\nCLAIM: {hypothesis}"},
        ],
        "temperature": 0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        return float(parsed.get("entailment_score", 0.0))
    except Exception:
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
