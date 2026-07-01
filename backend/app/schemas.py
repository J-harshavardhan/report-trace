from pydantic import BaseModel
from typing import List, Optional


class SummarizeRequest(BaseModel):
    report_text: str
    max_summary_sentences: Optional[int] = 8


class ClaimVerdict(BaseModel):
    sentence: str
    label: str          # "supported" | "partial" | "unsupported"
    entailment_score: float
    best_evidence: str
    flagged_entities: List[str] = []


class SummarizeResponse(BaseModel):
    summary_sentences: List[str]
    verdicts: List[ClaimVerdict]
    overall_confidence: float
    hallucination_count: int
    total_claims: int
