import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import SummarizeResponse
from app.summarizer import summarize_report
from app.hallucination_detector import verify_summary, compute_confidence
from app.utils import extract_text_from_pdf, clean_text

app = FastAPI(title="AI Medical Report Summarizer with Hallucination Detection")

# Allow the deployed frontend origin (set FRONTEND_ORIGIN in Vercel project
# settings once you have the frontend URL; * is fine while developing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    hf_token = os.environ.get("HF_API_TOKEN")
    groq_key = os.environ.get("GROQ_API_KEY")
    return {
        "status": "ok",
        "hf_token_present": bool(hf_token),
        "hf_token_length": len(hf_token) if hf_token else 0,
        "groq_key_present": bool(groq_key),
    }


@app.get("/debug_verify")
def debug_verify():
    from app.hallucination_detector import _retrieve_evidence, _entailment_score, HF_API_TOKEN
    from app.utils import split_sentences

    source_text = (
        "Patient: Test Patient. Age 45. "
        "Blood pressure 130/85 mmHg recorded on 15/06/2026. "
        "Prescribed Metformin 500mg twice daily. "
        "Lab results show HbA1c of 7.2%. "
        "Patient reports mild fatigue, no chest pain."
    )
    summary_sentence = "The patient's blood pressure was recorded as 130/85 mmHg on 15/06/2026."

    source_sentences = split_sentences(source_text)
    evidence_candidates = _retrieve_evidence(summary_sentence, source_sentences)

    hf_call_result = None
    hf_call_error = None
    if evidence_candidates:
        try:
            hf_call_result = _entailment_score(evidence_candidates[0], summary_sentence)
        except Exception as e:
            hf_call_error = f"{type(e).__name__}: {e}"

    return {
        "hf_token_present": bool(HF_API_TOKEN),
        "hf_token_length": len(HF_API_TOKEN) if HF_API_TOKEN else 0,
        "source_sentences_count": len(source_sentences),
        "source_sentences": source_sentences,
        "evidence_candidates_count": len(evidence_candidates),
        "evidence_candidates": evidence_candidates,
        "hf_call_result": hf_call_result,
        "hf_call_error": hf_call_error,
    }


@app.get("/debug_hf_raw")
def debug_hf_raw():
    import requests as req
    hf_token = os.environ.get("HF_API_TOKEN")
    nli_model = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    url = f"https://api-inference.huggingface.co/models/{nli_model}"

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": {"text": "The sky is blue.", "text_pair": "The sky has a blue color."}}

    try:
        resp = req.post(url, headers=headers, json=payload, timeout=15)
        return {
            "status_code": resp.status_code,
            "response_body": resp.text[:2000],
            "url_called": url,
        }
    except Exception as e:
        return {"exception": f"{type(e).__name__}: {e}"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    file: UploadFile = File(None),
    report_text: str = Form(None),
    max_summary_sentences: int = Form(8),
):
    if not file and not report_text:
        raise HTTPException(400, "Provide either a PDF file or report_text.")

    if file:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix != ".pdf":
            raise HTTPException(400, "Only PDF uploads are supported right now.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        raw_text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)
    else:
        raw_text = report_text

    source_text = clean_text(raw_text)
    if len(source_text) < 50:
        raise HTTPException(400, "Report text is too short to summarize meaningfully.")

    summary_sentences = summarize_report(source_text, max_summary_sentences)
    verdicts = verify_summary(summary_sentences, source_text)
    confidence = compute_confidence(verdicts)
    hallucination_count = sum(1 for v in verdicts if v.label == "unsupported")

    return SummarizeResponse(
        summary_sentences=summary_sentences,
        verdicts=verdicts,
        overall_confidence=confidence,
        hallucination_count=hallucination_count,
        total_claims=len(verdicts),
    )


if __name__ == "__main__":
    # Local dev only -- Vercel invokes `app` directly, it doesn't run this.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
