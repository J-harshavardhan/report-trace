import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import SummarizeResponse
from app.summarizer import summarize_report
from app.hallucination_detector import verify_summary, compute_confidence
from app.utils import extract_text_from_pdf, clean_text

app = FastAPI(title="AI Medical Report Summarizer with Hallucination Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
