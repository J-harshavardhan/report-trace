import re
import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """Pull raw text out of a medical report PDF, page by page."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Common medical abbreviations that end in a period but should NOT be
# treated as sentence boundaries (e.g. "Dr. Smith", "150 mg. daily").
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "vs", "etc", "e.g", "i.e",
    "mg", "ml", "mmol", "approx", "no", "fig", "St",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str):
    """Lightweight regex sentence splitter -- avoids a runtime download of
    nltk's punkt model, which is a bad fit for a serverless cold start."""
    # Split by newlines first to preserve paragraph boundaries, then by sentence regex
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_sentences = []
    for line in lines:
        raw_sentences.extend(_SENTENCE_SPLIT_RE.split(line))

    sentences = []
    buffer = ""
    for chunk in raw_sentences:
        buffer = f"{buffer} {chunk}".strip() if buffer else chunk
        last_word = re.sub(r"[.!?]$", "", buffer.split(" ")[-1]).lower()
        if last_word in _ABBREVIATIONS:
            continue  # keep accumulating, this wasn't a real sentence end
        sentences.append(buffer.strip())
        buffer = ""
    if buffer:
        sentences.append(buffer.strip())

    return [s for s in sentences if len(s) > 8]


# --- Medical entity patterns (dosages, labs, vitals, dates) ---
# These are the highest-stakes hallucination targets: a wrong number here
# is far more dangerous than a wrong adjective.
ENTITY_PATTERNS = [
    r"\b\d+\.?\d*\s?(mg|mcg|g|ml|mmol/L|mg/dL|g/dL|IU|bpm|mmHg|%|kg|cm)\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",          # dates
    r"\b\d+\.?\d*\s?°?[CF]\b",                       # temperature
    r"\b\d{2,3}/\d{2,3}\b",                           # blood pressure e.g. 120/80
]


def extract_entities(sentence: str):
    matches = []
    for pattern in ENTITY_PATTERNS:
        matches.extend(re.finditer(pattern, sentence, flags=re.IGNORECASE))
    return [m.group(0) for m in matches]
