import os
import json
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("SUMMARY_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a clinical summarization assistant. You will be given the
raw text of a medical report (lab results, discharge summary, radiology
report, etc.). Produce a concise, accurate summary for a treating physician.

Rules:
- Every sentence in your summary must be directly traceable to a fact stated
  in the source text. Do not infer, extrapolate, or add clinical judgment
  that is not explicitly present in the report.
- Preserve exact numbers, units, drug names, and dates verbatim -- never
  round, convert, or approximate them.
- If the report is ambiguous or a value is missing, state that explicitly
  rather than guessing.
- Return ONLY valid JSON, no preamble, no markdown fences, in this exact
  shape: {"summary_sentences": ["sentence 1", "sentence 2", ...]}
"""


def summarize_report(report_text: str, max_sentences: int = 8) -> list[str]:
    user_prompt = (
        f"Summarize the following medical report in at most {max_sentences} "
        f"sentences, following the rules above.\n\n"
        f"--- REPORT START ---\n{report_text}\n--- REPORT END ---"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
        return parsed["summary_sentences"]
    except (json.JSONDecodeError, KeyError):
        return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
