import os
import json
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = os.environ.get("SUMMARY_MODEL", "claude-sonnet-4-6")

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

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
        return parsed["summary_sentences"]
    except (json.JSONDecodeError, KeyError):
        # Fallback: treat the raw output as newline-separated sentences
        return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
