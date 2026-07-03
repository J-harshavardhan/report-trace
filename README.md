# Report Trace — AI Medical Report Summarizer with Hallucination Detection
### (Vercel-deployable version)

Summarizes medical reports with Groq (Llama 3), then verifies every summary sentence
against the source document before showing it to the user, using a two-tier
independent check rather than trusting the LLM's output wholesale.

This version is restructured to deploy fully on Vercel: the original design
loaded a ~500MB NLI model locally via `torch`/`transformers`, which doesn't
fit in a Vercel serverless function. Here the same entailment check runs
through the **Hugging Face Inference API** instead — a hosted call, no model
download, no `torch` dependency.

## Architecture

```
report (PDF or text)
        │
        ▼
 1. Summarizer (Groq API)
    → produces N discrete summary sentences (not one paragraph)
        │
        ▼
 2. Hallucination Detector — per summary sentence:
    a) Retrieval: TF-IDF cosine similarity picks the top-3 most relevant
       sentences from the source report (scikit-learn, in-process)
    b) Entailment: a hosted NLI model (DeBERTa-v3, MNLI+FEVER+ANLI
       fine-tune) scores whether that evidence entails the summary
       sentence, called via the Hugging Face Inference API
    c) Entity check: dosages, lab values, vitals, and dates are extracted
       via regex and must appear verbatim in the source — this overrides
       the entailment score, since numbers are the highest-risk hallucination
        │
        ▼
 3. Verdict per sentence: supported / partial / unsupported
    + overall confidence score for the whole report
```

**Why two tiers instead of just asking the LLM to self-check?** Asking a
model to grade its own output is circular — the same failure mode that
produced the hallucination can produce a false "looks fine." The NLI model
never saw the summarization prompt, and the entity check is pure string
matching with no model involved at all. That independence is the point.

## Project structure — two separate Vercel projects

```
report-trace/
  backend/
    main.py                    FastAPI app (Vercel entrypoint)
    app/
      summarizer.py             Groq API call, structured sentence output
      hallucination_detector.py TF-IDF retrieval + HF Inference API + entity check
      utils.py                  PDF extraction, regex sentence splitter, entity regex
      schemas.py                Pydantic request/response models
    requirements.txt
    vercel.json                 maxDuration config
    .env.example
  frontend/
    src/
      App.jsx                   Upload/paste UI + verified summary view
      index.css
    package.json
    vite.config.js
```

Backend and frontend deploy as **two independent Vercel projects** from this
one repo, each with its own "Root Directory" setting. This is the standard,
predictable way to run a Python API alongside a static frontend on Vercel —
no monorepo routing config to get right, and each half can be redeployed or
debugged on its own.

## Deploying

### 1. Backend

- Push this repo to GitHub.
- In Vercel: **New Project** → import the repo → set **Root Directory** to `backend`.
- Vercel auto-detects the FastAPI/Python framework from `requirements.txt` + `main.py`.
- Add environment variables in the project's Settings → Environment Variables:
  - `GROQ_API_KEY` — your Groq API key (get one at [console.groq.com](https://console.groq.com))
  - `HF_API_TOKEN` — a free Hugging Face access token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), "Read" scope is enough)
  - `SUMMARY_MODEL` (optional, defaults to `llama-3.3-70b-versatile`)
  - `NLI_MODEL` (optional, defaults to the DeBERTa-v3 MNLI model)
  - `FRONTEND_ORIGIN` — set once you know it, e.g. `https://report-trace.vercel.app` (leave unset / `*` while testing)
- Deploy. Note the resulting URL, e.g. `https://report-trace-api.vercel.app`.

**Heads up on the free HF Inference API:** the first call to a model that
hasn't been used recently can return a 503 while it spins up — the backend
already retries with a short wait for this. If a model is unavailable or
your token has hit a rate limit, `_entailment_score` degrades to `0.0`
rather than failing the whole request; the entity check still runs.

### 2. Frontend

- In Vercel: **New Project** → import the same repo → set **Root Directory** to `frontend`.
- Add environment variable:
  - `VITE_API_BASE` = your backend URL from step 1, e.g. `https://report-trace-api.vercel.app`
- Deploy.

### Local development

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY and HF_API_TOKEN
python main.py          # runs on http://localhost:8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev              # runs on http://localhost:5173, defaults to localhost:8000 backend
```

## Extending this for the viva

Things worth being able to speak to:

- **Why TF-IDF for retrieval and not embeddings?** It's fast, needs no
  extra model download, and is fully explainable — you can show exactly
  why a sentence was picked as evidence. Swapping in embeddings is a
  natural "future work" point, trading transparency for better semantic
  matching.
- **Why a hosted NLI model instead of running it locally?** This is the
  actual deployment constraint that shaped the architecture: Vercel
  serverless functions cap out at 500MB, and `torch` + `transformers` +
  a DeBERTa checkpoint blow past that on their own. Calling a hosted
  inference API keeps the function small and fast to cold-start, at the
  cost of a network round-trip per check and a dependency on a third
  party's uptime — a real, discussable tradeoff.
- **Why regex for entities and not a medical NER model** (e.g. scispaCy)?
  Regex is deterministic and auditable, which matters when you're the one
  defending false positives/negatives in a viva. A NER model would catch
  drug names and conditions the regex misses — another natural "future
  work" point.
- **Why regex sentence splitting instead of nltk?** nltk's tokenizer needs
  to download its punkt model on first use, which is a poor fit for a
  serverless cold start (no guaranteed writable cache, added latency). The
  regex splitter trades some accuracy on edge cases (unusual abbreviations)
  for zero runtime dependencies.
- **Failure modes worth naming:** the entailment model can be fooled by
  sentences that are topically similar but logically unsupported; the
  entity regex won't catch a hallucinated drug *name* (only malformed
  numbers/dates); TF-IDF retrieval can miss evidence if the summary
  paraphrases heavily; the hosted NLI call adds latency and an external
  dependency compared to an in-process model.

## Notes

- Only PDF and pasted text are supported for input right now.
- The confidence score is `(supported + 0.5·partial) / total_claims` — a
  simple, explainable weighting, not a learned metric.
- Vercel's Hobby plan caps function duration at 60s (already set in
  `vercel.json`); a long report with many summary sentences, each needing
  an HF API round-trip, can approach that. Keep `max_summary_sentences`
  modest (the default of 8) for a smooth demo.
