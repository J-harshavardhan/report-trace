import React, { useState } from "react";

// Set VITE_API_BASE in Vercel project settings to your deployed backend URL,
// e.g. https://report-trace-api.vercel.app
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function ClaimRow({ verdict }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(verdict.entailment_score * 100);

  return (
    <div className="claim">
      <div className={`claim-marker ${verdict.label}`} />
      <div className="claim-body">
        <p className="claim-text">{verdict.sentence}</p>
        <div className="claim-meta" onClick={() => setOpen(!open)}>
          <span className={`badge ${verdict.label}`}>{verdict.label}</span>
          <span className="score">entailment {pct}%</span>
          <span className="toggle-hint">{open ? "hide evidence ▲" : "show evidence ▼"}</span>
        </div>
        {open && (
          <div className="evidence-panel">
            <span className="evidence-label">Closest source evidence</span>
            {verdict.best_evidence || "No matching sentence found in source report."}
            {verdict.flagged_entities.length > 0 && (
              <div className="entity-flags">
                {verdict.flagged_entities.map((e, i) => (
                  <span key={i}>{e} not found in source</span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("paste");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [maxSentences, setMaxSentences] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const canRun = mode === "paste" ? text.trim().length > 50 : !!file;

  async function handleRun() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      if (mode === "paste") {
        form.append("report_text", text);
      } else {
        form.append("file", file);
      }
      form.append("max_summary_sentences", String(maxSentences));

      const res = await fetch(`${API_BASE}/summarize`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const tally = result
    ? {
        supported: result.verdicts.filter((v) => v.label === "supported").length,
        partial: result.verdicts.filter((v) => v.label === "partial").length,
        unsupported: result.verdicts.filter((v) => v.label === "unsupported").length,
      }
    : null;

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="masthead-eyebrow">Clinical Summarization · Trace View</div>
        <h1>Report Trace</h1>
        <p>
          Paste or upload a medical report. Each summary sentence is checked
          against the source text with an entailment model and a dosage/date
          verifier, and traced back to the exact evidence it came from.
        </p>
      </header>

      <div className="input-panel">
        <div className="tab-row">
          <button
            className={`tab-btn ${mode === "paste" ? "active" : ""}`}
            onClick={() => setMode("paste")}
          >
            Paste text
          </button>
          <button
            className={`tab-btn ${mode === "pdf" ? "active" : ""}`}
            onClick={() => setMode("pdf")}
          >
            Upload PDF
          </button>
        </div>

        {mode === "paste" ? (
          <textarea
            placeholder="Paste the raw report text here (labs, discharge summary, radiology notes, etc.)"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        ) : (
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
        )}

        <div className="controls-row">
          <label className="sentence-count">
            Max summary sentences
            <input
              type="number"
              min={3}
              max={20}
              value={maxSentences}
              onChange={(e) => setMaxSentences(Number(e.target.value))}
            />
          </label>
          <button className="run-btn" disabled={!canRun || loading} onClick={handleRun}>
            {loading ? "Tracing…" : "Summarize & Verify"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <>
          <div className="confidence-strip">
            <div className="confidence-figure">
              <div className="value">{Math.round(result.overall_confidence * 100)}%</div>
              <div className="label">Traced confidence</div>
            </div>
            <div className="tally">
              <div className="tally-item">
                <span className="dot supported" />
                {tally.supported} supported
              </div>
              <div className="tally-item">
                <span className="dot partial" />
                {tally.partial} partial
              </div>
              <div className="tally-item">
                <span className="dot unsupported" />
                {tally.unsupported} unsupported
              </div>
            </div>
          </div>

          <div className="claims">
            {result.verdicts.map((v, i) => (
              <ClaimRow key={i} verdict={v} />
            ))}
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div className="empty-state">Summary and verification trace will appear here.</div>
      )}
    </div>
  );
}
