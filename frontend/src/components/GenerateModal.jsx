import { useState, useEffect, useRef } from "react";
import { triggerGeneration, fetchGenerationStatus } from "../lib/api";
import { statusInfo } from "../lib/utils";
import Spinner from "./Spinner";

export default function GenerateModal({ onClose, onSuccess }) {
  const [selectedSets, setSelectedSets] = useState([1]);
  const [noDb, setNoDb] = useState(false);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const pollRef = useRef(null);
  const logEndRef = useRef(null);

  const allSets = Array.from({ length: 10 }, (_, i) => i + 1);

  function toggleSet(n) {
    setSelectedSets(prev =>
      prev.includes(n) ? prev.filter(s => s !== n) : [...prev, n].sort((a, b) => a - b)
    );
  }

  async function handleSubmit() {
    if (selectedSets.length === 0) {
      setError("Select at least one exam set to generate.");
      return;
    }
    setError("");
    try {
      const result = await triggerGeneration({ sets: selectedSets, no_db: noDb });
      setJob({ ...result, log_lines: [] });
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Failed to start generation");
    }
  }

  // Poll every 1.5 s while job is active
  useEffect(() => {
    if (!job?.job_id) return;
    if (job.status === "completed" || job.status === "failed") return;

    pollRef.current = setInterval(async () => {
      try {
        const updated = await fetchGenerationStatus(job.job_id);
        setJob(updated);
        if (updated.status === "completed") {
          clearInterval(pollRef.current);
          // Small delay so user sees the "Completed" state before modal closes
          setTimeout(() => onSuccess?.(), 1200);
        } else if (updated.status === "failed") {
          clearInterval(pollRef.current);
        }
      } catch {
        // ignore transient network errors
      }
    }, 1500);

    return () => clearInterval(pollRef.current);
  }, [job?.job_id, job?.status]);

  // Auto-scroll log to bottom on new lines
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [job?.log_lines?.length]);

  const si = job ? statusInfo(job.status) : null;
  const isActive = job?.status === "queued" || job?.status === "running";
  const logLines = job?.log_lines || [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget && !isActive) onClose(); }}
    >
      <div className="card w-full max-w-lg p-6 space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-slate-900">Generate Exam Sets</h2>
            <p className="text-sm text-slate-500 mt-0.5">
              Triggers the AI pipeline to produce 100 questions per set.
            </p>
          </div>
          {!isActive && (
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {/* ── Set selector (pre-submit) ── */}
        {!job && (
          <>
            <div>
              <p className="text-sm font-medium text-slate-700 mb-2">Select sets to generate</p>
              <div className="grid grid-cols-5 gap-2">
                {allSets.map(n => (
                  <button
                    key={n}
                    onClick={() => toggleSet(n)}
                    className={`py-2 rounded-lg text-sm font-semibold border transition-colors ${
                      selectedSets.includes(n)
                        ? "bg-brand-600 text-white border-brand-600"
                        : "bg-white text-slate-700 border-slate-300 hover:border-brand-400"
                    }`}
                  >
                    Set {n}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex items-center gap-3 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={noDb}
                onChange={e => setNoDb(e.target.checked)}
                className="w-4 h-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              <span className="text-sm text-slate-700">Skip PostgreSQL — JSON output only</span>
            </label>

            {error && (
              <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{error}</p>
            )}

            <div className="flex gap-3 pt-1">
              <button onClick={onClose} className="btn-secondary flex-1">Cancel</button>
              <button onClick={handleSubmit} className="btn-primary flex-1">
                Start Generation
              </button>
            </div>
          </>
        )}

        {/* ── Job status + live log ── */}
        {job && (
          <div className="space-y-4">
            {/* Status badge row */}
            <div className="flex items-center gap-3">
              {isActive && <Spinner size="sm" />}
              <span className={`badge ${si.color} text-sm px-3 py-1 font-semibold`}>
                {si.label}
              </span>
              {job.sets && (
                <span className="text-xs text-slate-500">
                  Sets: {job.sets.join(", ")}
                </span>
              )}
            </div>

            {/* Live log terminal */}
            <div className="rounded-lg border border-slate-700 bg-slate-900 overflow-hidden">
              {/* Terminal title bar */}
              <div className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 border-b border-slate-700">
                <span className="w-3 h-3 rounded-full bg-red-500 opacity-75" />
                <span className="w-3 h-3 rounded-full bg-amber-400 opacity-75" />
                <span className="w-3 h-3 rounded-full bg-emerald-500 opacity-75" />
                <span className="ml-2 text-xs text-slate-400 font-mono">pipeline output</span>
                {isActive && (
                  <span className="ml-auto flex items-center gap-1 text-xs text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    live
                  </span>
                )}
              </div>

              {/* Log lines */}
              <div className="h-56 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
                {logLines.length === 0 ? (
                  <p className="text-slate-500 italic">
                    {job.status === "queued" ? "Waiting for pipeline to start…" : "Starting pipeline…"}
                  </p>
                ) : (
                  logLines.map((line, i) => (
                    <div key={i} className={`${_lineColor(line)} whitespace-pre-wrap break-all`}>
                      {line}
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </div>

            {/* Completion / failure messages */}
            {job.status === "completed" && (
              <div className="flex items-center gap-2 px-4 py-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                <svg className="w-5 h-5 text-emerald-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <p className="text-sm font-medium text-emerald-700">
                  Generation complete — {job.sets?.length} set{job.sets?.length !== 1 ? "s" : ""} ready.
                </p>
              </div>
            )}

            {job.status === "failed" && (
              <div className="flex items-center gap-2 px-4 py-3 bg-red-50 border border-red-200 rounded-lg">
                <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-red-700">Pipeline failed</p>
                  <p className="text-xs text-red-500 mt-0.5">{job.message}</p>
                </div>
              </div>
            )}

            {/* Close button — only when done */}
            {(job.status === "completed" || job.status === "failed") && (
              <button onClick={onClose} className="btn-primary w-full">
                {job.status === "completed" ? "Done — View Results" : "Close"}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Colour-code log lines by content */
function _lineColor(line) {
  const l = line.toLowerCase();
  if (l.includes("✓") || l.includes("complete") || l.includes("accepted") || l.includes("success"))
    return "text-emerald-400";
  if (l.includes("✗") || l.includes("fail") || l.includes("error") || l.includes("exception"))
    return "text-red-400";
  if (l.includes("↻") || l.includes("top-up") || l.includes("retry") || l.includes("warn"))
    return "text-amber-400";
  if (l.includes("►") || l.includes("chapter") || l.includes("set #") || l.includes("exam set"))
    return "text-sky-300";
  if (l.startsWith("=") || l.includes("generation"))
    return "text-violet-300";
  return "text-slate-300";
}
