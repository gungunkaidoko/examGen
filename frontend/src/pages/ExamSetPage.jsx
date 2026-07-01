import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { fetchExamSet, fetchFilterOptions, fetchMetrics, getPdfUrl } from "../lib/api";
import FilterBar from "../components/FilterBar";
import QuestionCard from "../components/QuestionCard";
import MetricsPanel from "../components/MetricsPanel";
import Spinner from "../components/Spinner";
import ErrorMessage from "../components/ErrorMessage";
import { pad } from "../lib/utils";

const EMPTY_FILTERS = { search: "", chapter: "", difficulty: "", bloom_level: "" };

export default function ExamSetPage() {
  const { setNumber } = useParams();
  const navigate = useNavigate();
  const setNum = parseInt(setNumber, 10);

  // Tab: "questions" | "metrics"
  const [activeTab, setActiveTab] = useState("questions");

  // Questions state
  const [data, setData] = useState(null);
  const [filterOptions, setFilterOptions] = useState(null);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Metrics state
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");

  // PDF state
  const [pdfLoading, setPdfLoading] = useState(false);

  // Debounced filters
  const [debouncedFilters, setDebouncedFilters] = useState(EMPTY_FILTERS);
  useEffect(() => {
    const t = setTimeout(() => setDebouncedFilters(filters), 300);
    return () => clearTimeout(t);
  }, [filters]);

  async function loadSet(activeFilters = debouncedFilters) {
    setLoading(true);
    setError("");
    try {
      const result = await fetchExamSet(setNum, activeFilters);
      setData(result);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Failed to load exam set");
    } finally {
      setLoading(false);
    }
  }

  async function loadMetrics() {
    if (metrics) return; // already loaded
    setMetricsLoading(true);
    setMetricsError("");
    try {
      const result = await fetchMetrics(setNum);
      setMetrics(result);
    } catch (e) {
      setMetricsError(e?.response?.data?.detail || e.message || "Failed to compute metrics");
    } finally {
      setMetricsLoading(false);
    }
  }

  // Load filter options once
  useEffect(() => {
    fetchFilterOptions().then(setFilterOptions).catch(() => {});
  }, []);

  // Reload questions when debounced filters change
  useEffect(() => {
    loadSet(debouncedFilters);
  }, [setNum, debouncedFilters]);

  // Load metrics when tab switches to metrics
  useEffect(() => {
    if (activeTab === "metrics") loadMetrics();
  }, [activeTab]);

  function handleFilterChange(key, value) {
    setFilters(prev => ({ ...prev, [key]: value }));
  }

  function handleClear() {
    setFilters(EMPTY_FILTERS);
  }

  function handleDownloadPdf() {
    setPdfLoading(true);
    const url = getPdfUrl(setNum);
    const a = document.createElement("a");
    a.href = url;
    a.download = `exam_set_${pad(setNum)}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => setPdfLoading(false), 2000);
  }

  const activeFilterCount = Object.values(filters).filter(Boolean).length;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-6">

      {/* ── Page header ────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/question-bank")}
            className="p-2 rounded-lg border border-slate-200 hover:bg-slate-100 transition-colors text-slate-600"
            aria-label="Back to Question Bank"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">
              Exam Set {pad(setNum)}
            </h1>
            {data && (
              <p className="text-sm text-slate-500 mt-0.5">
                {data.total_questions} question{data.total_questions !== 1 ? "s" : ""}
                {activeFilterCount > 0 && " · filtered"}
              </p>
            )}
          </div>
        </div>

        <button
          onClick={handleDownloadPdf}
          disabled={pdfLoading}
          className="btn-secondary self-start sm:self-auto"
        >
          {pdfLoading ? (
            <Spinner size="sm" />
          ) : (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 10v6m0 0l-3-3m3 3l3-3M3 17a2 2 0 002 2h14a2 2 0 002-2v-1" />
            </svg>
          )}
          Download PDF
        </button>
      </div>

      {/* ── Tab switcher ───────────────────────────────────────────────────── */}
      <div className="flex gap-1 p-1 bg-slate-100 rounded-xl w-fit">
        {[
          { id: "questions", label: "Questions", icon: "📝" },
          { id: "metrics",   label: "Quality Metrics", icon: "📊" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
              activeTab === tab.id
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* QUESTIONS TAB                                                       */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === "questions" && (
        <>
          {/* Filter bar */}
          <div className="card p-4">
            <FilterBar
              filters={filters}
              options={filterOptions}
              onChange={handleFilterChange}
              onClear={handleClear}
            />
          </div>

          {/* Active filter pills */}
          {activeFilterCount > 0 && (
            <div className="flex flex-wrap gap-2">
              {filters.chapter && (
                <FilterPill label={`Chapter: ${filters.chapter}`} onRemove={() => handleFilterChange("chapter", "")} />
              )}
              {filters.difficulty && (
                <FilterPill label={`Difficulty: ${filters.difficulty}`} onRemove={() => handleFilterChange("difficulty", "")} />
              )}
              {filters.bloom_level && (
                <FilterPill label={`Bloom: ${filters.bloom_level}`} onRemove={() => handleFilterChange("bloom_level", "")} />
              )}
              {filters.search && (
                <FilterPill label={`Search: "${filters.search}"`} onRemove={() => handleFilterChange("search", "")} />
              )}
            </div>
          )}

          {loading && <div className="flex justify-center py-20"><Spinner size="lg" /></div>}

          {!loading && error && <ErrorMessage message={error} onRetry={() => loadSet()} />}

          {!loading && !error && data?.questions?.length === 0 && (
            <div className="card py-20 flex flex-col items-center gap-4 text-center">
              <div className="text-4xl">🔍</div>
              <div>
                <p className="font-semibold text-slate-800">No questions match your filters</p>
                <p className="text-slate-500 text-sm mt-1">Try adjusting or clearing the filters.</p>
              </div>
              <button onClick={handleClear} className="btn-secondary text-sm">Clear all filters</button>
            </div>
          )}

          {!loading && !error && data?.questions?.length > 0 && (
            <div className="space-y-4">
              {data.questions.map((q, i) => (
                <QuestionCard key={q.question_uuid || i} question={q} index={i + 1} />
              ))}
            </div>
          )}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* METRICS TAB                                                         */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activeTab === "metrics" && (
        <>
          {metricsLoading && (
            <div className="flex flex-col items-center justify-center py-20 gap-3 text-slate-500">
              <Spinner size="lg" />
              <p className="text-sm">Computing quality metrics…</p>
            </div>
          )}

          {!metricsLoading && metricsError && (
            <ErrorMessage message={metricsError} onRetry={loadMetrics} />
          )}

          {!metricsLoading && !metricsError && metrics && (
            <MetricsPanel metrics={metrics} />
          )}
        </>
      )}
    </div>
  );
}

function FilterPill({ label, onRemove }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-brand-50 text-brand-700
                     border border-brand-200 rounded-full text-xs font-medium">
      {label}
      <button onClick={onRemove} className="hover:text-brand-900 transition-colors">
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </span>
  );
}
