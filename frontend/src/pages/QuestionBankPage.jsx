import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchExamSets, fetchStats } from "../lib/api";
import Spinner from "../components/Spinner";
import ErrorMessage from "../components/ErrorMessage";
import GenerateModal from "../components/GenerateModal";
import { formatDate, pad } from "../lib/utils";

export default function QuestionBankPage() {
  const [sets, setSets] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showModal, setShowModal] = useState(false);
  const navigate = useNavigate();

  async function loadData() {
    setLoading(true);
    setError("");
    try {
      const [setsData, statsData] = await Promise.all([fetchExamSets(), fetchStats()]);
      setSets(setsData);
      setStats(statsData);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Failed to load question bank");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadData(); }, []);

  return (
    <>
      {showModal && (
        <GenerateModal
          onClose={() => setShowModal(false)}
          onSuccess={() => { setShowModal(false); loadData(); }}
        />
      )}

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
        {/* Page header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Question Bank</h1>
            <p className="text-slate-500 mt-1">
              {stats
                ? `${stats.total_sets} exam set${stats.total_sets !== 1 ? "s" : ""} · ${stats.total_questions?.toLocaleString()} questions`
                : "All generated CCC exam sets"}
            </p>
          </div>
          <button onClick={() => setShowModal(true)} className="btn-primary self-start sm:self-auto">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            Generate New Set
          </button>
        </div>

        {/* Stats summary */}
        {stats && (
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: "Easy", value: stats.by_difficulty?.easy || 0, color: "text-emerald-600" },
              { label: "Medium", value: stats.by_difficulty?.medium || 0, color: "text-amber-600" },
              { label: "Hard", value: stats.by_difficulty?.hard || 0, color: "text-red-600" },
            ].map(({ label, value, color }) => (
              <div key={label} className="card px-4 py-3 text-center">
                <p className={`text-2xl font-bold ${color}`}>{value.toLocaleString()}</p>
                <p className="text-xs text-slate-500 mt-0.5">{label}</p>
              </div>
            ))}
          </div>
        )}

        {/* Content */}
        {loading && (
          <div className="flex justify-center py-20"><Spinner size="lg" /></div>
        )}

        {!loading && error && (
          <ErrorMessage message={error} onRetry={loadData} />
        )}

        {!loading && !error && sets.length === 0 && (
          <div className="card py-20 flex flex-col items-center gap-5 text-center">
            <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center text-3xl">📭</div>
            <div>
              <p className="font-semibold text-slate-800 text-lg">No exam sets yet</p>
              <p className="text-slate-500 text-sm mt-1 max-w-sm">
                Click "Generate New Set" to run the AI pipeline and create your first CCC exam.
              </p>
            </div>
            <button onClick={() => setShowModal(true)} className="btn-primary">
              Generate First Set
            </button>
          </div>
        )}

        {!loading && !error && sets.length > 0 && (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {sets.map(set => (
              <ExamSetCard
                key={set.set_number}
                set={set}
                chapterCount={Object.keys(stats?.by_chapter || {}).length}
                onClick={() => navigate(`/question-bank/${set.set_number}`)}
              />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function ExamSetCard({ set, onClick }) {
  return (
    <button
      onClick={onClick}
      className="card p-6 text-left group hover:shadow-lg hover:border-brand-300 transition-all duration-200 w-full"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="w-12 h-12 rounded-xl bg-brand-600 flex items-center justify-center shadow-md
                        group-hover:bg-brand-700 transition-colors">
          <span className="text-white font-bold text-lg">{pad(set.set_number)}</span>
        </div>
        <svg className="w-5 h-5 text-slate-400 group-hover:text-brand-600 group-hover:translate-x-0.5
                        transition-all duration-150 mt-1" fill="none" viewBox="0 0 24 24"
          stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </div>

      <h3 className="font-bold text-slate-900 text-lg group-hover:text-brand-700 transition-colors">
        Exam Set {pad(set.set_number)}
      </h3>
      <p className="text-slate-500 text-sm mt-1">
        {set.total_questions} questions · CCC Course
      </p>

      <div className="mt-4 pt-4 border-t border-slate-100 flex items-center gap-2">
        <span className="badge bg-brand-50 text-brand-700">
          {set.source === "database" ? "Database" : "JSON"}
        </span>
        <span className="text-xs text-slate-400">{formatDate(set.generated_at)}</span>
      </div>
    </button>
  );
}
