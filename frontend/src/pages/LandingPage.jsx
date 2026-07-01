import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchStats } from "../lib/api";
import GenerateModal from "../components/GenerateModal";
import StatsRow from "../components/StatsRow";
import Spinner from "../components/Spinner";

export default function LandingPage() {
  const [showModal, setShowModal] = useState(false);
  const [stats, setStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoadingStats(false));
  }, []);

  function handleSuccess() {
    setShowModal(false);
    navigate("/question-bank");
  }

  return (
    <>
      {showModal && (
        <GenerateModal onClose={() => setShowModal(false)} onSuccess={handleSuccess} />
      )}

      {/* Hero */}
      <main>
        <section className="relative overflow-hidden bg-gradient-to-br from-brand-900 via-brand-800 to-brand-700 text-white">
          {/* Background pattern */}
          <div className="absolute inset-0 opacity-10">
            <svg width="100%" height="100%">
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="white" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          </div>

          <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32 text-center">
            {/* Pill badge */}
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-1.5 rounded-full text-sm font-medium mb-8 border border-white/20">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              AI-Powered · NIELIT Official Syllabus
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight mb-6">
              CCC Exam Question Bank
            </h1>

            <p className="text-lg sm:text-xl text-brand-200 max-w-2xl mx-auto mb-10 leading-relaxed">
              Automatically generate question MCQ exam sets aligned to the NIELIT CCC
              syllabus using  Bloom's taxonomy — complete with difficulty
              grading, RAG-verified answers, and multi-layer validation.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => setShowModal(true)}
                className="inline-flex items-center gap-2 px-8 py-4 bg-white text-brand-700 font-bold
                           rounded-xl shadow-lg hover:bg-brand-50 active:scale-95 transition-all text-base"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
                Generate Exam Set
              </button>
              <button
                onClick={() => navigate("/question-bank")}
                className="inline-flex items-center gap-2 px-8 py-4 bg-white/10 backdrop-blur text-white font-semibold
                           rounded-xl border border-white/30 hover:bg-white/20 active:scale-95 transition-all text-base"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 7h18M3 12h18M3 17h18" />
                </svg>
                View Question Bank
              </button>
            </div>
          </div>
        </section>

        {/* Stats strip */}
        <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 -mt-10 relative z-10">
          {loadingStats ? (
            <div className="flex justify-center py-12"><Spinner size="lg" /></div>
          ) : (
            <StatsRow stats={stats} />
          )}
        </section>

        {/* Features section */}
        <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 text-center mb-12">
            How it works
          </h2>
          <div className="grid sm:grid-cols-3 gap-8">
            {features.map(({ icon, title, desc }) => (
              <div key={title} className="flex flex-col items-center text-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-brand-50 text-brand-600 flex items-center justify-center text-2xl shadow-sm">
                  {icon}
                </div>
                <h3 className="font-semibold text-slate-900 text-lg">{title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Chapter coverage */}
        {stats?.by_chapter && Object.keys(stats.by_chapter).length > 0 && (
          <section className="bg-white border-t border-slate-200">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
              <h2 className="text-2xl font-bold text-slate-900 mb-8 text-center">Chapter Coverage</h2>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(stats.by_chapter).map(([chapter, count]) => (
                  <div key={chapter} className="flex items-center justify-between px-4 py-3 card">
                    <span className="text-sm text-slate-700 truncate mr-2">{chapter}</span>
                    <span className="badge bg-brand-50 text-brand-700 flex-shrink-0">{count}Q</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white">
        <div className="max-w-5xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          CCC Exam Platform · NIELIT National Digital Literacy Mission ·{" "}
          <span className="text-brand-600">AI-generated, RAG-verified questions</span>
        </div>
      </footer>
    </>
  );
}

const features = [
  {
    icon: "🤖",
    title: "AI Generation",
    desc: "AWS Bedrock Claude Sonnet generates MCQs aligned to the official CCC syllabus and Bloom's taxonomy levels.",
  },
  {
    icon: "🔍",
    title: "RAG Verification",
    desc: "Pinecone vector search retrieves relevant textbook passages to ground every question in authoritative content.",
  },
  {
    icon: "✅",
    title: "Multi-layer Validation",
    desc: "Correctness checks, difficulty classification, duplicate detection, and formatting rules ensure question quality.",
  },
];
