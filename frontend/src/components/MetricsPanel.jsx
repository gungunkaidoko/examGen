/**
 * MetricsPanel — quality dashboard for one exam set.
 * Receives pre-fetched `metrics` object as a prop from ExamSetPage.
 */

// ── colour thresholds ──────────────────────────────────────────────────────
const THRESHOLDS = {
  question_accuracy:   { warn: 98,  crit: 95  },
  answer_accuracy:     { warn: 99,  crit: 97  },
  rejection_rate:      { warn: 5,   crit: 10, inverted: true },
  semantic_uniqueness: { warn: 95,  crit: 90  },
  concept_overlap:     { warn: 5,   crit: 10, inverted: true },
  difficulty_balance:  { warn: 90,  crit: 80  },
  bloom_accuracy:      { warn: 90,  crit: 80  },
};

function statusColor(key, value) {
  const t = THRESHOLDS[key];
  if (!t) return "green";
  if (t.inverted) {
    if (value >= t.crit) return "red";
    if (value >= t.warn) return "amber";
    return "green";
  }
  if (value < t.crit) return "red";
  if (value < t.warn) return "amber";
  return "green";
}

const CLS = {
  green: { bar: "bg-emerald-500", badge: "bg-emerald-100 text-emerald-700", dot: "text-emerald-500", label: "Excellent" },
  amber: { bar: "bg-amber-400",   badge: "bg-amber-100 text-amber-700",     dot: "text-amber-500",   label: "Review"    },
  red:   { bar: "bg-red-500",     badge: "bg-red-100 text-red-700",         dot: "text-red-500",     label: "Poor"      },
};

// ── sub-components ─────────────────────────────────────────────────────────
function MetricRow({ label, value, unit = "%", metricKey, description, inverted = false }) {
  const color  = statusColor(metricKey, value);
  const cls    = CLS[color];
  const barPct = inverted ? Math.max(0, 100 - value) : Math.min(100, value);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <svg className={`w-3 h-3 flex-shrink-0 ${cls.dot}`} viewBox="0 0 8 8" fill="currentColor">
            <circle cx="4" cy="4" r="4" />
          </svg>
          <span className="text-sm font-medium text-slate-800 truncate">{label}</span>
          {description && (
            <span className="hidden sm:inline text-xs text-slate-400 truncate">— {description}</span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-base font-bold text-slate-900">{value}{unit}</span>
          <span className={`badge text-xs ${cls.badge}`}>{cls.label}</span>
        </div>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${cls.bar}`}
          style={{ width: `${barPct}%` }}
        />
      </div>
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="text-center px-3 py-2 bg-slate-50 rounded-lg border border-slate-100">
      <p className="text-lg font-bold text-slate-900">{value}</p>
      <p className="text-xs text-slate-500 mt-0.5">{label}</p>
    </div>
  );
}

function DistCard({ title, data, colorMap, total }) {
  return (
    <div className="card p-4 space-y-3">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{title}</p>
      <div className="space-y-2">
        {Object.entries(data || {})
          .sort((a, b) => b[1] - a[1])
          .map(([key, count]) => {
            const pct      = total > 0 ? Math.round(count / total * 100) : 0;
            const barColor = colorMap[key] || "bg-slate-400";
            return (
              <div key={key} className="space-y-0.5">
                <div className="flex justify-between text-xs text-slate-600">
                  <span className="capitalize">{key}</span>
                  <span className="font-medium">{count} ({pct}%)</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="space-y-6">

      {/* ── Summary strip ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MiniStat label="Total Questions"  value={metrics.total_questions} />
        <MiniStat label="Avg Gen Time"     value={`${metrics.avg_generation_time}s`} />
        <MiniStat label="Avg Retries"      value={metrics.avg_retry_count} />
        <MiniStat label="Rejection Rate"   value={`${metrics.rejection_rate}%`} />
      </div>

      {/* ── Quality scores ─────────────────────────────────────────────── */}
      <div className="card p-5 space-y-4">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
          Quality Scores
        </p>
        <MetricRow
          metricKey="question_accuracy"
          label="Question Accuracy"
          description="passed all validation layers"
          value={metrics.question_accuracy}
        />
        <MetricRow
          metricKey="answer_accuracy"
          label="Answer Accuracy"
          description="correct answer is present in options"
          value={metrics.answer_accuracy}
        />
        <MetricRow
          metricKey="semantic_uniqueness"
          label="Semantic Uniqueness"
          description="no near-duplicate question pairs"
          value={metrics.semantic_uniqueness}
        />
        <MetricRow
          metricKey="concept_overlap"
          label="Concept Overlap"
          description="pairs sharing identical topic tags"
          value={metrics.concept_overlap}
          inverted
        />
        <MetricRow
          metricKey="difficulty_balance"
          label="Difficulty Balance"
          description="matches blueprint easy / medium / hard split"
          value={metrics.difficulty_balance}
        />
        <MetricRow
          metricKey="bloom_accuracy"
          label="Bloom Accuracy"
          description="bloom level is consistent with difficulty"
          value={metrics.bloom_accuracy}
        />
        <MetricRow
          metricKey="rejection_rate"
          label="Rejection Rate"
          description="questions requiring retries during generation"
          value={metrics.rejection_rate}
          inverted
        />
      </div>

      {/* ── Distributions ──────────────────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 gap-4">
        <DistCard
          title="Difficulty Distribution"
          data={metrics.difficulty_dist}
          colorMap={{ easy: "bg-emerald-400", medium: "bg-amber-400", hard: "bg-red-400" }}
          total={metrics.total_questions}
        />
        <DistCard
          title="Bloom Level Distribution"
          data={metrics.bloom_dist}
          colorMap={{
            remember:   "bg-sky-400",
            understand: "bg-violet-400",
            apply:      "bg-teal-400",
            analyze:    "bg-orange-400",
          }}
          total={metrics.total_questions}
        />
      </div>

      {/* ── Chapter coverage ───────────────────────────────────────────── */}
      <div className="card p-5 space-y-4">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
          Chapter Coverage vs Blueprint
        </p>
        <div className="space-y-3">
          {Object.entries(metrics.chapter_coverage).map(([key, cv]) => {
            const label   = key
              .replace(/^chapter_0?/, "Ch ")
              .replace(/_/g, " ")
              .replace(/\b\w/g, c => c.toUpperCase());
            const pct     = cv.max > 0 ? (cv.actual / cv.max) * 100 : 0;
            const inRange = cv.in_range;
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-xs gap-2">
                  <span className="text-slate-700 font-medium truncate">{label}</span>
                  <span className={`flex-shrink-0 font-semibold ${inRange ? "text-emerald-600" : "text-amber-600"}`}>
                    {cv.actual}Q
                    <span className="font-normal text-slate-400 ml-1">
                      (target {cv.min}–{cv.max})
                    </span>
                    <span className="ml-1">{inRange ? "✓" : "⚠"}</span>
                  </span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${inRange ? "bg-brand-500" : "bg-amber-400"}`}
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
