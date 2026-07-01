/**
 * A row of stat cards shown on the landing page.
 * Props: stats { total_questions, total_sets, by_difficulty, by_chapter }
 */
export default function StatsRow({ stats }) {
  if (!stats) return null;

  const items = [
    { label: "Total Questions", value: stats.total_questions?.toLocaleString() || "—", icon: "📝" },
    { label: "Exam Sets", value: stats.total_sets || "—", icon: "📚" },
    { label: "Easy / Medium / Hard", value: buildDiff(stats.by_difficulty), icon: "🎯" },
    { label: "Chapters Covered", value: Object.keys(stats.by_chapter || {}).length || "—", icon: "📖" },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      {items.map(({ label, value, icon }) => (
        <div key={label} className="card p-4 text-center">
          <div className="text-2xl mb-1">{icon}</div>
          <p className="text-xl font-bold text-slate-900">{value}</p>
          <p className="text-xs text-slate-500 mt-0.5">{label}</p>
        </div>
      ))}
    </div>
  );
}

function buildDiff(map) {
  if (!map) return "—";
  return `${map.easy || 0} / ${map.medium || 0} / ${map.hard || 0}`;
}
