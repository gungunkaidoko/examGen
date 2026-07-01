/**
 * Generic coloured badge pill.
 * Pass `colorClass` as a Tailwind bg+text pair like "bg-emerald-100 text-emerald-700".
 */
export default function Badge({ label, colorClass = "bg-slate-100 text-slate-600" }) {
  if (!label) return null;
  return (
    <span className={`badge ${colorClass}`}>
      {label}
    </span>
  );
}
