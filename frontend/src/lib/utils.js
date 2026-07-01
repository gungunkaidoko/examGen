/** Format an ISO datetime string to "DD MMM YYYY, HH:MM" */
export function formatDate(isoString) {
  if (!isoString) return "—";
  try {
    return new Date(isoString).toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString.slice(0, 10);
  }
}

/** Map difficulty string → Tailwind colour class pair */
export function difficultyColor(level) {
  switch (level?.toLowerCase()) {
    case "easy":   return "bg-emerald-100 text-emerald-700";
    case "medium": return "bg-amber-100 text-amber-700";
    case "hard":   return "bg-red-100 text-red-700";
    default:       return "bg-slate-100 text-slate-600";
  }
}

/** Map bloom level → colour class pair */
export function bloomColor(level) {
  switch (level?.toLowerCase()) {
    case "remember":   return "bg-sky-100 text-sky-700";
    case "understand": return "bg-violet-100 text-violet-700";
    case "apply":      return "bg-teal-100 text-teal-700";
    case "analyze":    return "bg-orange-100 text-orange-700";
    default:           return "bg-slate-100 text-slate-600";
  }
}

/** Map generation status → colour + label */
export function statusInfo(status) {
  switch (status) {
    case "queued":    return { color: "bg-slate-100 text-slate-600", label: "Queued" };
    case "running":   return { color: "bg-blue-100 text-blue-700", label: "Running" };
    case "completed": return { color: "bg-emerald-100 text-emerald-700", label: "Completed" };
    case "failed":    return { color: "bg-red-100 text-red-700", label: "Failed" };
    default:          return { color: "bg-slate-100 text-slate-600", label: status };
  }
}

/** Zero-pad a number to two digits */
export function pad(n) {
  return String(n).padStart(2, "0");
}
