/**
 * Horizontal filter bar with search input + three select dropdowns.
 * Props:
 *   filters: { search, chapter, difficulty, bloom_level }
 *   options: { chapters, difficulties, bloom_levels }
 *   onChange(key, value): called on every filter change
 *   onClear(): resets all filters
 */
export default function FilterBar({ filters, options, onChange, onClear }) {
  const hasActive = filters.search || filters.chapter || filters.difficulty || filters.bloom_level;

  return (
    <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
      {/* Search */}
      <div className="relative flex-1 min-w-0">
        <svg
          className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none"
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          type="text"
          placeholder="Search questions…"
          value={filters.search}
          onChange={e => onChange("search", e.target.value)}
          className="w-full pl-9 pr-4 py-2.5 border border-slate-300 rounded-lg text-sm bg-white
                     focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition"
        />
      </div>

      {/* Chapter */}
      <Select
        value={filters.chapter}
        onChange={v => onChange("chapter", v)}
        placeholder="All Chapters"
        options={options?.chapters || []}
      />

      {/* Difficulty */}
      <Select
        value={filters.difficulty}
        onChange={v => onChange("difficulty", v)}
        placeholder="All Difficulties"
        options={options?.difficulties || []}
      />

      {/* Bloom Level */}
      <Select
        value={filters.bloom_level}
        onChange={v => onChange("bloom_level", v)}
        placeholder="All Bloom Levels"
        options={options?.bloom_levels || []}
      />

      {/* Clear */}
      {hasActive && (
        <button
          onClick={onClear}
          className="flex-shrink-0 text-sm font-medium text-slate-500 hover:text-red-600 transition-colors px-1"
        >
          Clear
        </button>
      )}
    </div>
  );
}

function Select({ value, onChange, placeholder, options }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="flex-shrink-0 px-3 py-2.5 border border-slate-300 rounded-lg text-sm bg-white
                 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500
                 text-slate-700 cursor-pointer transition min-w-[160px]"
    >
      <option value="">{placeholder}</option>
      {options.map(opt => (
        <option key={opt} value={opt}>{opt}</option>
      ))}
    </select>
  );
}
