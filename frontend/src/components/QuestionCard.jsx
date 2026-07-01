import { useState } from "react";
import Badge from "./Badge";
import { difficultyColor, bloomColor } from "../lib/utils";

const OPTION_LABELS = ["A", "B", "C", "D"];

export default function QuestionCard({ question, index }) {
  const [showAnswer, setShowAnswer] = useState(false);

  const q = question;
  const correctIdx = q.options?.indexOf(q.correct_answer);

  return (
    <div className="card p-5 space-y-4 hover:shadow-md transition-shadow">
      {/* Question header */}
      <div className="flex items-start gap-3">
        <span className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-50 text-brand-700 font-bold text-sm
                         flex items-center justify-center border border-brand-200">
          {index}
        </span>
        <p className="text-slate-900 font-medium leading-relaxed pt-1">{q.question}</p>
      </div>

      {/* Options */}
      <div className="grid sm:grid-cols-2 gap-2 ml-11">
        {q.options?.map((opt, i) => {
          const isCorrect = showAnswer && i === correctIdx;
          return (
            <div
              key={i}
              className={`flex items-start gap-2.5 px-3 py-2 rounded-lg border text-sm
                transition-colors ${
                  isCorrect
                    ? "bg-emerald-50 border-emerald-300 text-emerald-800"
                    : "bg-slate-50 border-slate-200 text-slate-700"
                }`}
            >
              <span className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center
                              text-xs font-bold mt-0.5 ${
                                isCorrect
                                  ? "bg-emerald-500 text-white"
                                  : "bg-slate-200 text-slate-600"
                              }`}>
                {OPTION_LABELS[i]}
              </span>
              <span className="leading-snug">{opt}</span>
            </div>
          );
        })}
      </div>

      {/* Metadata row */}
      <div className="ml-11 flex flex-wrap items-center gap-2">
        <Badge label={q.difficulty} colorClass={difficultyColor(q.difficulty)} />
        <Badge label={q.bloom_level} colorClass={bloomColor(q.bloom_level)} />
        {q.chapter && (
          <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full truncate max-w-[240px]">
            {q.chapter}
          </span>
        )}
        {q.question_type && q.question_type !== "standard" && (
          <span className="text-xs text-purple-600 bg-purple-50 px-2 py-0.5 rounded-full">
            {q.question_type.replace("_", " ")}
          </span>
        )}
      </div>

      {/* Show/hide answer toggle */}
      <div className="ml-11">
        <button
          onClick={() => setShowAnswer(v => !v)}
          className="text-xs font-medium text-brand-600 hover:text-brand-800 transition-colors"
        >
          {showAnswer ? "Hide answer ↑" : "Show answer ↓"}
        </button>
        {showAnswer && (
          <div className="mt-2 px-3 py-2 bg-emerald-50 border border-emerald-200 rounded-lg text-sm text-emerald-800">
            <span className="font-semibold">Correct answer: </span>
            {q.correct_answer}
          </div>
        )}
      </div>
    </div>
  );
}
