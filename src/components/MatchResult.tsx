"use client";

interface MatchResultProps {
  similarity: number;
  confidence: number;
  match: boolean;
}

export default function MatchResult({
  similarity,
  confidence,
  match,
}: MatchResultProps) {
  return (
    <div
      className={`
        w-full max-w-md mx-auto rounded-2xl p-6 text-center transition-all
        ${match
          ? "bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800"
          : "bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800"
        }
      `}
    >
      <div className="flex items-center justify-center gap-3 mb-4">
        {match ? (
          <svg className="w-8 h-8 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ) : (
          <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )}
        <h2
          className={`text-2xl font-bold ${
            match
              ? "text-emerald-700 dark:text-emerald-400"
              : "text-red-700 dark:text-red-400"
          }`}
        >
          {match ? "Match!" : "No Match"}
        </h2>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-zinc-600 dark:text-zinc-400">Confidence</span>
          <span className="font-mono font-semibold">{confidence}%</span>
        </div>
        <div className="w-full h-2 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              match ? "bg-emerald-500" : "bg-red-500"
            }`}
            style={{ width: `${Math.min(100, Math.max(0, confidence))}%` }}
          />
        </div>
        <p className="text-xs text-zinc-500 mt-2">
          Cosine similarity: {similarity.toFixed(4)} (threshold: 0.35)
        </p>
      </div>
    </div>
  );
}
