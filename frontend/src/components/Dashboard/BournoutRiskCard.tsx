// FILE: frontend/src/components/Dashboard/BurnoutRiskCard.tsx
import React from "react";

export type BurnoutRiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface BurnoutRiskCardProps {
    riskLevel: BurnoutRiskLevel;
    score: number; // 0–100
    factors: string[];
    recommendations: string[];
}

const riskColors: Record<BurnoutRiskLevel, string> = {
    LOW: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    MEDIUM: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    HIGH: "bg-red-500/20 text-red-300 border-red-500/40",
};

const riskLabel: Record<BurnoutRiskLevel, string> = {
    LOW: "Low burnout risk",
    MEDIUM: "Medium burnout risk",
    HIGH: "High burnout risk",
};

export const BurnoutRiskCard: React.FC<BurnoutRiskCardProps> = ({
    riskLevel,
    score,
    factors,
    recommendations,
}) => {
    const clamped = Math.min(Math.max(score, 0), 100);
    const barColor =
        clamped >= 70 ? "bg-red-500" : clamped >= 40 ? "bg-amber-400" : "bg-emerald-400";

    const subtitle =
        riskLevel === "LOW"
            ? "Your stress is manageable, but it's still worth maintaining healthy routines."
            : riskLevel === "MEDIUM"
                ? "Your pattern suggests accumulating stress and mild burnout risk."
                : "Your sessions show sustained high stress and strong burnout indicators.";

    return (
        <div className="h-full rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-semibold text-slate-100">
                        Burnout Risk Assessment
                    </h2>
                    <p className="text-xs text-slate-400">
                        Based on recent stress trends, negative mood and help-seeking frequency.
                    </p>
                </div>
                <div
                    className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide ${riskColors[riskLevel]}`}
                >
                    {riskLabel[riskLevel]}
                </div>
            </div>

            {/* Score Bar */}
            <div className="mb-3">
                <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
                    <span>Burnout score</span>
                    <span className="font-semibold">{clamped} / 100</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800">
                    <div
                        className={`h-2 rounded-full ${barColor}`}
                        style={{ width: `${clamped}%` }}
                    />
                </div>
            </div>

            <p className="mb-3 text-xs leading-relaxed text-slate-300">{subtitle}</p>

            {/* Factors */}
            <div className="mb-3 text-xs">
                <p className="mb-1 font-semibold text-slate-200">Key contributing factors:</p>
                {factors.length === 0 ? (
                    <p className="text-slate-500">
                        No strong burnout factors detected in the recent window.
                    </p>
                ) : (
                    <ul className="flex flex-wrap gap-2">
                        {factors.map((f) => (
                            <li
                                key={f}
                                className="rounded-full bg-slate-800 px-2 py-1 text-[11px] text-slate-200"
                            >
                                {f.replace(/_/g, " ")}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {/* Recommendations */}
            <div className="text-xs">
                <p className="mb-1 font-semibold text-slate-200">
                    Suggested next steps (not a medical diagnosis):
                </p>
                <ul className="list-disc space-y-1 pl-4 text-slate-300">
                    {recommendations.map((r, idx) => (
                        <li key={idx}>{r}</li>
                    ))}
                </ul>
                <p className="mt-2 text-[11px] text-slate-500">
                    If you notice your stress staying high or worsening, consider reaching out to
                    a qualified mental health professional.
                </p>
            </div>
        </div>
    );
};
