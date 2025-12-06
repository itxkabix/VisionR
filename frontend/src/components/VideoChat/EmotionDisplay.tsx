// FILE: src/components/VideoChat/EmotionDisplay.tsx
import React from "react";

export interface EmotionDisplayProps {
    emotion: string;
    stressScore: number; // 0–100
    hiddenStress: boolean;
}

const emotionLabelMap: Record<string, string> = {
    happy: "Happy",
    sad: "Sad",
    angry: "Angry",
    surprised: "Surprised",
    fearful: "Fearful",
    disgusted: "Disgusted",
    neutral: "Neutral",
    stressed: "Stressed",
    calm: "Calm",
};

export const EmotionDisplay: React.FC<EmotionDisplayProps> = ({
    emotion,
    stressScore,
    hiddenStress,
}) => {
    const normalizedEmotion = emotionLabelMap[emotion] ?? emotion ?? "Unknown";

    const clampedScore = Math.min(Math.max(stressScore, 0), 100);

    const stressColor =
        clampedScore >= 70
            ? "bg-red-500"
            : clampedScore >= 40
                ? "bg-amber-400"
                : "bg-emerald-400";

    const stressLabel =
        clampedScore >= 70
            ? "High stress"
            : clampedScore >= 40
                ? "Moderate stress"
                : "Low stress";

    return (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4 text-sm">
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                        Real-Time Emotional State
                    </h3>
                    <p className="mt-1 text-lg font-semibold text-slate-50">
                        {normalizedEmotion}
                    </p>
                </div>

                <div className="flex flex-col items-end gap-1">
                    <span className="text-xs text-slate-400">Stress score</span>
                    <span className="text-xl font-bold text-slate-50">{clampedScore}</span>
                    <span className="text-[10px] uppercase tracking-wide text-slate-400">
                        {stressLabel}
                    </span>
                </div>
            </div>

            {/* Stress bar */}
            <div className="mt-2">
                <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                    <span>Relaxed</span>
                    <span>Overloaded</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800">
                    <div
                        className={`h-2 rounded-full ${stressColor}`}
                        style={{ width: `${clampedScore}%` }}
                    />
                </div>
            </div>

            {/* Hidden stress + status */}
            <div className="mt-3 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                    <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                    <span className="text-slate-300">Multimodal monitoring active</span>
                </div>
                {hiddenStress && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-1 text-[11px] font-medium text-amber-300">
                        ⚠️ Hidden stress detected
                    </span>
                )}
            </div>
        </div>
    );
};
