// FILE: frontend/src/components/Dashboard/StressGauge.tsx
import React from "react";
import {
    RadialBarChart,
    RadialBar,
    PolarAngleAxis,
    ResponsiveContainer,
} from "recharts";

export interface StressGaugeProps {
    stressScore: number; // 0–100
    label?: string;
}

export const StressGauge: React.FC<StressGaugeProps> = ({ stressScore, label }) => {
    const clamped = Math.min(Math.max(stressScore, 0), 100);

    const data = [
        {
            name: "Stress",
            value: clamped,
            fill:
                clamped >= 70 ? "#ef4444" : clamped >= 40 ? "#facc15" : "#22c55e", // red/amber/green
        },
    ];

    const stressLabel =
        clamped >= 70 ? "High stress" : clamped >= 40 ? "Moderate stress" : "Low stress";

    return (
        <div className="h-full rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-semibold text-slate-100">
                        {label ?? "Current Stress"}
                    </h2>
                    <p className="text-xs text-slate-400">
                        Composite score based on face, voice & text
                    </p>
                </div>
            </div>

            <div className="flex items-center gap-4">
                <div className="h-40 w-40">
                    <ResponsiveContainer width="100%" height="100%">
                        <RadialBarChart
                            cx="50%"
                            cy="50%"
                            innerRadius="70%"
                            outerRadius="100%"
                            barSize={14}
                            data={data}
                            startAngle={180}
                            endAngle={0}
                        >
                            <PolarAngleAxis
                                type="number"
                                domain={[0, 100]}
                                angleAxisId={0}
                                tick={false}
                            />
                            <RadialBar
                                background
                                clockWise
                                cornerRadius={999}
                                dataKey="value"
                            />
                        </RadialBarChart>
                    </ResponsiveContainer>
                </div>

                <div className="flex flex-1 flex-col justify-center gap-2 text-sm">
                    <div className="text-3xl font-bold text-slate-50">{clamped}</div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                        {stressLabel}
                    </div>
                    <p className="text-xs leading-relaxed text-slate-400">
                        This value is calculated from recent sessions using multimodal
                        signals (facial arousal, vocal stress and text negativity).
                    </p>
                    <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-400">
                        <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                        <span>Real-time monitoring active</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
