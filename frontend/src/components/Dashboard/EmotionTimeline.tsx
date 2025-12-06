// FILE: frontend/src/components/Dashboard/EmotionTimeline.tsx
import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

export interface EmotionTimelinePoint {
    sessionId: string;
    dateLabel: string;
    stress: number; // 0–100
}

export interface EmotionTimelineProps {
    data: EmotionTimelinePoint[];
}

export const EmotionTimeline: React.FC<EmotionTimelineProps> = ({ data }) => {
    return (
        <div className="h-80 rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-semibold text-slate-100">
                        Stress Trend (Recent Sessions)
                    </h2>
                    <p className="text-xs text-slate-400">
                        Each point represents the average stress for a RIVION session.
                    </p>
                </div>
            </div>

            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis
                        dataKey="dateLabel"
                        tick={{ fill: "#94a3b8", fontSize: 11 }}
                        axisLine={{ stroke: "#1f2937" }}
                    />
                    <YAxis
                        domain={[0, 100]}
                        tick={{ fill: "#94a3b8", fontSize: 11 }}
                        axisLine={{ stroke: "#1f2937" }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: "#020617",
                            border: "1px solid #1f2937",
                            borderRadius: "0.5rem",
                            fontSize: "0.75rem",
                        }}
                        labelStyle={{ color: "#e2e8f0" }}
                    />
                    {/* Reference lines for stress bands */}
                    <ReferenceLine y={40} stroke="#22c55e" strokeDasharray="3 3" />
                    <ReferenceLine y={70} stroke="#facc15" strokeDasharray="3 3" />

                    <Line
                        type="monotone"
                        dataKey="stress"
                        stroke="#6366f1"
                        strokeWidth={2}
                        dot={{ r: 3, stroke: "#6366f1", fill: "#6366f1" }}
                        activeDot={{ r: 5 }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};
