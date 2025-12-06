// FILE: frontend/src/components/Dashboard/EmotionDistribution.tsx
import React from "react";
import {
    PieChart,
    Pie,
    Cell,
    ResponsiveContainer,
    Tooltip,
    Legend,
} from "recharts";

export interface EmotionDistributionEntry {
    emotion: string;
    value: number;
}

export interface EmotionDistributionProps {
    data: EmotionDistributionEntry[];
}

const COLORS = ["#6366f1", "#22c55e", "#f97316", "#ef4444", "#a855f7", "#eab308"];

export const EmotionDistribution: React.FC<EmotionDistributionProps> = ({ data }) => {
    const total = data.reduce((sum, d) => sum + d.value, 0) || 1;

    return (
        <div className="h-72 rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-semibold text-slate-100">
                        Emotion Distribution
                    </h2>
                    <p className="text-xs text-slate-400">
                        Breakdown of dominant emotions across recent sessions.
                    </p>
                </div>
            </div>

            <div className="flex h-[220px] items-center">
                <div className="h-full w-2/3">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                dataKey="value"
                                nameKey="emotion"
                                cx="50%"
                                cy="50%"
                                outerRadius={70}
                                innerRadius={40}
                                paddingAngle={3}
                            >
                                {data.map((entry, index) => (
                                    <Cell
                                        key={`cell-${entry.emotion}`}
                                        fill={COLORS[index % COLORS.length]}
                                        stroke="#020617"
                                    />
                                ))}
                            </Pie>
                            <Tooltip
                                formatter={(value: number, _name, props) => {
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return [`${value} sessions (${percentage}%)`, props.payload?.emotion];
                                }}
                                contentStyle={{
                                    backgroundColor: "#020617",
                                    border: "1px solid #1f2937",
                                    borderRadius: "0.5rem",
                                    fontSize: "0.75rem",
                                }}
                                labelStyle={{ color: "#e2e8f0" }}
                            />
                            <Legend
                                verticalAlign="middle"
                                align="right"
                                layout="vertical"
                                wrapperStyle={{
                                    fontSize: "0.75rem",
                                    color: "#cbd5f5",
                                }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                <div className="w-1/3 pl-3 text-[11px] text-slate-300">
                    <p className="mb-2 font-semibold text-slate-100">
                        Emotional profile (recent):
                    </p>
                    {data.map((d) => {
                        const percentage = ((d.value / total) * 100).toFixed(1);
                        return (
                            <div key={d.emotion} className="flex justify-between">
                                <span>{d.emotion}</span>
                                <span className="text-slate-400">{percentage}%</span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
