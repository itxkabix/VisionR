// FILE: frontend/src/pages/DashboardPage.tsx
import React from "react";
import { StressGauge } from "@/components/Dashboard/StressGauge";
import { EmotionTimeline } from "@/components/Dashboard/EmotionTimeline";
import { EmotionDistribution } from "@/components/Dashboard/EmotionDistribution";
import { BurnoutRiskCard } from "@/components/Dashboard/BurnoutRiskCard";

export const DashboardPage: React.FC = () => {
    // Mock current stress score
    const currentStress = 68;

    // Mock timeline data: last N sessions
    const stressTimeline = [
        { sessionId: "S1", dateLabel: "Mon", stress: 42 },
        { sessionId: "S2", dateLabel: "Tue", stress: 55 },
        { sessionId: "S3", dateLabel: "Wed", stress: 61 },
        { sessionId: "S4", dateLabel: "Thu", stress: 72 },
        { sessionId: "S5", dateLabel: "Fri", stress: 65 },
        { sessionId: "S6", dateLabel: "Sat", stress: 70 },
        { sessionId: "S7", dateLabel: "Sun", stress: 68 },
    ];

    // Mock emotion distribution over last 7 sessions
    const emotionDistribution = [
        { emotion: "Neutral", value: 10 },
        { emotion: "Happy", value: 6 },
        { emotion: "Stressed", value: 9 },
        { emotion: "Sad", value: 4 },
        { emotion: "Angry", value: 2 },
    ];

    // Mock burnout risk output from backend
    const burnoutRiskLevel = "MEDIUM" as const;
    const burnoutRiskScore = 62;
    const burnoutFactors = [
        "high_average_stress",
        "increasing_stress_trend",
        "persistent_negative_mood",
    ];
    const burnoutRecommendations = [
        "Introduce a fixed daily wind-down routine for at least 20–30 minutes.",
        "Schedule one full break day from work/college tasks this week.",
        "Consider talking to a trusted friend or mentor about ongoing stress.",
        "If stress stays high for 2+ weeks, seek professional mental health support.",
    ];

    return (
        <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
            {/* Header */}
            <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
                <div>
                    <h1 className="text-xl font-semibold tracking-tight">Wellness Dashboard</h1>
                    <p className="text-xs text-slate-400">
                        Overview of your emotional health and stress trends
                    </p>
                </div>
            </header>

            {/* Content */}
            <main className="flex-1 space-y-4 p-4">
                {/* Top row: Stress Gauge + Burnout Card */}
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                    <div className="lg:col-span-1">
                        <StressGauge stressScore={currentStress} label="Current Stress Level" />
                    </div>
                    <div className="lg:col-span-2">
                        <BurnoutRiskCard
                            riskLevel={burnoutRiskLevel}
                            score={burnoutRiskScore}
                            factors={burnoutFactors}
                            recommendations={burnoutRecommendations}
                        />
                    </div>
                </div>

                {/* Middle row: Stress timeline */}
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    <div className="lg:col-span-2">
                        <EmotionTimeline data={stressTimeline} />
                    </div>
                </div>

                {/* Bottom row: Emotion distribution */}
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    <div className="lg:col-span-1">
                        <EmotionDistribution data={emotionDistribution} />
                    </div>
                </div>
            </main>
        </div>
    );
};

export default DashboardPage;
