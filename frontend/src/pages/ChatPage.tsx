// FILE: src/pages/ChatPage.tsx
import React, { useState } from "react";
import { VideoChat } from "@/components/VideoChat/VideoChat";
import { EmotionDisplay } from "@/components/VideoChat/EmotionDisplay";
import { ChatBox, ChatMessage } from "@/components/VideoChat/ChatBox";

export const ChatPage: React.FC = () => {
    // Mock emotion state – will later come from fusion engine
    const [currentEmotion, setCurrentEmotion] = useState<string>("stressed");
    const [stressScore, setStressScore] = useState<number>(68);
    const [hiddenStress, setHiddenStress] = useState<boolean>(true);

    // Mock message history
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: "1",
            sender: "user",
            content: "Hey RIVION, I'm feeling very tired lately.",
            timestamp: "10:01",
            emotion: "sad",
        },
        {
            id: "2",
            sender: "ai",
            content:
                "Thanks for sharing that with me. It sounds like you’ve been under a lot of pressure. Do you want to tell me what’s been draining you the most?",
            timestamp: "10:02",
            emotion: "empathetic",
        },
        {
            id: "3",
            sender: "user",
            content: "Mostly work and college. I feel like I can’t catch a break.",
            timestamp: "10:03",
            emotion: "stressed",
        },
    ]);

    const handleSendMessage = (content: string) => {
        if (!content.trim()) return;

        const now = new Date();
        const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

        const userMessage: ChatMessage = {
            id: crypto.randomUUID(),
            sender: "user",
            content: content.trim(),
            timestamp: time,
            emotion: undefined,
        };

        setMessages((prev) => [...prev, userMessage]);

        // Simple mock: update emotion based on content keywords
        const lower = content.toLowerCase();
        if (lower.includes("tired") || lower.includes("exhausted") || lower.includes("burnt")) {
            setCurrentEmotion("stressed");
            setStressScore(78);
            setHiddenStress(true);
        } else if (lower.includes("better") || lower.includes("relaxed")) {
            setCurrentEmotion("calm");
            setStressScore(42);
            setHiddenStress(false);
        }

        // Simple mock AI response
        const mockResponses = [
            "I hear you. Let’s take this one step at a time. What’s the biggest stressor today?",
            "That sounds really heavy. Would you like to try a small breathing exercise together?",
            "It’s okay to feel this way. We can explore what’s making it harder recently.",
        ];
        const randomResponse =
            mockResponses[Math.floor(Math.random() * mockResponses.length)];

        const aiMessage: ChatMessage = {
            id: crypto.randomUUID(),
            sender: "ai",
            content: randomResponse,
            timestamp: time,
            emotion: "supportive",
        };

        setTimeout(() => {
            setMessages((prev) => [...prev, aiMessage]);
        }, 400);
    };

    return (
        <div className="flex h-screen w-full flex-col bg-slate-950 text-slate-100">
            {/* Header */}
            <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
                <div className="flex items-center gap-2">
                    <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-indigo-600 text-sm font-bold">
                        R
                    </span>
                    <div>
                        <h1 className="text-lg font-semibold tracking-tight">RIVION</h1>
                        <p className="text-xs text-slate-400">
                            Real-time emotion-aware mental wellness chat
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                    <span>Session active</span>
                </div>
            </header>

            {/* Main layout: 2 columns on large screens */}
            <main className="grid flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-2">
                {/* Left: Video + Emotion */}
                <section className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
                    <VideoChat />
                    <EmotionDisplay
                        emotion={currentEmotion}
                        stressScore={stressScore}
                        hiddenStress={hiddenStress}
                    />
                </section>

                {/* Right: Chat */}
                <section className="flex flex-col rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
                    <ChatBox messages={messages} onSendMessage={handleSendMessage} />
                </section>
            </main>
        </div>
    );
};

export default ChatPage;
