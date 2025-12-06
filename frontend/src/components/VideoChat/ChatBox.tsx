// FILE: src/components/VideoChat/ChatBox.tsx
import React, { FormEvent, useEffect, useRef, useState } from "react";

export type Sender = "user" | "ai";

export interface ChatMessage {
    id: string;
    sender: Sender;
    content: string;
    timestamp: string;
    emotion?: string;
}

export interface ChatBoxProps {
    messages: ChatMessage[];
    onSendMessage: (content: string) => void;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ messages, onSendMessage }) => {
    const [inputValue, setInputValue] = useState<string>("");
    const [isSending, setIsSending] = useState<boolean>(false);
    const bottomRef = useRef<HTMLDivElement | null>(null);

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (!inputValue.trim()) return;

        setIsSending(true);
        onSendMessage(inputValue.trim());
        setInputValue("");

        // Simulate a short "sending" state
        setTimeout(() => setIsSending(false), 300);
    };

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages.length]);

    return (
        <div className="flex h-full flex-col">
            {/* Header */}
            <div className="mb-3 flex items-center justify-between">
                <div>
                    <h2 className="text-sm font-semibold text-slate-100">Conversation</h2>
                    <p className="text-xs text-slate-400">
                        Share how you feel. RIVION responds based on your emotions.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-slate-400">
                    <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                    <span>Emotion-aware mode</span>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 space-y-2 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                {messages.length === 0 && (
                    <div className="flex h-full items-center justify-center text-xs text-slate-500">
                        No messages yet. Say hi to RIVION!
                    </div>
                )}

                {messages.map((msg) => {
                    const isUser = msg.sender === "user";
                    return (
                        <div
                            key={msg.id}
                            className={`flex w-full ${isUser ? "justify-end" : "justify-start"
                                }`}
                        >
                            <div
                                className={`max-w-[80%] rounded-2xl px-3 py-2 text-xs shadow-sm ${isUser
                                        ? "rounded-br-sm bg-indigo-600 text-white"
                                        : "rounded-bl-sm bg-slate-800 text-slate-50"
                                    }`}
                            >
                                <div className="mb-1 flex items-center justify-between gap-2">
                                    <span className="text-[10px] font-semibold uppercase tracking-wide">
                                        {isUser ? "You" : "RIVION"}
                                    </span>
                                    <span className="text-[10px] text-slate-300/80">
                                        {msg.timestamp}
                                    </span>
                                </div>
                                <p className="whitespace-pre-line text-[13px] leading-relaxed">
                                    {msg.content}
                                </p>
                                {msg.emotion && (
                                    <div className="mt-1 text-[10px] text-slate-200/80">
                                        <span className="opacity-60">Emotion:</span> {msg.emotion}
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}

                <div ref={bottomRef} />
            </div>

            {/* Input area */}
            <form onSubmit={handleSubmit} className="mt-3 flex flex-col gap-2">
                <textarea
                    className="min-h-[60px] w-full resize-none rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    placeholder="Type how you’re feeling right now..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                />

                <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span>RIVION is not a replacement for professional help.</span>
                    <button
                        type="submit"
                        disabled={isSending || !inputValue.trim()}
                        className="inline-flex items-center rounded-full bg-indigo-600 px-4 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700"
                    >
                        {isSending ? "Sending..." : "Send"}
                    </button>
                </div>
            </form>
        </div>
    );
};
