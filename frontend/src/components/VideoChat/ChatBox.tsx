import React, { FormEvent, useState } from "react";

export interface ChatMessage {
    id: string;
    sender: "user" | "ai";
    content: string;
    createdAt: string;
}

interface ChatBoxProps {
    onSendMessage: (content: string) => void;
    // Later you can extend this to accept messages as props:
    // messages?: ChatMessage[];
}

/**
 * ChatBox:
 * - Displays conversation messages (user + AI)
 * - Allows user to type and send a new message
 * - Delegates the actual send action to the parent via onSendMessage
 */
export const ChatBox: React.FC<ChatBoxProps> = ({ onSendMessage }) => {
    const [inputValue, setInputValue] = useState<string>("");

    // TODO: Replace local stub with chatStore messages from global state
    const [messages] = useState<ChatMessage[]>([
        // Example starter message just for UI testing
        {
            id: "welcome",
            sender: "ai",
            content: "Hi, I'm RIVION. How are you feeling today?",
            createdAt: new Date().toISOString(),
        },
    ]);

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();
        const trimmed = inputValue.trim();
        if (!trimmed) return;

        onSendMessage(trimmed);
        setInputValue("");
    };

    return (
        <div className="flex flex-col h-full rounded-2xl border border-slate-800 bg-slate-900 shadow-lg">
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`max-w-[80%] px-3 py-2 rounded-xl text-sm ${message.sender === "user"
                                ? "ml-auto bg-emerald-500/10 border border-emerald-500/40"
                                : "mr-auto bg-slate-800 border border-slate-700"
                            }`}
                    >
                        <div className="opacity-60 text-xs mb-1">
                            {message.sender === "user" ? "You" : "RIVION"}
                        </div>
                        <div>{message.content}</div>
                    </div>
                ))}
            </div>

            {/* Input area */}
            <form
                onSubmit={handleSubmit}
                className="border-t border-slate-800 p-3 flex gap-2"
            >
                <input
                    className="flex-1 bg-slate-800 rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-emerald-500/70"
                    placeholder="Type how you feel..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    data-testid="chat-input"
                />
                <button
                    type="submit"
                    className="px-4 py-2 rounded-xl text-sm font-medium bg-emerald-500 hover:bg-emerald-400 text-slate-950 transition-colors"
                    data-testid="send-button"
                >
                    Send
                </button>
            </form>
        </div>
    );
};

export default ChatBox;
