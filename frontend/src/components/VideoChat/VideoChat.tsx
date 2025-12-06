// FILE: src/components/VideoChat/VideoChat.tsx
import React from "react";

export interface VideoChatProps {
    // Future props: onToggleMic, onToggleCamera, localStream, remoteStream, etc.
}

export const VideoChat: React.FC<VideoChatProps> = () => {
    return (
        <div className="flex flex-1 flex-col gap-3">
            {/* Video placeholder */}
            <div className="relative flex h-64 w-full items-center justify-center overflow-hidden rounded-2xl border border-slate-800 bg-slate-900">
                <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-80" />
                <div className="relative z-10 flex flex-col items-center justify-center text-center">
                    <div className="mb-3 flex h-16 w-16 items-center justify-center rounded-full border border-dashed border-slate-500/70 bg-slate-900/70 text-slate-300">
                        <span className="text-3xl">🎥</span>
                    </div>
                    <h2 className="text-sm font-semibold tracking-wide text-slate-100">
                        Video Stream Coming Soon
                    </h2>
                    <p className="mt-1 max-w-xs text-xs text-slate-400">
                        Your webcam and microphone will appear here when WebRTC is connected.
                    </p>
                </div>
            </div>

            {/* Simple controls row (non-functional for now) */}
            <div className="flex items-center justify-center gap-3">
                <button
                    type="button"
                    className="inline-flex items-center rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-100 hover:bg-slate-700"
                >
                    <span className="mr-1">🎧</span>
                    Connect Audio
                </button>
                <button
                    type="button"
                    className="inline-flex items-center rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-100 hover:bg-slate-700"
                >
                    <span className="mr-1">📷</span>
                    Enable Camera
                </button>
            </div>
        </div>
    );
};
