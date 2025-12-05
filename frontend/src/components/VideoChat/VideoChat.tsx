import React, { MutableRefObject, useEffect } from "react";

export type EmotionState = {
    label: string;
    confidence: number;
};

interface VideoChatProps {
    videoRef: MutableRefObject<HTMLVideoElement | null>;
    emotion: EmotionState | null;
    stressScore: number | null;
    hiddenStress: boolean;
    connectionState: "disconnected" | "connecting" | "connected";
}

/**
 * VideoChat component:
 * - Requests camera access
 * - Displays local webcam stream
 * - Shows basic emotion & stress overlays
 * - Delegates sending frames to a higher-level hook / service
 */
export const VideoChat: React.FC<VideoChatProps> = ({
    videoRef,
    emotion,
    stressScore,
    hiddenStress,
    connectionState,
}) => {
    useEffect(() => {
        // Basic local media setup; actual streaming handled in a hook/service.
        const enableCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: false, // audio handled separately via Web Audio API
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play().catch(() => {
                        // Handle autoplay blocking, if needed
                    });
                }

                // TODO: pass stream to WebRTC / WebSocket layer (sendMediaStream)
            } catch (error) {
                console.error("Error accessing camera:", error);
                // TODO: show UI feedback for permission errors
            }
        };

        void enableCamera();

        // Optional: cleanup on unmount
        return () => {
            const stream = videoRef.current?.srcObject as MediaStream | null;
            stream?.getTracks().forEach((t) => t.stop());
        };
    }, [videoRef]);

    return (
        <div className="w-full rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 shadow-lg">
            <div className="relative aspect-video bg-black">
                <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    muted
                    playsInline
                />
                {/* Overlay: connection + emotion + stress */}
                <div className="absolute top-3 left-3 flex flex-col gap-1 text-xs bg-slate-900/70 px-3 py-2 rounded-xl">
                    <span className="font-semibold">
                        Connection:{" "}
                        <span
                            className={
                                connectionState === "connected"
                                    ? "text-emerald-400"
                                    : connectionState === "connecting"
                                        ? "text-amber-400"
                                        : "text-red-400"
                            }
                        >
                            {connectionState}
                        </span>
                    </span>
                    <span>
                        Emotion:{" "}
                        <span className="font-semibold">
                            {emotion?.label ?? "detecting..."}
                        </span>
                    </span>
                    <span>
                        Stress:{" "}
                        <span className="font-semibold">
                            {stressScore !== null ? `${stressScore}/100` : "calculating..."}
                        </span>
                    </span>
                    {hiddenStress && (
                        <span className="text-amber-400">
                            Possible hidden stress detected
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};

export default VideoChat;
