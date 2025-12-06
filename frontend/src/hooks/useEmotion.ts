// FILE: frontend/src/hooks/useEmotion.ts
import { useEffect, useState } from "react";

/**
 * Shape of emotion update messages we expect from the backend.
 *
 * NOTE: This is a tentative schema; update it when the FastAPI backend
 * /api/chat/stream WebSocket contract is finalized.
 */
export interface EmotionUpdatePayload {
    emotion: string;
    stress_score: number; // 0–100
    hidden_stress_detected?: boolean;
    // Optional extras for future dashboard / debugging
    final_confidence?: number;
    modality_weights?: Record<string, number>;
}

export interface EmotionState {
    /** Current fused emotion label (e.g., "stressed", "calm"). */
    emotion: string | null;
    /** Stress score (0–100), derived from fusion engine. */
    stressScore: number | null;
    /** True if system has detected hidden stress. */
    hiddenStress: boolean;
    /** Last time (ms since epoch) an emotion update was received. */
    lastUpdatedAt: number | null;
}

/**
 * useEmotion
 *
 * - Subscribes to emotion_update messages coming from the WebSocket created in useWebRTC.
 * - Exposes a simple, UI-friendly state object with current emotion, stress score,
 *   hidden stress flag, and last update time.
 *
 * Usage:
 *   const { socket, ... } = useWebRTC(...);
 *   const emotion = useEmotion(socket);
 */
export const useEmotion = (socket: WebSocket | null): EmotionState => {
    const [emotion, setEmotion] = useState<string | null>("stressed");
    const [stressScore, setStressScore] = useState<number | null>(68);
    const [hiddenStress, setHiddenStress] = useState<boolean>(false);
    const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);

    useEffect(() => {
        if (!socket) return;

        const handleMessage = (event: MessageEvent) => {
            try {
                const data = JSON.parse(event.data);

                // TODO: finalize exact message envelope structure
                // Recommended pattern from backend:
                // { "type": "emotion_update", "payload": { ...EmotionUpdatePayload } }
                if (data.type !== "emotion_update" || !data.payload) {
                    return;
                }

                const payload = data.payload as EmotionUpdatePayload;

                // Defensive checks
                if (
                    typeof payload.stress_score !== "number" ||
                    typeof payload.emotion !== "string"
                ) {
                    return;
                }

                const clampedScore = Math.min(
                    Math.max(Math.round(payload.stress_score), 0),
                    100
                );

                setEmotion(payload.emotion);
                setStressScore(clampedScore);
                setHiddenStress(Boolean(payload.hidden_stress_detected));
                setLastUpdatedAt(Date.now());
            } catch (err) {
                // If it's not JSON or schema is different, just ignore for now.
                // In development, log to console.
                // eslint-disable-next-line no-console
                console.warn("Unexpected WebSocket message format:", event.data);
            }
        };

        socket.addEventListener("message", handleMessage);

        return () => {
            socket.removeEventListener("message", handleMessage);
        };
    }, [socket]);

    return {
        emotion,
        stressScore,
        hiddenStress,
        lastUpdatedAt,
    };
};
