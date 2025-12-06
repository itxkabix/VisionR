// FILE: frontend/src/hooks/useWebRTC.ts
import { useCallback, useEffect, useRef, useState } from "react";

export interface UseWebRTCOptions {
    /** Desired video constraints – can be tuned later */
    video?: MediaTrackConstraints;
    /** Desired audio constraints – can be tuned later */
    audio?: MediaTrackConstraints | boolean;
}

export interface UseWebRTCResult {
    /** Local combined MediaStream from getUserMedia (video + audio). */
    localStream: MediaStream | null;
    /** True when camera/mic are active and stream is available. */
    isMediaReady: boolean;
    /** WebSocket connection status to backend streaming endpoint. */
    isSocketConnected: boolean;
    /** Underlying WebSocket instance, for hooks like useEmotion to subscribe. */
    socket: WebSocket | null;
    /** Last error message (media or socket). */
    error: string | null;

    /** Start camera + mic. Should be called from a user gesture (click). */
    startMedia: () => Promise<void>;
    /** Stop all local tracks. */
    stopMedia: () => void;

    /** Connect to backend WebSocket for streaming. */
    connectSocket: () => void;
    /** Disconnect WebSocket. */
    disconnectSocket: () => void;

    /**
     * Attach a <video> element to show the local preview.
     * Typically called in <VideoChat> with a ref once media is ready.
     */
    attachVideoElement: (videoEl: HTMLVideoElement | null) => void;

    /**
     * Start sending video frames periodically to backend.
     * FPS and encoding can be tuned later.
     */
    startVideoFrameCapture: (fps?: number) => void;
    /** Stop sending video frames. */
    stopVideoFrameCapture: () => void;

    /**
     * Start streaming audio chunks via WebSocket.
     * Uses ScriptProcessorNode (simple) for now – can be upgraded to AudioWorklet.
     */
    startAudioStreaming: () => void;
    /** Stop streaming audio chunks. */
    stopAudioStreaming: () => void;

    /** Send a text chat message to backend over the same WebSocket. */
    sendTextMessage: (content: string) => void;
}

/**
 * useWebRTC (RIVION version)
 *
 * - Captures local webcam + microphone using getUserMedia.
 * - Manages a WebSocket connection to the backend /api/chat/stream endpoint.
 * - Sends video frames and audio chunks as JSON messages.
 * - Exposes the socket so other hooks (e.g. useEmotion) can subscribe to events.
 *
 * NOTE:
 * - This hook currently uses plain WebSocket, not full P2P WebRTC.
 *   For RIVION, the backend is the "peer" that receives media for emotion analysis.
 */
export const useWebRTC = (
    options: UseWebRTCOptions = {}
): UseWebRTCResult => {
    const {
        video = { width: 640, height: 480 },
        audio = true,
    } = options;

    const [localStream, setLocalStream] = useState<MediaStream | null>(null);
    const [isMediaReady, setIsMediaReady] = useState<boolean>(false);
    const [isSocketConnected, setIsSocketConnected] = useState<boolean>(false);
    const [socket, setSocket] = useState<WebSocket | null>(null);
    const [error, setError] = useState<string | null>(null);

    const videoElementRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const frameIntervalRef = useRef<number | null>(null);

    const audioContextRef = useRef<AudioContext | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);

    // -----------------------------
    // Media: Start / Stop
    // -----------------------------

    const startMedia = useCallback(async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video,
                audio,
            });

            setLocalStream(stream);
            setIsMediaReady(true);

            // Attach to video element if already available
            if (videoElementRef.current) {
                videoElementRef.current.srcObject = stream;
                videoElementRef.current.play().catch(() => {
                    // Autoplay may be blocked; user must click to play
                });
            }
        } catch (err) {
            console.error("Error starting media:", err);
            setError("Unable to access camera/microphone. Check browser permissions.");
            setIsMediaReady(false);
        }
    }, [audio, video]);

    const stopMedia = useCallback(() => {
        localStream?.getTracks().forEach((track) => track.stop());
        setLocalStream(null);
        setIsMediaReady(false);

        // Also stop frame/audio streaming if running
        stopVideoFrameCapture();
        stopAudioStreaming();
    }, [localStream]);

    // -----------------------------
    // WebSocket: Connect / Disconnect
    // -----------------------------

    const connectSocket = useCallback(() => {
        if (socket?.readyState === WebSocket.OPEN) return;

        setError(null);

        // TODO: finalize WebSocket URL and auth (JWT in query params or headers)
        const WS_URL =
            import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/api/chat/stream";

        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            setIsSocketConnected(true);
            // TODO: Send initial handshake (e.g., session_id, auth token)
            // ws.send(JSON.stringify({ type: "join", payload: { session_id, token } }));
        };

        ws.onclose = () => {
            setIsSocketConnected(false);
        };

        ws.onerror = (evt) => {
            console.error("WebSocket error:", evt);
            setError("WebSocket connection error.");
            setIsSocketConnected(false);
        };

        // NOTE: We do NOT set ws.onmessage here – other hooks like useEmotion
        // will attach listeners using addEventListener("message", ...) to keep
        // concerns separated.

        setSocket(ws);
    }, [socket]);

    const disconnectSocket = useCallback(() => {
        if (socket) {
            socket.close();
        }
        setSocket(null);
        setIsSocketConnected(false);
    }, [socket]);

    // -----------------------------
    // Video frame capture
    // -----------------------------

    const attachVideoElement = useCallback(
        (videoEl: HTMLVideoElement | null) => {
            videoElementRef.current = videoEl;
            if (videoEl && localStream) {
                videoEl.srcObject = localStream;
                videoEl.play().catch(() => {
                    // Autoplay may be blocked; that's fine
                });
            }
        },
        [localStream]
    );

    const startVideoFrameCapture = useCallback(
        (fps: number = 5) => {
            if (!videoElementRef.current || !socket || socket.readyState !== WebSocket.OPEN) {
                // Not ready – either no video element or no socket
                return;
            }

            const videoEl = videoElementRef.current;

            // Create off-screen canvas once
            if (!canvasRef.current) {
                canvasRef.current = document.createElement("canvas");
            }
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");

            if (!ctx) return;

            const width = videoEl.videoWidth || 640;
            const height = videoEl.videoHeight || 480;

            canvas.width = width;
            canvas.height = height;

            const intervalMs = 1000 / fps;

            const sendFrame = () => {
                if (
                    !videoElementRef.current ||
                    !canvasRef.current ||
                    !ctx ||
                    !socket ||
                    socket.readyState !== WebSocket.OPEN
                ) {
                    return;
                }

                try {
                    ctx.drawImage(videoEl, 0, 0, width, height);
                    canvas.toBlob(
                        (blob) => {
                            if (!blob) return;
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64data = reader.result as string;

                                // TODO: finalize message schema for video_frame
                                const payload = {
                                    type: "video_frame",
                                    payload: {
                                        frame_base64: base64data,
                                        // Optionally include client timestamp
                                        timestamp_ms: Date.now(),
                                    },
                                };

                                socket.send(JSON.stringify(payload));
                            };
                            reader.readAsDataURL(blob);
                        },
                        "image/jpeg",
                        0.7
                    );
                } catch (err) {
                    console.error("Error capturing video frame:", err);
                }
            };

            // use setInterval for now; can be refactored to requestAnimationFrame if needed
            if (frameIntervalRef.current == null) {
                frameIntervalRef.current = window.setInterval(sendFrame, intervalMs);
            }
        },
        [socket]
    );

    const stopVideoFrameCapture = useCallback(() => {
        if (frameIntervalRef.current != null) {
            window.clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }
    }, []);

    // -----------------------------
    // Audio streaming
    // -----------------------------

    const startAudioStreaming = useCallback(() => {
        if (!localStream || !socket || socket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Avoid re-creating if already active
        if (audioContextRef.current || processorRef.current) return;

        const context = new AudioContext();
        audioContextRef.current = context;

        const source = context.createMediaStreamSource(localStream);

        // NOTE: ScriptProcessorNode is deprecated but simple.
        // TODO: migrate to AudioWorkletNode for production.
        const processor = context.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (event: AudioProcessingEvent) => {
            if (!socket || socket.readyState !== WebSocket.OPEN) return;

            const inputBuffer = event.inputBuffer.getChannelData(0); // Float32Array

            // TODO: decide on encoding (e.g., PCM16, Opus, WAV chunk).
            // For now, we send Float32 samples as base64 just as a placeholder.
            const float32Array = new Float32Array(inputBuffer);
            const uint8Array = new Uint8Array(float32Array.buffer);
            const base64Audio = btoa(
                String.fromCharCode(...Array.from(uint8Array))
            );

            const payload = {
                type: "audio_chunk",
                payload: {
                    audio_base64: base64Audio,
                    sample_rate: context.sampleRate,
                    timestamp_ms: Date.now(),
                },
            };

            // This will be large – in production we should compress or batch chunks
            socket.send(JSON.stringify(payload));
        };

        source.connect(processor);
        processor.connect(context.destination); // required in some browsers
    }, [localStream, socket]);

    const stopAudioStreaming = useCallback(() => {
        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current.onaudioprocess = null;
            processorRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(() => undefined);
            audioContextRef.current = null;
        }
    }, []);

    // -----------------------------
    // Text messaging
    // -----------------------------

    const sendTextMessage = useCallback(
        (content: string) => {
            if (!socket || socket.readyState !== WebSocket.OPEN) return;
            if (!content.trim()) return;

            // TODO: finalize schema for chat messages (session_id, message_id, etc.)
            const payload = {
                type: "text_message",
                payload: {
                    content,
                    timestamp_ms: Date.now(),
                },
            };
            socket.send(JSON.stringify(payload));
        },
        [socket]
    );

    // -----------------------------
    // Cleanup on unmount
    // -----------------------------

    useEffect(() => {
        return () => {
            stopVideoFrameCapture();
            stopAudioStreaming();
            stopMedia();
            disconnectSocket();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return {
        localStream,
        isMediaReady,
        isSocketConnected,
        socket,
        error,
        startMedia,
        stopMedia,
        connectSocket,
        disconnectSocket,
        attachVideoElement,
        startVideoFrameCapture,
        stopVideoFrameCapture,
        startAudioStreaming,
        stopAudioStreaming,
        sendTextMessage,
    };
};
