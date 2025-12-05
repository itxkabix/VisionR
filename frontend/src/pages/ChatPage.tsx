import React, { useRef } from "react";
import { VideoChat } from "@/components/VideoChat/VideoChat";
import { ChatBox } from "@/components/VideoChat/ChatBox";
import { useEmotion } from "@/hooks/useEmotion";
import { useWebRTC } from "@/hooks/useWebRTC";

/**
 * ChatPage:
 * - Hosts the main video + chat layout.
 * - Wires together WebRTC / WebSocket hooks and child components.
 * - Displays real-time emotion / stress information via the VideoChat area.
 */
export const ChatPage: React.FC = () => {
    const videoRef = useRef<HTMLVideoElement | null>(null);

    const { connectionState, sendTextMessage, sendMediaStream } = useWebRTC();
    const { emotion, stressScore, hiddenStress } = useEmotion();

    // TODO: Wire sendMediaStream to VideoChat / media capture
    // TODO: Load or create sessionId and pass down if needed

    return (
        <div className="h-screen w-full grid grid-cols-2 gap-4 p-4 bg-slate-950 text-slate-50">
            {/* Left side: Video + emotion */}
            <div className="flex flex-col gap-4">
                <VideoChat
                    videoRef={videoRef}
                    emotion={emotion}
                    stressScore={stressScore}
                    hiddenStress={hiddenStress}
                    connectionState={connectionState}
                />
                {/* TODO: Add additional emotion widgets / gauges if needed */}
            </div>

            {/* Right side: Chat box */}
            <div className="flex flex-col">
                <ChatBox
                    onSendMessage={sendTextMessage}
                // TODO: pass messages from chatStore when implemented
                />
            </div>
        </div>
    );
};

export default ChatPage;
