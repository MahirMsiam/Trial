'use client';

import CaseDetailsModal from '@/components/case/case-details-modal';
import ChatInterface from '@/components/chat/chat-interface';
import MainLayout from '@/components/layout/main-layout';
import { useChat } from '@/hooks/use-chat';
import { useSession } from '@/hooks/use-session';
import { useState } from 'react';

export default function ChatPage() {
  const { sessionId, clearSession, refreshSession, isSessionExpiring } = useSession();
  const chat = useChat(sessionId);
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);

  const handleNewChat = async () => {
    // Clear the session on backend and create a new one
    await clearSession();
    // Clear local chat state
    chat.clearMessages();
  };

  return (
    <MainLayout currentPage="chat">
      <div className="max-w-6xl mx-auto h-[calc(100vh-12rem)]">
        <div className="space-y-4 mb-4">
          <h1 className="text-3xl font-bold">Legal Research Assistant</h1>
          <p className="text-muted-foreground">
            Ask questions about legal judgments in natural language
          </p>
        </div>

        <ChatInterface
          sessionId={sessionId}
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          isLoading={chat.isLoading}
          error={chat.error}
          currentStreamingMessage={chat.currentStreamingMessage}
          isSessionExpiring={isSessionExpiring}
          onSendMessage={chat.sendMessage}
          onClearMessages={chat.clearMessages}
          onNewChat={handleNewChat}
          onRefreshSession={refreshSession}
          onStopStreaming={chat.stopStreaming}
          onSourceClick={setSelectedCaseId}
        />

        {selectedCaseId && (
          <CaseDetailsModal
            caseId={selectedCaseId}
            isOpen={!!selectedCaseId}
            onClose={() => setSelectedCaseId(null)}
          />
        )}
      </div>
    </MainLayout>
  );
}
