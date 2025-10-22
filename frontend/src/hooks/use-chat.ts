import apiClient from '@/lib/api-client';
import type { ChunkResponse } from '@/types/api';
import { useCallback, useEffect, useRef, useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: ChunkResponse[];
}

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
  
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load conversation history from backend on mount
  useEffect(() => {
    if (!sessionId) return;

    const loadHistory = async () => {
      try {
        const history = await apiClient.getSessionHistory(sessionId);
        
        // Transform history to messages format
        const loadedMessages: Message[] = [];
        for (const turn of history) {
          if (turn.query) {
            loadedMessages.push({
              role: 'user',
              content: turn.query,
              timestamp: new Date(turn.timestamp || Date.now()),
            });
          }
          if (turn.response) {
            loadedMessages.push({
              role: 'assistant',
              content: turn.response,
              timestamp: new Date(turn.timestamp || Date.now()),
              sources: turn.sources || [],
            });
          }
        }
        
        setMessages(loadedMessages);
      } catch (err) {
        console.error('Error loading chat history:', err);
      }
    };

    loadHistory();
  }, [sessionId]);

  const sendMessage = useCallback(
    async (query: string, filters?: Record<string, any>, useStreaming: boolean = true) => {
      if (!sessionId || !query.trim()) return;

      setError(null);

      // Add user message immediately (optimistic update)
      const userMessage: Message = {
        role: 'user',
        content: query,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      if (useStreaming) {
        // Streaming response
        setIsStreaming(true);
        setCurrentStreamingMessage('');

        // Create new AbortController for this stream
        abortControllerRef.current = new AbortController();

        try {
          const stream = apiClient.chatStream({
            query,
            session_id: sessionId,
            filters,
            stream: true,
          }, abortControllerRef.current.signal);

          let accumulatedMessage = '';
          let sources: ChunkResponse[] = [];
          let finalQueryType = '';
          let finalSessionId = '';

          for await (const chunk of stream) {
            if (chunk.type === 'sources') {
              sources = chunk.sources;
            } else if (chunk.type === 'token') {
              accumulatedMessage += chunk.token;
              setCurrentStreamingMessage(accumulatedMessage);
            } else if (chunk.type === 'complete') {
              finalQueryType = chunk.query_type;
              finalSessionId = chunk.session_id;
              accumulatedMessage = chunk.response;
            } else if (chunk.type === 'error') {
              throw new Error(chunk.error);
            }
          }

          // Add complete assistant message
          const assistantMessage: Message = {
            role: 'assistant',
            content: accumulatedMessage,
            timestamp: new Date(),
            sources,
          };
          setMessages((prev) => [...prev, assistantMessage]);
          setCurrentStreamingMessage('');
        } catch (err) {
          // Don't log abort errors
          if (err instanceof Error && err.name !== 'AbortError') {
            const error = err instanceof Error ? err : new Error('Streaming failed');
            setError(error);
            console.error('Chat streaming error:', error);
          }
        } finally {
          setIsStreaming(false);
          abortControllerRef.current = null;
        }
      } else {
        // Standard response
        setIsLoading(true);

        try {
          const response = await apiClient.chat({
            query,
            session_id: sessionId,
            filters,
            stream: false,
          });

          const assistantMessage: Message = {
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
            sources: response.sources,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        } catch (err) {
          const error = err instanceof Error ? err : new Error('Chat request failed');
          setError(error);
          console.error('Chat error:', error);
        } finally {
          setIsLoading(false);
        }
      }
    },
    [sessionId]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentStreamingMessage('');
  }, []);

  const retryLastMessage = useCallback(async () => {
    if (messages.length === 0) return;

    // Find the last user message
    const lastUserMessage = [...messages].reverse().find((m) => m.role === 'user');
    if (!lastUserMessage) return;

    // Remove all messages after the last user message
    const indexOfLastUser = messages.lastIndexOf(lastUserMessage);
    setMessages((prev) => prev.slice(0, indexOfLastUser + 1));

    // Resend the message
    await sendMessage(lastUserMessage.content);
  }, [messages, sendMessage]);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setCurrentStreamingMessage('');
  }, []);

  return {
    messages,
    isStreaming,
    isLoading,
    error,
    currentStreamingMessage,
    sendMessage,
    clearMessages,
    retryLastMessage,
    stopStreaming,
  };
}
