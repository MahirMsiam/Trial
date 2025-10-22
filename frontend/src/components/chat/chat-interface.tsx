'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ChatFilters, ChunkResponse } from '@/types/api';
import { Send, Settings, StopCircle, Trash2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import MessageBubble from './message-bubble';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: ChunkResponse[];
}

interface ChatInterfaceProps {
  sessionId: string | null;
  messages: Message[];
  isStreaming: boolean;
  isLoading: boolean;
  error: Error | null;
  currentStreamingMessage: string;
  onSendMessage: (query: string, filters?: Record<string, any>, useStreaming?: boolean) => Promise<void>;
  onClearMessages: () => void;
  onStopStreaming: () => void;
  onSourceClick: (caseId: number) => void;
}

export default function ChatInterface({
  sessionId,
  messages,
  isStreaming,
  isLoading,
  error,
  currentStreamingMessage,
  onSendMessage,
  onClearMessages,
  onStopStreaming,
  onSourceClick,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [chatFilters, setChatFilters] = useState<ChatFilters>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check if streaming is enabled via environment variable (defaults to true)
  const isStreamingEnabled = 
    process.env.NEXT_PUBLIC_ENABLE_STREAMING !== 'false';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingMessage]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId) return;

    const query = input;
    setInput('');
    await onSendMessage(query, chatFilters, isStreamingEnabled);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="border-b">
        <div className="flex items-center justify-between">
          <CardTitle>Chat</CardTitle>
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings className="h-4 w-4 mr-2" />
              Filters
            </Button>
            <Button variant="outline" size="sm" onClick={onClearMessages}>
              <Trash2 className="h-4 w-4 mr-2" />
              Clear
            </Button>
          </div>
        </div>

        {/* Filter Settings Panel */}
        {showSettings && (
          <div className="mt-4 p-4 border rounded-lg bg-muted/50 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold">Chat Filters</h4>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setChatFilters({})}
              >
                <X className="h-3 w-3 mr-1" />
                Clear All
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-medium">Case Type</label>
                <Select
                  value={chatFilters.case_type || ''}
                  onValueChange={(value) =>
                    setChatFilters({ ...chatFilters, case_type: value || undefined })
                  }
                >
                  <SelectTrigger className="h-9">
                    <SelectValue placeholder="Any" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Any</SelectItem>
                    <SelectItem value="Criminal Appeal">Criminal Appeal</SelectItem>
                    <SelectItem value="Civil Appeal">Civil Appeal</SelectItem>
                    <SelectItem value="Writ Petition">Writ Petition</SelectItem>
                    <SelectItem value="Civil Revision">Civil Revision</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Court</label>
                <Select
                  value={chatFilters.court_name || ''}
                  onValueChange={(value) =>
                    setChatFilters({ ...chatFilters, court_name: value || undefined })
                  }
                >
                  <SelectTrigger className="h-9">
                    <SelectValue placeholder="Any" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">Any</SelectItem>
                    <SelectItem value="Appellate Division">Appellate Division</SelectItem>
                    <SelectItem value="High Court Division">High Court Division</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium">Year</label>
                <Input
                  type="text"
                  placeholder="e.g., 2023"
                  value={chatFilters.year || ''}
                  onChange={(e) =>
                    setChatFilters({ ...chatFilters, year: e.target.value || undefined })
                  }
                  className="h-9"
                />
              </div>
            </div>

            {Object.keys(chatFilters).length > 0 && (
              <div className="text-xs text-muted-foreground">
                Active filters will be applied to all chat queries
              </div>
            )}
          </div>
        )}
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !currentStreamingMessage && (
          <div className="text-center text-muted-foreground py-8">
            <p>Start a conversation by asking a question about legal judgments</p>
            <p className="text-sm mt-2">Try: &quot;What are the requirements for filing a writ petition?&quot;</p>
          </div>
        )}

        {messages.map((message, idx) => (
          <MessageBubble
            key={idx}
            message={message}
            isStreaming={false}
            onSourceClick={onSourceClick}
          />
        ))}

        {currentStreamingMessage && (
          <MessageBubble
            message={{
              role: 'assistant',
              content: currentStreamingMessage,
              timestamp: new Date(),
            }}
            isStreaming={true}
            onSourceClick={onSourceClick}
          />
        )}

        {error && (
          <div className="text-destructive text-sm">
            Error: {error.message}
          </div>
        )}

        <div ref={messagesEndRef} />
      </CardContent>

      <div className="border-t p-4">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            className="flex-1 min-h-[60px] max-h-[120px] resize-none rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isLoading || isStreaming || !sessionId}
          />
          <div className="flex flex-col gap-2">
            {isStreaming ? (
              <Button onClick={onStopStreaming} variant="destructive" size="icon">
                <StopCircle className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || !sessionId}
                size="icon"
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
