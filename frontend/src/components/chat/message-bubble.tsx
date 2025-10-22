'use client';

import { Badge } from '@/components/ui/badge';
import { formatRelativeTime } from '@/lib/utils';
import { ChunkResponse } from '@/types/api';
import { Check, Copy } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: ChunkResponse[];
}

interface MessageBubbleProps {
  message: Message;
  isStreaming: boolean;
  onSourceClick: (caseId: number) => void;
}

export default function MessageBubble({ message, isStreaming, onSourceClick }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-lg bg-primary text-primary-foreground px-4 py-3">
          <p className="whitespace-pre-wrap">{message.content}</p>
          <p className="text-xs opacity-70 mt-1">{formatRelativeTime(message.timestamp)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-lg bg-muted px-4 py-3 space-y-3">
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-foreground animate-pulse" />
        )}

        {!isStreaming && (
          <>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>{formatRelativeTime(message.timestamp)}</span>
              <button onClick={handleCopy} className="hover:text-foreground">
                {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
              </button>
            </div>

            {message.sources && message.sources.length > 0 && (
              <div className="space-y-2 border-t pt-3">
                <p className="text-xs font-medium">
                  Sources ({message.sources.length})
                </p>
                <div className="space-y-2">
                  {message.sources.slice(0, 3).map((source, idx) => (
                    <div
                      key={idx}
                      className="text-xs p-2 bg-background rounded cursor-pointer hover:bg-accent"
                      onClick={() => onSourceClick(source.case_id)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant="outline" className="text-xs">
                          Case #{source.case_id}
                        </Badge>
                        {source.similarity !== undefined && (
                          <span className="text-muted-foreground">
                            {(source.similarity * 100).toFixed(1)}% match
                          </span>
                        )}
                      </div>
                      <p className="line-clamp-2">{source.chunk_text}</p>
                    </div>
                  ))}
                  {message.sources.length > 3 && (
                    <p className="text-xs text-muted-foreground">
                      +{message.sources.length - 3} more sources
                    </p>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
