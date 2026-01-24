import { clsx } from 'clsx';
import Markdown from 'react-markdown';
import type { ChatMessage } from '@/types';
import { User, Bot } from 'lucide-react';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming = false }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={clsx(
        'flex gap-3 animate-slide-up',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-primary-600' : 'bg-surface-200'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-surface-600" />
        )}
      </div>

      {/* Message content */}
      <div className={clsx('flex flex-col gap-2', isUser ? 'items-end' : 'items-start')}>
        {/* Image attachment */}
        {message.imageUrl && (
          <div className="max-w-[300px] rounded-lg overflow-hidden border border-surface-200">
            <img
              src={message.imageUrl}
              alt="Uploaded image"
              className="w-full h-auto object-cover"
            />
          </div>
        )}

        {/* Text content */}
        {message.content && (
          <div className={clsx('message-bubble', isUser ? 'user' : 'assistant')}>
            {isUser ? (
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </p>
            ) : (
              <div className="prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-headings:my-2">
                <Markdown>{message.content}</Markdown>
                {isStreaming && (
                  <span className="inline-block w-1.5 h-4 bg-current ml-0.5 animate-pulse" />
                )}
              </div>
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-surface-400 px-1">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-surface-200">
        <Bot className="w-4 h-4 text-surface-600" />
      </div>
      <div className="typing-indicator">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}
