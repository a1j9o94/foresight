import { useState, useRef, useEffect, useCallback } from 'react';
import { clsx } from 'clsx';
import type { ChatMessage } from '@/types';
import { MessageBubble, TypingIndicator } from './MessageBubble';
import { Send, ImagePlus, X, Trash2 } from 'lucide-react';

interface ChatPanelProps {
  messages: ChatMessage[];
  isLoading?: boolean;
  isStreaming?: boolean;
  streamingMessageId?: string;
  onSendMessage: (message: string, image?: File) => void;
  onClearChat?: () => void;
  className?: string;
}

export function ChatPanel({
  messages,
  isLoading = false,
  isStreaming = false,
  streamingMessageId,
  onSendMessage,
  onClearChat,
  className,
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
  }, [inputValue]);

  // Handle image selection
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image must be less than 10MB');
      return;
    }

    setSelectedImage(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const removeImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();

    const trimmedInput = inputValue.trim();
    if (!trimmedInput && !selectedImage) return;
    if (isLoading) return;

    onSendMessage(trimmedInput, selectedImage || undefined);
    setInputValue('');
    removeImage();
  }, [inputValue, selectedImage, isLoading, onSendMessage, removeImage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }, [handleSubmit]);

  return (
    <div className={clsx('panel', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-200">
        <div>
          <h2 className="text-lg font-semibold text-surface-900">Chat</h2>
          <p className="text-xs text-surface-500">Ask about future states</p>
        </div>
        {messages.length > 0 && onClearChat && (
          <button
            onClick={onClearChat}
            className="btn btn-ghost p-2 text-surface-400 hover:text-red-500"
            title="Clear chat"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isStreaming={isStreaming && message.id === streamingMessageId}
              />
            ))}
            {isLoading && !isStreaming && <TypingIndicator />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-surface-200">
        {/* Image preview */}
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img
              src={imagePreview}
              alt="Selected image"
              className="h-20 w-auto rounded-lg border border-surface-200"
            />
            <button
              onClick={removeImage}
              className="absolute -top-2 -right-2 p-1 bg-surface-800 rounded-full text-white hover:bg-surface-700"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          {/* Image upload */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="btn btn-ghost p-2 text-surface-500 hover:text-primary-600"
            title="Upload image"
          >
            <ImagePlus className="w-5 h-5" />
          </button>

          {/* Text input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="What will happen if..."
              className="input resize-none min-h-[44px] max-h-[120px] py-3 pr-12"
              rows={1}
              disabled={isLoading}
            />
          </div>

          {/* Send button */}
          <button
            type="submit"
            disabled={isLoading || (!inputValue.trim() && !selectedImage)}
            className={clsx(
              'btn btn-primary p-3',
              (isLoading || (!inputValue.trim() && !selectedImage)) && 'opacity-50 cursor-not-allowed'
            )}
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center mb-4">
        <svg
          className="w-8 h-8 text-primary-600"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
          />
        </svg>
      </div>
      <h3 className="text-lg font-medium text-surface-800 mb-2">
        Start a conversation
      </h3>
      <p className="text-sm text-surface-500 max-w-xs mb-6">
        Upload an image and ask what will happen next. The AI will predict the future and show you a video visualization.
      </p>
      <div className="space-y-2 text-left text-sm text-surface-600">
        <p className="flex items-center gap-2">
          <span className="text-primary-500">1.</span>
          Upload an image of a scene
        </p>
        <p className="flex items-center gap-2">
          <span className="text-primary-500">2.</span>
          Ask "What will happen if..."
        </p>
        <p className="flex items-center gap-2">
          <span className="text-primary-500">3.</span>
          Watch the predicted future
        </p>
      </div>
    </div>
  );
}
