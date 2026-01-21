import { useState, useCallback } from 'react';
import { clsx } from 'clsx';
import { ChatPanel } from '@/components/ChatPanel';
import { ThoughtsPanel } from '@/components/ThoughtsPanel';
import { useChat } from '@/hooks/useChat';
import { Settings, HelpCircle, Wifi, WifiOff, AlertCircle } from 'lucide-react';

export function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    messages,
    currentPrediction,
    predictionHistory,
    isLoading,
    isStreaming,
    streamingMessageId,
    isConnected,
    sendMessage,
    clearChat,
    selectPrediction,
  } = useChat({
    onError: (message) => {
      setError(message);
      setTimeout(() => setError(null), 5000);
    },
  });

  const handleSendMessage = useCallback(
    async (message: string, image?: File) => {
      setError(null);
      await sendMessage(message, image);
    },
    [sendMessage]
  );

  return (
    <div className="flex flex-col h-screen bg-surface-100">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-surface-200">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-surface-900">Foresight</h1>
            <p className="text-xs text-surface-500">AI that sees the future</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Connection status */}
          <div
            className={clsx(
              'flex items-center gap-1.5 px-2 py-1 rounded-full text-xs',
              isConnected
                ? 'bg-green-50 text-green-700'
                : 'bg-red-50 text-red-700'
            )}
          >
            {isConnected ? (
              <>
                <Wifi className="w-3 h-3" />
                Connected
              </>
            ) : (
              <>
                <WifiOff className="w-3 h-3" />
                Disconnected
              </>
            )}
          </div>

          <button
            onClick={() => setShowHelp(true)}
            className="btn btn-ghost p-2"
            title="Help"
          >
            <HelpCircle className="w-5 h-5 text-surface-500" />
          </button>

          <button
            onClick={() => setShowSettings(true)}
            className="btn btn-ghost p-2"
            title="Settings"
          >
            <Settings className="w-5 h-5 text-surface-500" />
          </button>
        </div>
      </header>

      {/* Error toast */}
      {error && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-50 animate-slide-up">
          <div className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg shadow-lg">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-2 hover:bg-white/20 rounded p-0.5"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" stroke="currentColor" fill="none">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Chat Panel (60%) */}
        <div className="w-[60%] border-r border-surface-200">
          <ChatPanel
            messages={messages}
            isLoading={isLoading}
            isStreaming={isStreaming}
            streamingMessageId={streamingMessageId}
            onSendMessage={handleSendMessage}
            onClearChat={clearChat}
          />
        </div>

        {/* Thoughts Panel (40%) */}
        <div className="w-[40%]">
          <ThoughtsPanel
            currentPrediction={currentPrediction}
            predictionHistory={predictionHistory}
            onSelectPrediction={selectPrediction}
          />
        </div>
      </main>

      {/* Status bar */}
      <footer className="flex items-center justify-between px-6 py-2 bg-surface-800 text-surface-300 text-xs">
        <div className="flex items-center gap-4">
          <span>Model: Qwen2.5-VL-7B + LTX-Video</span>
          <span className="text-surface-500">|</span>
          <span>Hybrid Encoder: DINOv2</span>
        </div>
        <div className="flex items-center gap-4">
          <span>VRAM: --</span>
          <span className="text-surface-500">|</span>
          <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
            {isConnected ? 'Ready' : 'Offline'}
          </span>
        </div>
      </footer>

      {/* Help Modal */}
      {showHelp && (
        <Modal onClose={() => setShowHelp(false)} title="How to use Foresight">
          <div className="space-y-4 text-sm text-surface-600">
            <p>
              Foresight is an AI that can predict visual outcomes. Upload an image
              and ask what will happen next.
            </p>
            <div className="space-y-3">
              <h4 className="font-medium text-surface-800">Example prompts:</h4>
              <ul className="list-disc list-inside space-y-1 text-surface-500">
                <li>"What will happen if I pour water into this glass?"</li>
                <li>"Predict what happens when I push this ball"</li>
                <li>"Show me the next 5 seconds of this scene"</li>
                <li>"What would happen if I turn off the light?"</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-surface-800">Tips:</h4>
              <ul className="list-disc list-inside space-y-1 text-surface-500">
                <li>Use clear, simple scenes for best results</li>
                <li>Focus on physical actions with predictable outcomes</li>
                <li>Short time horizons work better than long ones</li>
              </ul>
            </div>
          </div>
        </Modal>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <Modal onClose={() => setShowSettings(false)} title="Settings">
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-surface-700">
                Video Quality
              </label>
              <select className="input" disabled>
                <option>Standard (512x512)</option>
                <option>High (768x768)</option>
              </select>
              <p className="text-xs text-surface-500">
                Higher quality takes longer to generate
              </p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-surface-700">
                Number of Frames
              </label>
              <select className="input" disabled>
                <option>30 frames (2s)</option>
                <option>60 frames (4s)</option>
                <option>90 frames (6s)</option>
              </select>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-surface-700">
                  Show Metrics
                </label>
                <p className="text-xs text-surface-500">
                  Display quality metrics alongside predictions
                </p>
              </div>
              <input type="checkbox" defaultChecked className="toggle" disabled />
            </div>
            <p className="text-xs text-surface-400 pt-2 border-t">
              Settings are disabled in demo mode
            </p>
          </div>
        </Modal>
      )}
    </div>
  );
}

interface ModalProps {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}

function Modal({ title, children, onClose }: ModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal content */}
      <div className="relative bg-white rounded-xl shadow-xl max-w-md w-full mx-4 animate-slide-up">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200">
          <h2 className="text-lg font-semibold text-surface-900">{title}</h2>
          <button
            onClick={onClose}
            className="btn btn-ghost p-1"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" stroke="currentColor" fill="none">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="px-6 py-4">
          {children}
        </div>
      </div>
    </div>
  );
}
