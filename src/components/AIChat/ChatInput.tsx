import React, { useState } from 'react';
import { Send, Mic, MicOff } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  disabled = false 
}) => {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Voice input (basic implementation)
  const toggleVoiceInput = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      
      if (!isListening) {
        setIsListening(true);
        recognition.start();
        
        recognition.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          setInput(transcript);
          setIsListening(false);
        };
        
        recognition.onerror = () => {
          setIsListening(false);
        };
        
        recognition.onend = () => {
          setIsListening(false);
        };
      } else {
        recognition.stop();
        setIsListening(false);
      }
    } else {
      alert('Speech recognition is not supported in your browser');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end space-x-2">
      <div className="flex-1">
        <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
        disabled={disabled}
          placeholder="Ask me anything about forecasting, algorithms, or your data..."
          rows={1}
          className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
          style={{ minHeight: '44px', maxHeight: '120px' }}
        />
      </div>
      
      {/* Voice Input Button */}
      <button
        type="button"
        onClick={toggleVoiceInput}
        disabled={disabled}
        className={`p-3 rounded-lg transition-colors ${
          isListening
            ? 'bg-red-500 text-white animate-pulse'
            : disabled
            ? 'bg-gray-300 text-gray-500'
            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
        }`}
        title={isListening ? 'Stop listening' : 'Voice input'}
      >
        {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
      </button>
      
      {/* Send Button */}
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className={`p-3 rounded-lg transition-colors ${
          disabled || !input.trim()
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-gradient-to-r from-purple-500 to-blue-500 text-white hover:from-purple-600 hover:to-blue-600 shadow-md'
        }`}
        title="Send message"
      >
        <Send className="w-5 h-5" />
      </button>
    </form>
  );
};
