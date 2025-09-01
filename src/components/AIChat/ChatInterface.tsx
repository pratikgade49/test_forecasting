import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { sendChatMessage } from './aiChatService';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  references?: any[];
}

interface ChatInterfaceProps {
  forecastId?: number;
  onClose?: () => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ forecastId, onClose }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add welcome message on first load
  useEffect(() => {
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! I can help you understand your forecast data. What would you like to know?'
      }
    ]);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setLoading(true);
    
    try {
      // Send to backend
      const response = await sendChatMessage(message, forecastId);
      
      // Add AI response to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message,
        references: response.references
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      <div className="flex justify-between items-center p-4 border-b">
        <h2 className="text-lg font-semibold">AI Assistant</h2>
        {onClose && (
          <button 
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ×
          </button>
        )}
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <ChatMessage 
            key={i} 
            message={msg} 
          />
        ))}
        {loading && (
          <div className="flex items-center text-gray-500">
            <div className="animate-pulse">AI is thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="border-t p-4">
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={loading}
        />
      </div>
    </div>
  );
};
