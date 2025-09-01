import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { QuickActions } from './QuickActions';
import { sendChatMessage } from './aiChatService';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  references?: any[];
  forecast_result?: any;
  forecast_config?: any;
  chat_type?: string;
  available_options?: any;
  suggestions?: string[];
}

interface ChatInterfaceProps {
  forecastId?: number;
  onClose?: () => void;
  onForecastGenerated?: (result: any, config: any) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  forecastId, 
  onClose, 
  onForecastGenerated 
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add welcome message on first load
  useEffect(() => {
    setMessages([
      {
        role: 'assistant',
        content: `Hello! I'm your AI forecasting assistant. I can help you:

🔮 **Generate Forecasts** - Just tell me what you want to forecast
📊 **Analyze Data** - Ask about your data statistics and trends  
💡 **Get Insights** - Understand your forecast results
⚙️ **Learn Algorithms** - Discover which forecasting method works best

Try saying something like:
• "Generate a forecast for Product A"
• "Predict monthly sales for Customer X" 
• "Show me my data statistics"
• "What's the best algorithm for seasonal data?"

What would you like to do?`,
        chat_type: 'welcome'
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
        references: response.references,
        forecast_result: response.forecast_result,
        forecast_config: response.forecast_config,
        chat_type: response.chat_type,
        available_options: response.available_options,
        suggestions: response.suggestions
      }]);
      
      // Handle forecast generation
      if (response.forecast_result && response.forecast_config && onForecastGenerated) {
        onForecastGenerated(response.forecast_result, response.forecast_config);
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please make sure the backend server is running and try again.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickAction = (action: string) => {
    handleSendMessage(action);
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      <div className="flex justify-between items-center p-4 border-b">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <h2 className="text-lg font-semibold">AI Forecasting Assistant</h2>
        </div>
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
            <div className="flex items-center space-x-2">
              <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
              <span>AI is analyzing your request...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Quick Actions */}
      <QuickActions onActionClick={handleQuickAction} />
      
      <div className="border-t p-4">
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={loading}
        />
      </div>
    </div>
  );
};
