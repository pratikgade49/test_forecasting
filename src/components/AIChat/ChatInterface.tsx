import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { QuickActions } from './QuickActions';
import { AlgorithmExplorer } from './AlgorithmExplorer';
import { DataInsightsPanel } from './DataInsightsPanel';
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
  const [activePanel, setActivePanel] = useState<'chat' | 'algorithms' | 'data'>('chat');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add welcome message on first load
  useEffect(() => {
    setMessages([
      {
        role: 'assistant',
        content: `Hello! I'm your expert AI forecasting assistant with comprehensive knowledge of 23+ algorithms and data analysis. I can help you:

🔮 **Generate Forecasts** - "Generate a forecast for Product A using Random Forest"
📊 **Analyze Your Data** - "Show me statistics for Customer X" or "What's my top product?"
💡 **Algorithm Expertise** - "Explain ARIMA algorithm" or "Which algorithm is best for seasonal data?"
⚙️ **Data Insights** - "What trends do you see in my data?" or "Analyze my sales patterns"
🎯 **Smart Recommendations** - Get personalized algorithm and forecasting advice
📈 **Business Intelligence** - Understand your data in business context

**Try these examples:**
• "Generate a 6-month forecast for my top product using best fit"
• "Explain the difference between ARIMA and Holt-Winters"
• "What algorithm should I use for seasonal retail data?"
• "Show me insights about Customer ABC's purchasing patterns"
• "Which products have the strongest growth trends?"

I have deep knowledge of your data and all 23 forecasting algorithms. What would you like to explore?`,
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

  const handleAlgorithmSelect = (algorithmKey: string) => {
    handleSendMessage(`Tell me about the ${algorithmKey.replace('_', ' ')} algorithm`);
    setActivePanel('chat');
  };
  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="flex justify-between items-center p-4 border-b">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <h2 className="text-lg font-semibold">Expert AI Assistant</h2>
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
      
      {/* Tab Navigation */}
      <div className="flex border-b bg-gray-50">
        <button
          onClick={() => setActivePanel('chat')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activePanel === 'chat'
              ? 'bg-white text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          💬 Chat
        </button>
        <button
          onClick={() => setActivePanel('algorithms')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activePanel === 'algorithms'
              ? 'bg-white text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          🧠 Algorithms
        </button>
        <button
          onClick={() => setActivePanel('data')}
          className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
            activePanel === 'data'
              ? 'bg-white text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          📊 Data
        </button>
      </div>
      
      <div className="flex-1 overflow-hidden">
        {activePanel === 'chat' && (
          <div className="h-full flex flex-col">
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
        )}
        
        {activePanel === 'algorithms' && (
          <AlgorithmExplorer onAlgorithmSelect={handleAlgorithmSelect} />
        )}
        
        {activePanel === 'data' && (
          <DataInsightsPanel onInsightRequest={handleDataInsightRequest} />
        )}
      </div>
    </div>
  );
};

  const handleDataInsightRequest = (insight: string) => {
    handleSendMessage(insight);
    setActivePanel('chat');
  };