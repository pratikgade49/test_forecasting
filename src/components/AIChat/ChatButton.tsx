import React, { useState } from 'react';
import { ChatInterface } from './ChatInterface';

interface ChatButtonProps {
  forecastId?: number;
  onForecastGenerated?: (result: any, config: any) => void;
}

export const ChatButton: React.FC<ChatButtonProps> = ({ forecastId, onForecastGenerated }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-500 to-blue-500 text-white p-4 rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 z-40"
        aria-label="Open AI Chat"
      >
        <div className="relative">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            width="24" 
            height="24" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
        </div>
      </button>
      
      {/* Floating hint when not open */}
      {!isOpen && (
        <div className="fixed bottom-20 right-6 bg-white rounded-lg shadow-lg p-3 border border-gray-200 z-30 max-w-xs">
          <div className="text-sm text-gray-700">
            <div className="flex items-center space-x-2 mb-1">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <span className="font-medium">AI Assistant</span>
            </div>
            <p className="text-xs text-gray-600">
              Ask me to generate forecasts, analyze data, or get insights!
            </p>
          </div>
        </div>
      )}
      
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] z-50">
          <ChatInterface 
            forecastId={forecastId} 
            onClose={() => setIsOpen(false)} 
            onForecastGenerated={onForecastGenerated}
          />
        </div>
      )}
    </>
  );
};
