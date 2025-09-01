import React, { useState } from 'react';
import { ChatInterface } from './ChatInterface';

interface ChatButtonProps {
  forecastId?: number;
}

export const ChatButton: React.FC<ChatButtonProps> = ({ forecastId }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 bg-blue-500 text-white p-3 rounded-full shadow-lg hover:bg-blue-600 transition-colors"
        aria-label="Open AI Chat"
      >
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
      </button>
      
      {isOpen && (
        <div className="fixed bottom-4 right-4 w-96 h-[500px] z-50">
          <ChatInterface 
            forecastId={forecastId} 
            onClose={() => setIsOpen(false)} 
          />
        </div>
      )}
    </>
  );
};
