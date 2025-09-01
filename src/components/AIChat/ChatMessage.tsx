import React from 'react';

interface MessageProps {
  message: {
    role: 'user' | 'assistant';
    content: string;
    references?: any[];
  };
}

export const ChatMessage: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`max-w-3/4 p-3 rounded-lg ${
          isUser 
            ? 'bg-blue-500 text-white rounded-br-none' 
            : 'bg-gray-100 text-gray-800 rounded-bl-none'
        }`}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
        
        {message.references && message.references.length > 0 && (
          <div className="mt-2 text-xs border-t pt-2">
            <div className="font-semibold">References:</div>
            <ul className="list-disc pl-4">
              {message.references.map((ref, i) => (
                <li key={i}>{JSON.stringify(ref)}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
