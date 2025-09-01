import React from 'react';
import { TrendingUp, Target, BarChart3, Lightbulb, AlertCircle, CheckCircle } from 'lucide-react';

interface MessageProps {
  message: {
    role: 'user' | 'assistant';
    content: string;
    references?: any[];
    forecast_result?: any;
    forecast_config?: any;
    chat_type?: string;
    available_options?: any;
    suggestions?: string[];
  };
}

export const ChatMessage: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  const getChatTypeIcon = () => {
    switch (message.chat_type) {
      case 'forecast_generated':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'data_response':
        return <BarChart3 className="w-4 h-4 text-blue-500" />;
      case 'clarification_needed':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Lightbulb className="w-4 h-4 text-purple-500" />;
    }
  };
  
  const getChatTypeColor = () => {
    switch (message.chat_type) {
      case 'forecast_generated':
        return 'border-l-green-500 bg-green-50';
      case 'data_response':
        return 'border-l-blue-500 bg-blue-50';
      case 'clarification_needed':
        return 'border-l-yellow-500 bg-yellow-50';
      case 'error':
        return 'border-l-red-500 bg-red-50';
      default:
        return 'border-l-gray-300 bg-gray-50';
    }
  };
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`max-w-4/5 p-4 rounded-lg ${
          isUser 
            ? 'bg-blue-500 text-white rounded-br-none shadow-md' 
            : `${getChatTypeColor()} text-gray-800 rounded-bl-none border-l-4 shadow-sm`
        }`}
      >
        {!isUser && message.chat_type && message.chat_type !== 'general' && (
          <div className="flex items-center space-x-2 mb-2">
            {getChatTypeIcon()}
            <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
              {message.chat_type.replace('_', ' ')}
            </span>
          </div>
        )}
        
        <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
        
        {/* Forecast Result Summary */}
        {message.forecast_result && (
          <div className="mt-3 p-3 bg-white rounded-lg border border-green-200">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium text-green-800">Forecast Generated</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-medium text-green-700 ml-1">
                  {message.forecast_result.accuracy}%
                </span>
              </div>
              <div>
                <span className="text-gray-600">Algorithm:</span>
                <span className="font-medium text-gray-700 ml-1">
                  {message.forecast_result.selectedAlgorithm}
                </span>
              </div>
            </div>
          </div>
        )}
        
        {/* Available Options */}
        {message.available_options && (
          <div className="mt-3 p-3 bg-white rounded-lg border border-blue-200">
            <div className="text-xs text-gray-600 space-y-2">
              {message.available_options.products && (
                <div>
                  <span className="font-medium">Products:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {message.available_options.products.slice(0, 5).map((product: string) => (
                      <span key={product} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        {product}
                      </span>
                    ))}
                    {message.available_options.products.length > 5 && (
                      <span className="text-gray-500 text-xs">
                        +{message.available_options.products.length - 5} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Suggestions */}
        {message.suggestions && (
          <div className="mt-3 space-y-1">
            {message.suggestions.map((suggestion, index) => (
              <button
                key={index}
                className="block w-full text-left p-2 bg-white hover:bg-gray-50 rounded border border-gray-200 text-xs text-gray-700 transition-colors"
                onClick={() => {
                  // You could implement quick suggestion clicking here
                  console.log('Suggestion clicked:', suggestion);
                }}
              >
                💡 {suggestion}
              </button>
            ))}
          </div>
        )}
        
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
