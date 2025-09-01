import React from 'react';
import { TrendingUp, BarChart3, Target, Zap, Brain, Package, Users, MapPin } from 'lucide-react';

interface QuickActionsProps {
  onActionClick: (action: string) => void;
}

export const QuickActions: React.FC<QuickActionsProps> = ({ onActionClick }) => {
  const quickActions = [
    {
      icon: <Target className="w-4 h-4" />,
      label: "Best Fit Forecast",
      action: "Generate a forecast for my top product using best fit algorithm",
      color: "bg-green-100 text-green-700 hover:bg-green-200"
    },
    {
      icon: <BarChart3 className="w-4 h-4" />,
      label: "Data Overview",
      action: "Show me a comprehensive overview of my data with insights and trends",
      color: "bg-blue-100 text-blue-700 hover:bg-blue-200"
    },
    {
      icon: <Brain className="w-4 h-4" />,
      label: "Algorithm Advice",
      action: "Which forecasting algorithm should I use for my data and why?",
      color: "bg-purple-100 text-purple-700 hover:bg-purple-200"
    },
    {
      icon: <TrendingUp className="w-4 h-4" />,
      label: "Trend Analysis",
      action: "Analyze trends and patterns in my historical data",
      color: "bg-orange-100 text-orange-700 hover:bg-orange-200"
    },
    {
      icon: <Package className="w-4 h-4" />,
      label: "Top Products",
      action: "Show me my top 5 products by volume and their trends",
      color: "bg-pink-100 text-pink-700 hover:bg-pink-200"
    },
    {
      icon: <Users className="w-4 h-4" />,
      label: "Customer Insights",
      action: "Analyze my customer data and identify key patterns",
      color: "bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
    }
  ];

  return (
    <div className="p-3 border-t border-gray-200 bg-gray-50">
      <div className="text-xs text-gray-600 mb-2 font-medium">Quick Questions:</div>
      <div className="grid grid-cols-2 gap-2">
        {quickActions.map((action, index) => (
          <button
            key={index}
            onClick={() => onActionClick(action.action)}
            className={`flex items-center space-x-1 p-2 rounded-lg text-xs font-medium transition-colors ${action.color}`}
          >
            {action.icon}
            <span className="truncate">{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};