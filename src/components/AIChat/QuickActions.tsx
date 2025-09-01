import React from 'react';
import { TrendingUp, BarChart3, Target, Zap, Calendar, Package } from 'lucide-react';

interface QuickActionsProps {
  onActionClick: (action: string) => void;
}

export const QuickActions: React.FC<QuickActionsProps> = ({ onActionClick }) => {
  const quickActions = [
    {
      icon: <TrendingUp className="w-4 h-4" />,
      label: "Generate Best Fit Forecast",
      action: "Generate a forecast using best fit algorithm",
      color: "bg-purple-100 text-purple-700 hover:bg-purple-200"
    },
    {
      icon: <BarChart3 className="w-4 h-4" />,
      label: "Show Data Stats",
      action: "Show me my data statistics",
      color: "bg-blue-100 text-blue-700 hover:bg-blue-200"
    },
    {
      icon: <Target className="w-4 h-4" />,
      label: "Monthly Forecast",
      action: "Generate a monthly forecast for 6 periods",
      color: "bg-green-100 text-green-700 hover:bg-green-200"
    },
    {
      icon: <Zap className="w-4 h-4" />,
      label: "Quick Analysis",
      action: "Run a quick analysis on my top product",
      color: "bg-orange-100 text-orange-700 hover:bg-orange-200"
    }
  ];

  return (
    <div className="p-3 border-t border-gray-200 bg-gray-50">
      <div className="text-xs text-gray-600 mb-2 font-medium">Quick Actions:</div>
      <div className="grid grid-cols-2 gap-2">
        {quickActions.map((action, index) => (
          <button
            key={index}
            onClick={() => onActionClick(action.action)}
            className={`flex items-center space-x-2 p-2 rounded-lg text-xs font-medium transition-colors ${action.color}`}
          >
            {action.icon}
            <span className="truncate">{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};