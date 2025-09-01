import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Package, Users, MapPin, Calendar, Target, Search } from 'lucide-react';

interface DataInsightsPanelProps {
  onInsightRequest: (insight: string) => void;
}

export const DataInsightsPanel: React.FC<DataInsightsPanelProps> = ({ onInsightRequest }) => {
  const [databaseStats, setDatabaseStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDatabaseStats();
  }, []);

  const loadDatabaseStats = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/database/stats', {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined
      });
      
      if (response.ok) {
        const stats = await response.json();
        setDatabaseStats(stats);
      }
    } catch (error) {
      console.error('Failed to load database stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const insightQuestions = [
    {
      category: "Data Overview",
      icon: <BarChart3 className="w-4 h-4" />,
      questions: [
        "Show me my complete data overview",
        "What's the total volume in my database?",
        "What's the date range of my data?",
        "How many unique products do I have?"
      ]
    },
    {
      category: "Product Analysis",
      icon: <Package className="w-4 h-4" />,
      questions: [
        "What are my top 5 products by volume?",
        "Which products have the strongest growth trends?",
        "Show me product performance analysis",
        "Which products have seasonal patterns?"
      ]
    },
    {
      category: "Customer Insights",
      icon: <Users className="w-4 h-4" />,
      questions: [
        "Who are my top customers by volume?",
        "Which customers show growth trends?",
        "Analyze customer purchasing patterns",
        "Show me customer segmentation insights"
      ]
    },
    {
      category: "Location Analysis",
      icon: <MapPin className="w-4 h-4" />,
      questions: [
        "Which locations perform best?",
        "Show me regional performance analysis",
        "Which locations have seasonal patterns?",
        "Compare location performance trends"
      ]
    },
    {
      category: "Trend Analysis",
      icon: <TrendingUp className="w-4 h-4" />,
      questions: [
        "What trends do you see in my data?",
        "Identify seasonal patterns in my data",
        "Show me growth and decline patterns",
        "Analyze demand volatility"
      ]
    },
    {
      category: "Forecasting Recommendations",
      icon: <Target className="w-4 h-4" />,
      questions: [
        "Which algorithm should I use for my data?",
        "Recommend forecasting approach for seasonal data",
        "What's the best interval for my forecasts?",
        "How many historical periods should I use?"
      ]
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b bg-gray-50">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Data Insights</h3>
        </div>
        
        {/* Quick Stats */}
        {databaseStats && (
          <div className="grid grid-cols-2 gap-2 mb-4">
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-blue-600" />
                <div>
                  <p className="text-xs text-blue-600 font-medium">Records</p>
                  <p className="text-sm font-bold text-blue-900">{databaseStats.totalRecords?.toLocaleString()}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-green-50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <Package className="w-4 h-4 text-green-600" />
                <div>
                  <p className="text-xs text-green-600 font-medium">Products</p>
                  <p className="text-sm font-bold text-green-900">{databaseStats.uniqueProducts}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-purple-600" />
                <div>
                  <p className="text-xs text-purple-600 font-medium">Customers</p>
                  <p className="text-sm font-bold text-purple-900">{databaseStats.uniqueCustomers}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-orange-50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <MapPin className="w-4 h-4 text-orange-600" />
                <div>
                  <p className="text-xs text-orange-600 font-medium">Locations</p>
                  <p className="text-sm font-bold text-orange-900">{databaseStats.uniqueLocations}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {insightQuestions.map((category, categoryIndex) => (
          <div key={categoryIndex} className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              {category.icon}
              <h4 className="font-medium text-gray-900">{category.category}</h4>
            </div>
            
            <div className="space-y-2">
              {category.questions.map((question, questionIndex) => (
                <button
                  key={questionIndex}
                  onClick={() => onInsightRequest(question)}
                  className="w-full text-left p-3 bg-white hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all text-sm"
                >
                  <div className="flex items-center justify-between">
                    <span>{question}</span>
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
        
        {/* Custom Question Input */}
        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <div className="flex items-center space-x-2 mb-3">
            <Search className="w-4 h-4 text-blue-600" />
            <h4 className="font-medium text-blue-900">Ask Custom Question</h4>
          </div>
          <p className="text-sm text-blue-700 mb-3">
            Ask any specific question about your data, trends, or patterns.
          </p>
          <div className="space-y-2">
            <button
              onClick={() => onInsightRequest("Analyze the correlation between my products and customers")}
              className="w-full text-left p-2 bg-white hover:bg-blue-100 rounded border border-blue-200 text-sm transition-colors"
            >
              Analyze correlations in my data
            </button>
            <button
              onClick={() => onInsightRequest("What patterns do you see in my historical data?")}
              className="w-full text-left p-2 bg-white hover:bg-blue-100 rounded border border-blue-200 text-sm transition-colors"
            >
              Identify patterns in my data
            </button>
            <button
              onClick={() => onInsightRequest("Recommend the best forecasting strategy for my business")}
              className="w-full text-left p-2 bg-white hover:bg-blue-100 rounded border border-blue-200 text-sm transition-colors"
            >
              Get forecasting strategy recommendations
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};