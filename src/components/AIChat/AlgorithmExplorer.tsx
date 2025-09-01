import React, { useState, useEffect } from 'react';
import { Brain, Target, TrendingUp, Zap, Search, Info, ChevronRight } from 'lucide-react';

interface AlgorithmInfo {
  name: string;
  description: string;
  use_cases: string[];
  pros: string[];
  cons: string[];
  best_for: string;
}

interface AlgorithmExplorerProps {
  onAlgorithmSelect: (algorithmKey: string) => void;
}

export const AlgorithmExplorer: React.FC<AlgorithmExplorerProps> = ({ onAlgorithmSelect }) => {
  const [algorithms, setAlgorithms] = useState<Record<string, AlgorithmInfo>>({});
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'statistical' | 'ml' | 'simple' | 'specialized'>('all');

  useEffect(() => {
    loadAlgorithms();
  }, []);

  const loadAlgorithms = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/ai/algorithms', {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined
      });
      
      if (response.ok) {
        const data = await response.json();
        setAlgorithms(data.algorithms);
      }
    } catch (error) {
      console.error('Failed to load algorithms:', error);
    } finally {
      setLoading(false);
    }
  };

  const algorithmCategories = {
    statistical: ['linear_regression', 'polynomial_regression', 'exponential_smoothing', 'holt_winters', 'arima', 'sarima'],
    ml: ['random_forest', 'xgboost', 'neural_network', 'svr', 'knn', 'gaussian_process', 'lstm_like'],
    simple: ['moving_average', 'ses', 'naive_seasonal', 'drift_method'],
    specialized: ['croston', 'theta_method', 'prophet_like', 'best_fit']
  };

  const getCategoryAlgorithms = () => {
    if (selectedCategory === 'all') {
      return Object.keys(algorithms);
    }
    return algorithmCategories[selectedCategory] || [];
  };

  const filteredAlgorithms = getCategoryAlgorithms().filter(key => {
    const algo = algorithms[key];
    if (!algo) return false;
    
    const searchLower = searchTerm.toLowerCase();
    return (
      algo.name.toLowerCase().includes(searchLower) ||
      algo.description.toLowerCase().includes(searchLower) ||
      algo.best_for.toLowerCase().includes(searchLower) ||
      algo.use_cases.some(uc => uc.toLowerCase().includes(searchLower))
    );
  });

  const getAlgorithmColor = (algorithmKey: string) => {
    if (algorithmCategories.statistical.includes(algorithmKey)) {
      return 'bg-blue-100 text-blue-800 border-blue-200';
    } else if (algorithmCategories.ml.includes(algorithmKey)) {
      return 'bg-green-100 text-green-800 border-green-200';
    } else if (algorithmCategories.simple.includes(algorithmKey)) {
      return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    } else {
      return 'bg-purple-100 text-purple-800 border-purple-200';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'statistical': return '📊';
      case 'ml': return '🤖';
      case 'simple': return '📈';
      case 'specialized': return '🎯';
      default: return '🔍';
    }
  };

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
          <Brain className="w-5 h-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Algorithm Explorer</h3>
          <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
            {Object.keys(algorithms).length} algorithms
          </span>
        </div>
        
        {/* Search */}
        <div className="relative mb-4">
          <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search algorithms..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
        </div>
        
        {/* Category Filter */}
        <div className="flex space-x-1">
          {[
            { key: 'all', label: 'All', icon: '🔍' },
            { key: 'statistical', label: 'Statistical', icon: '📊' },
            { key: 'ml', label: 'ML', icon: '🤖' },
            { key: 'simple', label: 'Simple', icon: '📈' },
            { key: 'specialized', label: 'Specialized', icon: '🎯' }
          ].map(category => (
            <button
              key={category.key}
              onClick={() => setSelectedCategory(category.key as any)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                selectedCategory === category.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {category.icon} {category.label}
            </button>
          ))}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {filteredAlgorithms.length === 0 ? (
          <div className="text-center py-8">
            <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No algorithms found matching your search</p>
          </div>
        ) : (
          filteredAlgorithms.map(algorithmKey => {
            const algo = algorithms[algorithmKey];
            return (
              <div
                key={algorithmKey}
                onClick={() => onAlgorithmSelect(algorithmKey)}
                className={`p-4 border rounded-lg cursor-pointer hover:shadow-md transition-all ${getAlgorithmColor(algorithmKey)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="font-semibold">{algo.name}</h4>
                      <ChevronRight className="w-4 h-4 opacity-60" />
                    </div>
                    <p className="text-sm opacity-90 mb-2">{algo.description}</p>
                    <p className="text-xs opacity-75">
                      <strong>Best for:</strong> {algo.best_for}
                    </p>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {algo.use_cases.slice(0, 3).map((useCase, index) => (
                        <span key={index} className="text-xs bg-white bg-opacity-50 px-2 py-1 rounded">
                          {useCase}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};