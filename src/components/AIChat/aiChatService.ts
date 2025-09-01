import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface ChatResponse {
  message: string;
  references?: any[];
}

export const sendChatMessage = async (
  message: string,
  forecastId?: number
): Promise<ChatResponse> => {
  try {
    const token = localStorage.getItem('access_token');
    const response = await axios.post(`${API_BASE_URL}/ai/chat`, {
      message,
      forecast_id: forecastId,
      context_type: forecastId ? 'specific_forecast' : 'general'
    }, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined
    });
    
    return response.data;
  } catch (error) {
    console.error('Error in AI chat service:', error);
    throw error;
  }
};
