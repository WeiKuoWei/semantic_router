const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const sendQuery = async (query, sessionId) => {
  try {
    const response = await fetch(`${API_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId })
    });
    
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error sending query:', error);
    throw error;
  }
};

export const getHistory = async (sessionId) => {
  try {
    const response = await fetch(`${API_URL}/api/history/${sessionId}`);
    
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching history:', error);
    throw error;
  }
};

export const clearHistory = async (sessionId) => {
  try {
    const response = await fetch(`${API_URL}/api/history/${sessionId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error clearing history:', error);
    throw error;
  }
};