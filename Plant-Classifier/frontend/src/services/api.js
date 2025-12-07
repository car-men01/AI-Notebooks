import axios from 'axios';

// Use /api prefix when in production (Docker), localhost in development
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' 
  : (process.env.REACT_APP_API_URL || 'http://localhost:8080');

export const plantApi = {
  // Get all available plant classes
  getClasses: async () => {
    const response = await axios.get(`${API_BASE_URL}/classes`);
    return response.data;
  },

  // Classify plant (basic)
  classifyPlant: async (imageFile) => {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get plant care with 3 methods
  getPlantCare: async (imageFile, onProgress) => {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await axios.post(`${API_BASE_URL}/plant-care`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    return response.data;
  },

  // Get plant care with RAG
  getPlantCareRAG: async (imageFile, onProgress) => {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await axios.post(`${API_BASE_URL}/plant-care-rag`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },
};
