/**
 * API service for communicating with the backend.
 */
import axios from 'axios';
import type {
  Conversation,
  ConversationWithMessages,
  ChatRequest,
  ChatResponse,
  ConversationListResponse,
  MessageListResponse,
  SystemStatus,
  ApiError
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE || ''; // Use environment variable or relative paths

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Error handling helper
function handleApiError(error: any): never {
  if (axios.isAxiosError(error)) {
    const apiError = error.response?.data as ApiError;
    throw new Error(apiError?.error || apiError?.detail || error.message);
  }
  throw error;
}

// Conversation APIs
export const conversationApi = {
  // List conversations
  async listConversations(limit: number = 50, offset: number = 0): Promise<ConversationListResponse> {
    try {
      const response = await api.get('/conversations', {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Create conversation
  async createConversation(title?: string): Promise<Conversation> {
    try {
      const response = await api.post('/conversations', {
        title: title || '新对话',
        metadata: {}
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Get conversation with messages
  async getConversation(conversationId: string, limit: number = 100): Promise<ConversationWithMessages> {
    try {
      const response = await api.get(`/conversations/${conversationId}`, {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Update conversation title
  async updateConversationTitle(conversationId: string, title: string): Promise<Conversation> {
    try {
      const response = await api.put(`/conversations/${conversationId}/title`, null, {
        params: { title }
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Delete conversation
  async deleteConversation(conversationId: string): Promise<void> {
    try {
      await api.delete(`/conversations/${conversationId}`);
    } catch (error) {
      return handleApiError(error);
    }
  },
};

// Message APIs
export const messageApi = {
  // List messages in a conversation
  async listMessages(conversationId: string, limit: number = 100, offset: number = 0): Promise<MessageListResponse> {
    try {
      const response = await api.get(`/conversations/${conversationId}/messages`, {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Get a specific message
  async getMessage(conversationId: string, messageId: string) {
    try {
      const response = await api.get(`/conversations/${conversationId}/messages/${messageId}`);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Delete a message
  async deleteMessage(conversationId: string, messageId: string): Promise<void> {
    try {
      await api.delete(`/conversations/${conversationId}/messages/${messageId}`);
    } catch (error) {
      return handleApiError(error);
    }
  },
};

// Chat APIs
export const chatApi = {
  // Send message (auto-create or use existing conversation)
  async sendMessage(message: string, conversationId?: string): Promise<ChatResponse> {
    try {
      const request: ChatRequest = { message, conversation_id: conversationId, stream: false };
      const response = await api.post('/chat', request);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Send message to specific conversation
  async sendMessageToConversation(conversationId: string, message: string): Promise<ChatResponse> {
    try {
      const request: ChatRequest = { message, conversation_id: conversationId, stream: false };
      const response = await api.post(`/chat/${conversationId}`, request);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },
};

// System APIs
export const systemApi = {
  // Get system status
  async getStatus(): Promise<SystemStatus> {
    try {
      const response = await api.get('/system/status');
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Cleanup system
  async cleanup(maxAgeHours: number = 24): Promise<any> {
    try {
      const response = await api.post('/system/cleanup', null, {
        params: { max_age_hours: maxAgeHours }
      });
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },
};

// Health check
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

export default {
  conversationApi,
  messageApi,
  chatApi,
  systemApi,
  healthCheck,
};