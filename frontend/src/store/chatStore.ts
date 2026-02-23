/**
 * Zustand store for managing chat state.
 */
import { create } from 'zustand';
import type { ChatState, Conversation, Message } from '../types';
import { conversationApi, chatApi } from '../services/api';

interface ChatStore extends ChatState {
  // Actions
  fetchConversations: () => Promise<void>;
  createConversation: (title?: string) => Promise<string>;
  selectConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

const initialState: ChatState = {
  conversations: [],
  currentConversationId: null,
  messages: [],
  isLoading: false,
  error: null,
};

export const useChatStore = create<ChatStore>((set, get) => ({
  ...initialState,

  fetchConversations: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await conversationApi.listConversations();
      set({ conversations: response.conversations || [], isLoading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
    }
  },

  createConversation: async (title?: string) => {
    set({ isLoading: true, error: null });
    try {
      const conversation = await conversationApi.createConversation(title);

      set((state) => ({
        conversations: [conversation, ...state.conversations],
        currentConversationId: conversation.id,
        messages: [],
        isLoading: false,
      }));

      return conversation.id;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
      throw error;
    }
  },

  selectConversation: async (conversationId: string) => {
    const { currentConversationId } = get();
    if (currentConversationId === conversationId) return;

    set({ isLoading: true, error: null });
    try {
      // If we already have messages for this conversation, don't reload
      const { conversations } = get();
      const conversation = conversations.find(c => c.id === conversationId);

      if (conversation) {
        // Load messages for this conversation
        const response = await conversationApi.getConversation(conversationId);
        set({
          currentConversationId: conversationId,
          messages: response.messages || [],
          isLoading: false,
        });
      } else {
        // Conversation not in list, fetch it
        const response = await conversationApi.getConversation(conversationId);
        set((state) => ({
          conversations: [response, ...state.conversations.filter(c => c.id !== conversationId)],
          currentConversationId: conversationId,
          messages: response.messages || [],
          isLoading: false,
        }));
      }
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
    }
  },

  deleteConversation: async (conversationId: string) => {
    set({ isLoading: true, error: null });
    try {
      await conversationApi.deleteConversation(conversationId);

      set((state) => {
        const newConversations = state.conversations.filter(c => c.id !== conversationId);
        const newCurrentConversationId = state.currentConversationId === conversationId
          ? (newConversations[0]?.id || null)
          : state.currentConversationId;

        const newMessages = newCurrentConversationId === conversationId ? [] : state.messages;

        return {
          conversations: newConversations,
          currentConversationId: newCurrentConversationId,
          messages: newMessages,
          isLoading: false,
        };
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
    }
  },

  sendMessage: async (content: string) => {
    const { currentConversationId, messages } = get();
    set({ isLoading: true, error: null });

    try {
      // Add user message to local state immediately
      const userMessage: Message = {
        id: `temp_${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
        conversation_id: currentConversationId || 'new',
      };

      set((state) => ({
        messages: [...state.messages, userMessage],
      }));

      // Send to API
      const response = await chatApi.sendMessage(content, currentConversationId || undefined);

      // Update conversation ID if this was a new conversation
      const newConversationId = response.conversation_id;
      const isNewConversation = !currentConversationId || currentConversationId !== newConversationId;

      // Add assistant response
      const assistantMessage: Message = {
        id: response.message_id,
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        conversation_id: newConversationId,
      };

      set((state) => {
        let updatedMessages = [...state.messages];

        // Replace temporary user message with permanent one (if needed)
        const tempMessageIndex = updatedMessages.findIndex(m => m.id === userMessage.id);
        if (tempMessageIndex !== -1) {
          updatedMessages[tempMessageIndex] = {
            ...userMessage,
            conversation_id: newConversationId,
          };
        }

        // Add assistant message
        updatedMessages.push(assistantMessage);

        // Update conversations list if this is a new conversation
        let updatedConversations = state.conversations;
        if (isNewConversation) {
          const newConversation: Conversation = {
            id: newConversationId,
            title: content.length > 50 ? content.substring(0, 47) + '...' : content,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            message_count: 2,
          };
          updatedConversations = [newConversation, ...state.conversations];
        } else {
          // Update existing conversation title if it's still the default
          updatedConversations = state.conversations.map(conv => {
            if (conv.id === newConversationId && conv.title === '新对话') {
              return {
                ...conv,
                title: content.length > 50 ? content.substring(0, 47) + '...' : content,
                updated_at: new Date().toISOString(),
                message_count: (conv.message_count || 0) + 2,
              };
            }
            return conv;
          });
        }

        return {
          conversations: updatedConversations,
          currentConversationId: newConversationId,
          messages: updatedMessages,
          isLoading: false,
        };
      });

    } catch (error) {
      // Remove temporary user message on error
      set((state) => ({
        messages: state.messages.filter(m => !m.id.startsWith('temp_')),
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      }));
    }
  },

  clearError: () => {
    set({ error: null });
  },

  reset: () => {
    set(initialState);
  },
}));