/**
 * Type definitions for the conversation system.
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  conversation_id: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
  metadata?: Record<string, any>;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  stream?: boolean;
}

export interface ChatResponse {
  conversation_id: string;
  response: string;
  message_id: string;
}

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
}

export interface MessageListResponse {
  messages: Message[];
  conversation_id: string;
  total: number;
}

export interface SystemStatus {
  status: string;
  version: string;
  active_conversations: number;
  database_connected: boolean;
  vector_store_enabled: boolean;
}

export interface ApiError {
  error: string;
  detail?: string;
  code?: number;
}

// Store types
export interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

// Component props
export interface ConversationListProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (conversationId: string) => void;
  onCreateConversation: () => void;
  onDeleteConversation: (conversationId: string) => void;
}

export interface ChatWindowProps {
  conversationId: string | null;
  messages: Message[];
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

export interface MessageBubbleProps {
  message: Message;
  isCurrentUser: boolean;
}