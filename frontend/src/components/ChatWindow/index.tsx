/**
 * Main chat window component.
 */
import React, { useState, useRef, useEffect } from 'react';
import { Input, Button, Spin, Empty, Typography, Space } from 'antd';
import { SendOutlined, RobotOutlined } from '@ant-design/icons';
import { MessageBubble } from '../MessageBubble';
import type { Message } from '../../types';

const { TextArea } = Input;
const { Title } = Typography;

interface ChatWindowProps {
  conversationId: string | null;
  messages: Message[];
  onSendMessage: (message: string) => void;
  onCreateConversation: () => Promise<void> | void;
  onShowGuide?: () => void;
  isLoading: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  conversationId,
  messages,
  onSendMessage,
  onCreateConversation,
  onShowGuide,
  isLoading,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [isCreatingConversation, setIsCreatingConversation] = useState(false);
  const safeMessages = messages || [];
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<any>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when conversation changes
  useEffect(() => {
    if (conversationId && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [conversationId]);

  const handleSend = () => {
    const trimmedValue = inputValue.trim();
    if (!trimmedValue) return;

    onSendMessage(trimmedValue);
    setInputValue('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCreateConversationClick = async () => {
    if (isCreatingConversation) return;
    setIsCreatingConversation(true);
    try {
      await onCreateConversation();
    } catch (error) {
      console.error('Failed to create conversation:', error);
    } finally {
      setIsCreatingConversation(false);
    }
  };

  const handleShowGuide = () => {
    if (onShowGuide) {
      onShowGuide();
    } else {
      // Default behavior: open a modal or show alert
      alert('使用指南功能尚未实现。请查看项目README文件获取使用说明。');
    }
  };

  if (!conversationId) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px',
        textAlign: 'center',
      }}>
        <RobotOutlined style={{ fontSize: '64px', color: '#1890ff', marginBottom: '20px' }} />
        <Title level={3} style={{ marginBottom: '16px' }}>
          欢迎使用RAG对话系统
        </Title>
        <p style={{ color: '#666', marginBottom: '24px', maxWidth: '400px' }}>
          这是一个基于检索增强生成的对话系统，可以记忆和检索之前的对话内容。
          请选择一个对话或创建新对话开始聊天。
        </p>
        <Space>
          <Button
            type="primary"
            size="large"
            onClick={handleCreateConversationClick}
            loading={isCreatingConversation}
            disabled={isCreatingConversation}
          >
            创建新对话
          </Button>
          <Button
            size="large"
            onClick={handleShowGuide}
          >
            查看使用指南
          </Button>
        </Space>
      </div>
    );
  }

  return (
    <div className="chat-window" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Messages area */}
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
      }}>
        {safeMessages.length === 0 ? (
          <Empty
            description={
              <span style={{ color: '#666' }}>
                还没有消息，发送一条消息开始对话吧！
              </span>
            }
            style={{ margin: 'auto' }}
          />
        ) : (
          <>
            {safeMessages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isCurrentUser={message.role === 'user'}
              />
            ))}
            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'center', padding: '16px' }}>
                <Spin tip="AI正在思考..." />
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div style={{
        padding: '16px 24px',
        borderTop: '1px solid #f0f0f0',
        backgroundColor: '#fff',
      }}>
        <Space.Compact style={{ width: '100%' }}>
          <TextArea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="输入消息... (按Enter发送，Shift+Enter换行)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            disabled={isLoading}
            style={{ flex: 1 }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={isLoading}
            disabled={!inputValue.trim()}
            style={{ height: 'auto' }}
          >
            发送
          </Button>
        </Space.Compact>
        <div style={{ fontSize: '12px', color: '#999', marginTop: '8px', textAlign: 'center' }}>
          系统基于RAG记忆，可以记住之前的对话内容并提供相关回答
        </div>
      </div>
    </div>
  );
};