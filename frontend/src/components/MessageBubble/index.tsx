/**
 * Message bubble component for chat messages.
 */
import React from 'react';
import { Avatar, Typography, Space } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import type { Message } from '../../types';

const { Text, Paragraph } = Typography;

interface MessageBubbleProps {
  message: Message;
  isCurrentUser: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isCurrentUser }) => {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: isCurrentUser ? 'row-reverse' : 'row',
        alignItems: 'flex-start',
        gap: '12px',
        maxWidth: '80%',
        marginLeft: isCurrentUser ? 'auto' : '0',
        marginRight: isCurrentUser ? '0' : 'auto',
      }}
    >
      <Avatar
        icon={isCurrentUser ? <UserOutlined /> : <RobotOutlined />}
        style={{
          backgroundColor: isCurrentUser ? '#1890ff' : '#52c41a',
          flexShrink: 0,
        }}
      />

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: isCurrentUser ? 'flex-end' : 'flex-start',
          gap: '4px',
        }}
      >
        <Space size={8} style={{ fontSize: '12px', color: '#666' }}>
          <Text strong>{isCurrentUser ? '你' : 'AI助手'}</Text>
          <Text type="secondary">{formatTime(message.timestamp)}</Text>
        </Space>

        <div
          style={{
            backgroundColor: isCurrentUser ? '#1890ff' : '#f5f5f5',
            color: isCurrentUser ? '#fff' : '#000',
            padding: '12px 16px',
            borderRadius: '18px',
            borderTopLeftRadius: isCurrentUser ? '18px' : '4px',
            borderTopRightRadius: isCurrentUser ? '4px' : '18px',
            wordBreak: 'break-word',
            whiteSpace: 'pre-wrap',
            maxWidth: '100%',
          }}
        >
          <Paragraph
            style={{
              margin: 0,
              color: isCurrentUser ? '#fff' : 'inherit',
            }}
          >
            {message.content}
          </Paragraph>
        </div>
      </div>
    </div>
  );
};