/**
 * Conversation list sidebar component.
 */
import React from 'react';
import { List, Button, Typography, Space, Modal, message } from 'antd';
import { PlusOutlined, DeleteOutlined, MessageOutlined } from '@ant-design/icons';
import type { Conversation } from '../../types';

const { Title, Text } = Typography;

interface ConversationListProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (conversationId: string) => void;
  onCreateConversation: () => void;
  onDeleteConversation: (conversationId: string) => void;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  currentConversationId,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
}) => {
  const handleDelete = (conversationId: string, title: string) => {
    Modal.confirm({
      title: '删除对话',
      content: `确定要删除对话 "${title}" 吗？`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: () => {
        onDeleteConversation(conversationId);
        message.success('对话已删除');
      },
    });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '刚刚';
    if (diffMins < 60) return `${diffMins}分钟前`;
    if (diffHours < 24) return `${diffHours}小时前`;
    if (diffDays < 7) return `${diffDays}天前`;
    return date.toLocaleDateString('zh-CN');
  };

  return (
    <div className="conversation-list" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid #f0f0f0' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Title level={4} style={{ margin: 0 }}>
            <MessageOutlined /> 对话
          </Title>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={onCreateConversation}
            block
          >
            新建对话
          </Button>
        </Space>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        <List
          dataSource={conversations}
          locale={{ emptyText: '暂无对话，点击"新建对话"开始聊天' }}
          renderItem={(conversation) => (
            <List.Item
              style={{
                padding: '12px 16px',
                cursor: 'pointer',
                backgroundColor: currentConversationId === conversation.id ? '#e6f7ff' : 'transparent',
                borderRight: currentConversationId === conversation.id ? '3px solid #1890ff' : 'none',
              }}
              onClick={() => onSelectConversation(conversation.id)}
              actions={[
                <Button
                  key="delete"
                  type="text"
                  icon={<DeleteOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(conversation.id, conversation.title);
                  }}
                  size="small"
                  danger
                />,
              ]}
            >
              <List.Item.Meta
                title={
                  <Text
                    strong
                    ellipsis={{ tooltip: conversation.title }}
                    style={{ fontSize: '14px' }}
                  >
                    {conversation.title}
                  </Text>
                }
                description={
                  <Space direction="vertical" size={2} style={{ fontSize: '12px' }}>
                    <Text type="secondary">
                      {formatDate(conversation.updated_at)}
                    </Text>
                    {conversation.message_count !== undefined && (
                      <Text type="secondary">
                        {conversation.message_count} 条消息
                      </Text>
                    )}
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </div>

      <div style={{ padding: '16px', borderTop: '1px solid #f0f0f0', fontSize: '12px', color: '#999' }}>
        共 {conversations.length} 个对话
      </div>
    </div>
  );
};