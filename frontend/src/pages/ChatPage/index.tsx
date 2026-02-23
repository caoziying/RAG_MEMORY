/**
 * Main chat page component.
 */
import React, { useEffect, useState } from 'react';
import { Layout, Spin, Alert, Card, Modal, Typography, Button } from 'antd';

const { Title, Paragraph } = Typography;
import { ConversationList } from '../../components/ConversationList';
import { ChatWindow } from '../../components/ChatWindow';
import { useChatStore } from '../../store/chatStore';

const { Sider, Content } = Layout;

export const ChatPage: React.FC = () => {
  const {
    conversations,
    currentConversationId,
    messages,
    isLoading,
    error,
    fetchConversations,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    clearError,
  } = useChatStore();

  const [guideVisible, setGuideVisible] = useState(false);

  // Load conversations on mount
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // Handle conversation selection
  const handleSelectConversation = (conversationId: string) => {
    selectConversation(conversationId);
  };

  // Handle new conversation creation
  const handleCreateConversation = async () => {
    try {
      await createConversation();
    } catch (error) {
      // Error is handled by store
      throw error; // Re-throw to let caller handle if needed
    }
  };

  // Show guide modal
  const handleShowGuide = () => {
    setGuideVisible(true);
  };

  // Hide guide modal
  const handleHideGuide = () => {
    setGuideVisible(false);
  };

  // Handle conversation deletion
  const handleDeleteConversation = (conversationId: string) => {
    deleteConversation(conversationId);
  };

  // Handle message sending
  const handleSendMessage = (message: string) => {
    sendMessage(message);
  };

  return (
    <>
      <Layout style={{ height: '100vh' }}>
      {/* Error alert */}
      {error && (
        <div style={{ position: 'fixed', top: '16px', right: '16px', zIndex: 1000, maxWidth: '400px' }}>
          <Alert
            message="错误"
            description={error}
            type="error"
            showIcon
            closable
            onClose={clearError}
          />
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && conversations.length === 0 && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1001,
        }}>
          <Spin size="large" tip="加载中..." />
        </div>
      )}

      {/* Sidebar */}
      <Sider
        width={300}
        style={{
          backgroundColor: '#fff',
          borderRight: '1px solid #f0f0f0',
          overflow: 'auto',
        }}
        breakpoint="lg"
        collapsedWidth="0"
      >
        <ConversationList
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onCreateConversation={handleCreateConversation}
          onDeleteConversation={handleDeleteConversation}
        />
      </Sider>

      {/* Main content */}
      <Content style={{ backgroundColor: '#fafafa' }}>
        <ChatWindow
          conversationId={currentConversationId}
          messages={messages}
          onSendMessage={handleSendMessage}
          onCreateConversation={handleCreateConversation}
          onShowGuide={handleShowGuide}
          isLoading={isLoading}
        />
      </Content>
    </Layout>

    {/* Guide Modal */}
    <Modal
      title="使用指南"
      open={guideVisible}
      onCancel={handleHideGuide}
      footer={[
        <Button key="close" onClick={handleHideGuide}>
          关闭
        </Button>
      ]}
      width={700}
    >
      <Typography>
        <Paragraph>
          <strong>欢迎使用RAG对话系统！</strong> 这是一个基于检索增强生成的智能对话系统，可以记忆和检索之前的对话内容。
        </Paragraph>

        <Title level={4}>主要功能</Title>
        <ul>
          <li><strong>对话记忆</strong>：系统会自动记住对话历史，并在后续对话中参考之前的聊天内容</li>
          <li><strong>用户信息提取</strong>：系统会从对话中自动提取您的个人信息（如姓名、职业、兴趣等），并在所有对话中共享</li>
          <li><strong>智能检索</strong>：使用向量检索技术从历史对话中查找相关信息</li>
          <li><strong>多对话管理</strong>：支持创建多个独立对话，每个对话有自己的上下文</li>
        </ul>

        <Title level={4}>使用方法</Title>
        <ol>
          <li><strong>开始新对话</strong>：点击"创建新对话"按钮开始一个新的聊天</li>
          <li><strong>选择对话</strong>：在左侧边栏中选择已有的对话继续聊天</li>
          <li><strong>发送消息</strong>：在下方输入框中输入消息，按Enter发送</li>
          <li><strong>删除对话</strong>：在对话列表中点击删除按钮可以删除不需要的对话</li>
        </ol>

        <Title level={4}>用户信息共享</Title>
        <Paragraph>
          系统会将您在所有对话中提供的个人信息（如姓名、职业、兴趣爱好等）存储到共享的 <code>user.md</code> 文件中。
          这意味着您在对话A中告诉系统的信息，在对话B中也可以被识别和使用。
        </Paragraph>

        <Title level={4}>注意事项</Title>
        <ul>
          <li>系统会保存所有对话记录，请勿分享敏感个人信息</li>
          <li>删除对话会同时删除该对话的所有记录和相关数据</li>
          <li>系统重启后对话记录会保留，用户信息也会持久化存储</li>
        </ul>

        <Paragraph style={{ marginTop: 20, padding: 12, backgroundColor: '#f6ffed', border: '1px solid #b7eb8f' }}>
          💡 <strong>提示</strong>：尝试告诉系统您的姓名和兴趣爱好，然后在新的对话中询问"我是谁？"，系统应该能正确回答。
        </Paragraph>
      </Typography>
    </Modal>
    </>
  );
};