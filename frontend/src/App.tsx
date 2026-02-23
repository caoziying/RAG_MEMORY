/**
 * Main application component.
 */
import React from 'react';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { ChatPage } from './pages/ChatPage';
import './App.css';

const App: React.FC = () => {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <div className="App">
        <ChatPage />
      </div>
    </ConfigProvider>
  );
};

export default App;