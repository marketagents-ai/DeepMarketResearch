import React, { useState } from 'react';
import { ChatInput } from './components/ChatInput/ChatInput';
import { ChatHistory } from './components/ChatHistory';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Header } from './components/Header/Header';
import { ToolsPanel } from './components/ToolsPanel/ToolsPanel';
import { ChatModeSelector } from './components/ChatModes/ChatModeSelector';
import { useChats } from './hooks/useChats';
import { fetchResearch, sendCustomMessage } from './services/api';
import { handleAPIError } from './utils/api';
import type { ChatItem, ChatMode } from './types/chat';
import { ResearchProvider } from './context/ResearchContext';
import { AIFinancialLoader } from './components/AIFinancialLoader';

export const App: React.FC = () => {
  const { chats, activeChat, setActiveChat, createChat, updateChat, updateChatMode } = useChats();
  const [isLoading, setIsLoading] = useState(false);

  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
  const activeChatMode = chats.find(chat => chat.id === activeChat)?.mode || 'custom';

  const handleSubmit = async (query: string, urls?: string[]) => {
    if (!activeChat) {
      const newChatId = createChat(activeChatMode);
      if (!newChatId) return;
    }
  
    const chatId = activeChat || chats[0].id;
    const userMessage: ChatItem = {
      isUser: true,
      content: query,
      timestamp: new Date(),
      mode: activeChatMode
    };
    
    const newMessages = [...activeMessages, userMessage];
    updateChat(chatId, newMessages);
    
    setIsLoading(true);
    try {
      let responseData;
      
      if (activeChatMode === 'research') {
        responseData = await fetchResearch(query, urls);
      } else {
        // Custom mode
        responseData = await sendCustomMessage(query);
      }
  
      const responseMessage: ChatItem = {
        isUser: false,
        content: responseData,
        timestamp: new Date(),
        mode: activeChatMode
      };
      
      updateChat(chatId, [...newMessages, responseMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: ChatItem = {
        isUser: false,
        content: handleAPIError(error),
        timestamp: new Date(),
        mode: activeChatMode
      };
      updateChat(chatId, [...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <ResearchProvider>
      <div className="h-screen flex flex-col bg-gray-900 text-white">
        <Header />
        <main className="flex-1 flex overflow-hidden">
          <Sidebar
            chats={chats}
            activeChat={activeChat}
            onNewChat={() => createChat(activeChatMode)}
            onSelectChat={setActiveChat}
          />
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-gray-800">
              <ChatModeSelector
                mode={activeChatMode}
                onModeChange={(mode) => updateChatMode(activeChat!, mode)}
                disabled={!activeChat}
              />
            </div>
            <div className="flex-1 flex flex-col overflow-hidden relative">
              <div className="flex-1 overflow-y-auto">
                <ChatHistory messages={activeMessages} />
                {isLoading && activeChatMode === 'research' && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <AIFinancialLoader />
                  </div>
                )}
              </div>
              <div className="sticky bottom-0 border-t border-gray-800 bg-[#0d1117] p-4 z-50">
                <ChatInput
                  mode={activeChatMode}
                  onSubmit={handleSubmit}
                  isLoading={isLoading}
                  disabled={!activeChat}
                />
              </div>
            </div>
          </div>
          <ToolsPanel />
        </main>
      </div>
    </ResearchProvider>
  );
};