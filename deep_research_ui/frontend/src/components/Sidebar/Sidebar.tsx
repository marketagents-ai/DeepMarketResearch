import React from 'react';
import { Plus } from 'lucide-react';
import { ChatList } from './ChatList';
import type { Chat } from '../../types/chat';

interface SidebarProps {
  chats: Chat[];
  activeChat: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  chats,
  activeChat,
  onNewChat,
  onSelectChat,
}) => {
  return (
    <div className="w-64 border-r border-gray-700 flex flex-col bg-gray-800">
      <div className="p-4">
        <button
          onClick={onNewChat}
          className="w-full bg-blue-600 text-white p-2 rounded flex items-center justify-center gap-2 hover:bg-blue-700"
        >
          <Plus size={20} />
          New Chat
        </button>
      </div>
      
      <ChatList
        chats={chats}
        activeChat={activeChat}
        onSelectChat={onSelectChat}
      />
    </div>
  );
};