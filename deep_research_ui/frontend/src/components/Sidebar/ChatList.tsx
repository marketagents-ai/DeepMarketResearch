import React from 'react';
import { MessageSquare } from 'lucide-react';
import type { Chat } from '../../types/chat';

interface ChatListProps {
  chats: Chat[];
  activeChat: string | null;
  onSelectChat: (chatId: string) => void;
}

export const ChatList: React.FC<ChatListProps> = ({
  chats,
  activeChat,
  onSelectChat,
}) => {
  return (
    <div className="space-y-1 p-2">
      {chats.map((chat) => (
        <button
          key={chat.id}
          onClick={() => onSelectChat(chat.id)}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
            chat.id === activeChat
              ? 'bg-gray-700 text-white'
              : 'text-gray-300 hover:bg-gray-700/50'
          }`}
        >
          <MessageSquare size={18} />
          <span className="truncate">
            {chat.title && chat.title !== 'New Chat' 
              ? chat.title 
              : chat.messages[0]?.content.slice(0, 30) || 'New Chat'}
          </span>
        </button>
      ))}
    </div>
  );
};