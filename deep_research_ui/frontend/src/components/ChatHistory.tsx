import React from 'react';
import { ChatMessage } from './ChatMessage';
import type { ChatItem } from '../types/chat';

interface ChatHistoryProps {
  messages: ChatItem[];
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({ messages }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message, index) => (
        <ChatMessage
          key={index}
          isUser={message.isUser}
          content={message.content}
        />
      ))}
    </div>
  );
};