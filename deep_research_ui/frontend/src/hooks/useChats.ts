import { useState, useEffect } from 'react';
import type { Chat, ChatItem, ChatMode } from '../types/chat';

export const useChats = () => {
  const [chats, setChats] = useState<Chat[]>(() => {
    try {
      const saved = localStorage.getItem('chats');
      const parsedChats = saved ? JSON.parse(saved) : [];
      // Filter out empty "New Chat" entries
      return parsedChats.filter((chat: Chat) => 
        chat.messages.length > 0 || chat.title !== 'New Chat'
      );
    } catch (error) {
      console.error('Error loading chats:', error);
      return [];
    }
  });
  
  const [activeChat, setActiveChat] = useState<string | null>(() => {
    try {
      const saved = localStorage.getItem('activeChat');
      return saved || null;
    } catch (error) {
      console.error('Error loading active chat:', error);
      return null;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('chats', JSON.stringify(chats));
    } catch (error) {
      console.error('Error saving chats:', error);
    }
  }, [chats]);

  useEffect(() => {
    try {
      if (activeChat) {
        localStorage.setItem('activeChat', activeChat);
      } else {
        localStorage.removeItem('activeChat');
      }
    } catch (error) {
      console.error('Error saving active chat:', error);
    }
  }, [activeChat]);

  const createChat = (mode: ChatMode = 'custom') => {
    try {
      const newChat: Chat = {
        id: Date.now().toString(),
        title: '',  // Start with empty title
        messages: [],
        mode: mode,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      setChats(prev => [newChat, ...prev]);
      setActiveChat(newChat.id);
      return newChat.id;
    } catch (error) {
      console.error('Error creating chat:', error);
      return null;
    }
  };

  const updateChat = (chatId: string, messages: ChatItem[]) => {
    if (!chatId) return;
    
    setChats(prev => prev.map(chat => {
      if (chat.id === chatId) {
        const firstMessage = messages[0]?.content;
        let title = chat.title;
        
        // Update title only if there are messages and current title is empty or "New Chat"
        if (firstMessage && (!chat.title || chat.title === 'New Chat')) {
          title = typeof firstMessage === 'string' 
            ? firstMessage.slice(0, 30) 
            : 'Chat';
        }
        
        return {
          ...chat,
          title,
          messages,
          updatedAt: new Date()
        };
      }
      return chat;
    }));
  };

  const updateChatMode = (chatId: string, mode: ChatMode) => {
    if (!chatId) return;

    setChats(prev => prev.map(chat =>
      chat.id === chatId
        ? { ...chat, mode, updatedAt: new Date() }
        : chat
    ));
  };

  const deleteChat = (chatId: string) => {
    setChats(prev => prev.filter(chat => chat.id !== chatId));
    if (activeChat === chatId) {
      setActiveChat(null);
    }
  };

  const clearEmptyChats = () => {
    setChats(prev => prev.filter(chat => 
      chat.messages.length > 0 || chat.title !== 'New Chat'
    ));
  };

  // Clean up empty chats when component unmounts
  useEffect(() => {
    return () => {
      clearEmptyChats();
    };
  }, []);

  return {
    chats,
    activeChat,
    setActiveChat,
    createChat,
    updateChat,
    updateChatMode,
    deleteChat,
  };
};