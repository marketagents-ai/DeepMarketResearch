import { useState, useEffect } from 'react';
import type { SystemMessage, SystemState } from '../types/system';

export const useSystem = () => {
  const [state, setState] = useState<SystemState>(() => {
    const saved = localStorage.getItem('system-state');
    return saved ? JSON.parse(saved) : {
      messages: [],
      activeMessageId: null
    };
  });

  useEffect(() => {
    localStorage.setItem('system-state', JSON.stringify(state));
  }, [state]);

  const addMessage = (message: Omit<SystemMessage, 'id' | 'createdAt'>) => {
    const newMessage: SystemMessage = {
      ...message,
      id: crypto.randomUUID(),
      createdAt: new Date()
    };
    setState(prev => ({
      ...prev,
      messages: [...prev.messages, newMessage]
    }));
  };

  const updateMessage = (id: string, content: string) => {
    setState(prev => ({
      ...prev,
      messages: prev.messages.map(msg => 
        msg.id === id ? { ...msg, content } : msg
      )
    }));
  };

  const deleteMessage = (id: string) => {
    setState(prev => ({
      ...prev,
      messages: prev.messages.filter(msg => msg.id !== id),
      activeMessageId: prev.activeMessageId === id ? null : prev.activeMessageId
    }));
  };

  const setActiveMessage = (id: string | null) => {
    setState(prev => ({
      ...prev,
      activeMessageId: id
    }));
  };

  return {
    messages: state.messages,
    activeMessageId: state.activeMessageId,
    addMessage,
    updateMessage,
    deleteMessage,
    setActiveMessage
  };
};