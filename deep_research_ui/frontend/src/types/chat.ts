import type { ResearchData } from './research';

export type ChatMode = 'custom' | 'research';


export interface ChatItem {
  isUser: boolean;
  content: string | any; // or more specific type depending on your data
  timestamp: Date;
  mode: ChatMode;
  customSchemas?: any[]; // Add this if you're using customSchemas
  error?: boolean;
}

export interface Chat {
  id: string;
  title: string;
  messages: ChatItem[];
  mode: ChatMode;
  createdAt: Date;
  updatedAt: Date;
}