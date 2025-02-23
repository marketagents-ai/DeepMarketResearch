// import { toast } from 'react-hot-toast';

export class APIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'APIError';
  }
}

export const handleAPIError = (error: unknown): string => {
  console.error('API Error:', error);
  if (error instanceof APIError) {
    return error.message;
  }
  return 'An unexpected error occurred. Please try again.';
};
export const toggleCustomToolsAPI = async (enabled: boolean): Promise<void> => {
  try {
    const response = await fetch('http://localhost:8000/api/tools/toggle', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ enabled }),
    });

    if (!response.ok) {
      throw new APIError('Failed to toggle custom tools', response.status);
    }
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('Network error while toggling custom tools');
  }
};

// import { APIError } from '../utils/api';
import type { ResearchData } from '../types/research';
import type { Chat, ChatItem, ChatMode } from '../types/chat';
import type { SystemMessage } from '../types/system';
import type { CustomTool } from '../types/tools';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Chat related API calls
export const fetchChats = async (): Promise<Chat[]> => {
  try {
    const response = await fetch(`${API_URL}/api/chats`);
    if (!response.ok) {
      throw new APIError('Failed to fetch chats', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const createChat = async (mode: ChatMode): Promise<Chat> => {
  try {
    const response = await fetch(`${API_URL}/api/chats`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      throw new APIError('Failed to create chat', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const updateChat = async (chatId: string, messages: ChatItem[]): Promise<Chat> => {
  try {
    const response = await fetch(`${API_URL}/api/chats/${chatId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });
    if (!response.ok) {
      throw new APIError('Failed to update chat', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const deleteChat = async (chatId: string): Promise<void> => {
  try {
    const response = await fetch(`${API_URL}/api/chats/${chatId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new APIError('Failed to delete chat', response.status);
    }
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};
const debugApiCall = (endpoint: string, method: string) => {
  console.trace(`API Call: ${method} ${endpoint}`);
};

export const fetchResearch = async (query: string, urls?: string[]): Promise<ResearchData[]> => {
  try {
    // Log the request for debugging
    console.log('Sending research request:', {
      url: `${API_URL}/api/research`,
      query,
      urls
    });

    const response = await fetch(`${API_URL}/api/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        urls: urls || []
      }),
    });

    if (!response.ok) {
      throw new APIError('Failed to fetch research data', response.status);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Research API Error:', error);
    throw error instanceof APIError ? error : new APIError('Failed to connect to research service');
  }
};

// System messages related API calls
export const fetchSystemMessages = async (): Promise<SystemMessage[]> => {
  try {
    const response = await fetch(`${API_URL}/api/system/messages`);
    if (!response.ok) {
      throw new APIError('Failed to fetch system messages', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const updateSystemMessage = async (messageId: string, content: string): Promise<SystemMessage> => {
  try {
    const response = await fetch(`${API_URL}/api/system/messages/${messageId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });
    if (!response.ok) {
      throw new APIError('Failed to update system message', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

// Custom tools related API calls
export const fetchCustomTools = async (): Promise<CustomTool[]> => {
  try {
    const response = await fetch(`${API_URL}/api/tools/custom`);
    if (!response.ok) {
      throw new APIError('Failed to fetch custom tools', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const createCustomTool = async (tool: Omit<CustomTool, 'id'>): Promise<CustomTool> => {
  try {
    const response = await fetch(`${API_URL}/api/tools/custom`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tool),
    });
    if (!response.ok) {
      throw new APIError('Failed to create custom tool', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const updateCustomTool = async (toolId: string, updates: Partial<CustomTool>): Promise<CustomTool> => {
  try {
    const response = await fetch(`${API_URL}/api/tools/custom/${toolId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });
    if (!response.ok) {
      throw new APIError('Failed to update custom tool', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const deleteCustomTool = async (toolId: string): Promise<void> => {
  try {
    const response = await fetch(`${API_URL}/api/tools/custom/${toolId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new APIError('Failed to delete custom tool', response.status);
    }
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('An unexpected error occurred');
  }
};

export const sendCustomMessage = async (formData: FormData): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/custom`, {
      method: 'POST',
      body: formData, // FormData handles content-type automatically
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to send message' }));
      throw new APIError(errorData.detail || `Error: ${response.status}`, response.status);
    }

    return await response.json();
  } catch (error) {
    console.error('Custom API Error:', error);
    throw error instanceof APIError ? error : new APIError('Failed to send message');
  }
};