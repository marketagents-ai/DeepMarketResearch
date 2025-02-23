import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { FileUpload } from './FileUpload/FileUpload';

interface ChatInputProps {
  onSubmit: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, isLoading, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-gray-700 p-4 bg-gray-800">
      <div className="flex gap-2 items-center">
        <FileUpload />
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={disabled ? 'Create a new chat to start' : 'Type your research query...'}
          className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading || disabled}
        />
        <button
          type="submit"
          disabled={isLoading || disabled || !message.trim()}
          className="bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          <Send size={20} />
        </button>
      </div>
    </form>
  );
};