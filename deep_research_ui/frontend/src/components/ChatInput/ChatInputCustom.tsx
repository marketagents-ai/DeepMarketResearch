// src/components/ChatInput/ChatInputCustom.tsx

import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { FileUpload } from '../FileUpload/FileUpload';

interface ChatInputCustomProps {
  onSubmit: (message: string, file?: File) => void;
  isLoading?: boolean;
}

export const ChatInputCustom: React.FC<ChatInputCustomProps> = ({
  onSubmit,
  isLoading
}) => {
  const [message, setMessage] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('ChatInputCustom handleSubmit called'); // Debug log
    if (message.trim() || file) {
      console.log('Submitting message:', message); // Debug log
      onSubmit(message, file || undefined);
      setMessage('');
      setFile(null);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <FileUpload onDataProcessed={(uploadedFile) => setFile(uploadedFile)} />
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        className="flex-1 bg-gray-800 rounded-lg px-4 py-2 text-white"
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={isLoading || (!message.trim() && !file)}
        className="p-2 rounded-lg bg-blue-600 text-white disabled:opacity-50"
      >
        <Send size={20} />
      </button>
    </form>
  );
};