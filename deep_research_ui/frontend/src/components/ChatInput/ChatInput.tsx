import React, { useState, useRef } from 'react';
import { Send, Upload } from 'lucide-react';

interface ChatInputProps {
  mode: 'custom' | 'research';
  onSubmit: (message: string, files?: File[]) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  mode,
  onSubmit,
  isLoading = false,
  disabled = false
}) => {
  const [message, setMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() || selectedFiles.length > 0) {
      onSubmit(message, selectedFiles);
      setMessage('');
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        multiple
      />
      <button
        type="button"
        onClick={handleUploadClick}
        className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        disabled={isLoading || disabled}
      >
        <Upload className="w-5 h-5 text-gray-400" />
      </button>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={isLoading || disabled}
      />
      <button
        type="submit"
        className="p-2 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
        disabled={isLoading || disabled || (!message.trim() && !selectedFiles.length)}
      >
        <Send className="w-5 h-5 text-blue-500" />
      </button>
    </form>
  );
};