import React, { useState, FormEvent } from 'react';

interface ChatInputResearchProps {
  onSubmit: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export const ChatInputResearch: React.FC<ChatInputResearchProps> = ({
  onSubmit,
  isLoading = false,
  disabled = false
}) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message.trim());
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-3">
      <button
        type="button"
        className="p-2 text-gray-400 hover:text-gray-300"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </button>
      
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        disabled={isLoading || disabled}
        className="flex-1 bg-[#1e2430] text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-1 focus:ring-[#63e2ff] placeholder-gray-500"
      />
      
      <button
        type="submit"
        disabled={!message.trim() || isLoading || disabled}
        className="p-2 text-[#63e2ff] hover:text-[#63e2ff]/80 disabled:opacity-50"
      >
        <svg className="w-5 h-5 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
        </svg>
      </button>
    </form>
  );
};