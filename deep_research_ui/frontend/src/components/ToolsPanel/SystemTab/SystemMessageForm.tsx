import React, { useState } from 'react';
import { X } from 'lucide-react';
import type { ChatMode } from '../../../types/chat';

interface SystemMessageFormProps {
  onSubmit: (data: { name: string; content: string; mode: ChatMode }) => void;
  onCancel: () => void;
}

export const SystemMessageForm: React.FC<SystemMessageFormProps> = ({ onSubmit, onCancel }) => {
  const [name, setName] = useState('');
  const [content, setContent] = useState('');
  const [mode, setMode] = useState<ChatMode>('research');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim() && content.trim()) {
      onSubmit({ name, content, mode });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-4">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            placeholder="E.g., Research Assistant"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as ChatMode)}
            className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
          >
            <option value="research">Research Mode</option>
            <option value="custom">Custom Mode</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">System Message</label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="w-full bg-gray-700 rounded px-3 py-2 text-sm h-32 resize-none"
            placeholder="Enter the system message that defines the AI assistant's behavior..."
          />
        </div>

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 text-sm hover:bg-gray-700 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!name.trim() || !content.trim()}
            className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
          >
            Save Message
          </button>
        </div>
      </div>
    </form>
  );
};