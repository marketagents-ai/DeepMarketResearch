import React, { useState } from 'react';
import { Plus, Pencil, Trash2 } from 'lucide-react';
import { SystemMessageForm } from './SystemMessageForm';
import { useSystem } from '../../../hooks/useSystem';
import type { SystemMessage } from '../../../types/system';

export const SystemTab: React.FC = () => {
  const { 
    messages, 
    activeMessageId,
    addMessage, 
    updateMessage, 
    deleteMessage,
    setActiveMessage 
  } = useSystem();
  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">System Messages</h3>
        <button
          onClick={() => setIsAdding(true)}
          className="p-1.5 rounded-md hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
        >
          <Plus size={18} />
        </button>
      </div>

      {isAdding && (
        <SystemMessageForm
          onSubmit={(data) => {
            addMessage(data);
            setIsAdding(false);
          }}
          onCancel={() => setIsAdding(false)}
        />
      )}

      <div className="space-y-2">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`p-3 rounded-lg transition-colors ${
              message.id === activeMessageId
                ? 'bg-gray-700'
                : 'bg-gray-800/50 hover:bg-gray-800'
            }`}
          >
            {editingId === message.id ? (
              <SystemMessageForm
                onSubmit={(data) => {
                  updateMessage(message.id, data.content);
                  setEditingId(null);
                }}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <div>
                <div className="flex items-center justify-between mb-1">
                  <div>
                    <h4 className="font-medium text-sm">{message.name}</h4>
                    <span className="text-xs text-gray-400">
                      {message.mode === 'research' ? 'Research Mode' : 'Custom Mode'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setEditingId(message.id)}
                      className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white"
                    >
                      <Pencil size={14} />
                    </button>
                    <button
                      onClick={() => deleteMessage(message.id)}
                      className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-300 line-clamp-2">{message.content}</p>
                <button
                  onClick={() => setActiveMessage(
                    message.id === activeMessageId ? null : message.id
                  )}
                  className="mt-2 text-xs text-blue-400 hover:text-blue-300"
                >
                  {message.id === activeMessageId ? 'Deactivate' : 'Activate'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

