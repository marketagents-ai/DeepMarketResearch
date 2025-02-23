import React from 'react';

interface TabSelectorProps {
  activeTab: 'system' | 'tools';
  onChange: (tab: 'system' | 'tools') => void;
}

export const TabSelector: React.FC<TabSelectorProps> = ({ activeTab, onChange }) => {
  return (
    <div className="bg-gray-800/50 rounded-full p-1 flex">
      <button
        className={`flex-1 px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
          activeTab === 'system'
            ? 'bg-gray-700 text-white'
            : 'text-gray-400 hover:text-white'
        }`}
        onClick={() => onChange('system')}
      >
        System
      </button>
      <button
        className={`flex-1 px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
          activeTab === 'tools'
            ? 'bg-gray-700 text-white'
            : 'text-gray-400 hover:text-white'
        }`}
        onClick={() => onChange('tools')}
      >
        Tools
      </button>
    </div>
  );
};
