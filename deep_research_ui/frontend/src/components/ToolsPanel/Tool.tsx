import React from 'react';
import type { SystemTool } from '../../types/tools';

interface ToolProps {
  tool: SystemTool;
}

export const Tool: React.FC<ToolProps> = ({ tool }) => {
  const { name, description, icon: Icon } = tool;
  
  return (
    <button className="w-full text-left p-3 rounded-lg hover:bg-gray-700 transition-colors group">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-md bg-gray-700 group-hover:bg-gray-600 transition-colors">
          <Icon size={18} className="text-blue-400" />
        </div>
        <div>
          <h3 className="font-medium text-sm">{name}</h3>
          <p className="text-xs text-gray-400">{description}</p>
        </div>
      </div>
    </button>
  );
};