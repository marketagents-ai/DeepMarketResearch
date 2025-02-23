import React, { useState } from 'react';
import { Settings, Plus } from 'lucide-react';
import { TabSelector } from './TabSelector';
import { ToolsList } from './ToolsList';
import { SystemTab } from './SystemTab/SystemTab';
import { CustomToolBuilder } from './CustomTools/CustomToolBuilder';
import { useCustomTools } from '../../hooks/useCustomTools';

export const ToolsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'system' | 'tools'>('system');
  const [isAddingTool, setIsAddingTool] = useState(false);
  const { isEnabled, toggleCustomTools } = useCustomTools(); // Add just the toggle functionality

  return (
    <div className="w-80 border-l border-gray-800 bg-[#1a1b1e] flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <TabSelector activeTab={activeTab} onChange={setActiveTab} />
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeTab === 'system' ? (
          <SystemTab />
        ) : (
          <div className="p-4">
            {/* Custom Tools Section at Top */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium">Custom Tools</h2>
                <button
                  className="p-2 hover:bg-gray-700 rounded-lg"
                  onClick={() => setIsAddingTool(true)}
                >
                  <Plus size={20} />
                </button>
              </div>
              <CustomToolBuilder />
            </div>

            {/* System Tools Sections at Bottom */}
            <div className="space-y-6">
              <div>
                <h2 className="text-lg font-medium mb-4">Executable Tools</h2>
                <ToolsList type="executable" />
              </div>
              
              <div>
                <h2 className="text-lg font-medium mb-4">Typed Tools</h2>
                <ToolsList type="typed" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};