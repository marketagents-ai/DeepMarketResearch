import React, { useState, useEffect } from 'react';
import { Plus, Settings2, Trash2, Edit2, Eye, X } from 'lucide-react';
import { Switch } from '../Switch';
import { SchemaBuilder } from './SchemaBuilder';
import type { CustomTool } from '../../../types/tools';

export const CustomToolBuilder: React.FC = () => {
    const [customTools, setCustomTools] = useState<CustomTool[]>([]);
    const [isBuilding, setIsBuilding] = useState(false);
    const [editingTool, setEditingTool] = useState<CustomTool | null>(null);
    const [viewingTool, setViewingTool] = useState<CustomTool | null>(null);

    useEffect(() => {
        loadCustomTools();
        window.addEventListener('customToolsUpdated', loadCustomTools);
        return () => window.removeEventListener('customToolsUpdated', loadCustomTools);
    }, []);

    const loadCustomTools = () => {
        const savedTools = localStorage.getItem('customTools');
        if (savedTools) {
            setCustomTools(JSON.parse(savedTools));
        }
    };
    useEffect(() => {
      const savedTools = localStorage.getItem('customTools');
      console.log('Current tools in localStorage:', savedTools);
  }, [customTools]); 

  const handleToggle = (toolId: string, enabled: boolean) => {
    console.log('Toggling tool:', toolId, 'to:', enabled);
    const updatedTools = customTools.map(tool => 
      tool.id === toolId ? { ...tool, enabled } : tool
    );
    console.log('Updated tools after toggle:', updatedTools);
    localStorage.setItem('customTools', JSON.stringify(updatedTools));
    setCustomTools(updatedTools);
    
    // Add this debug log
    const enabledTools = updatedTools.filter(t => t.enabled);
    console.log('Currently enabled tools:', enabledTools);
    
    window.dispatchEvent(new Event('customToolsUpdated'));
  };

    const handleDelete = (toolId: string) => {
        if (window.confirm('Are you sure you want to delete this tool?')) {
            const updatedTools = customTools.filter(tool => tool.id !== toolId);
            localStorage.setItem('customTools', JSON.stringify(updatedTools));
            setCustomTools(updatedTools);
            window.dispatchEvent(new Event('customToolsUpdated'));
        }
    };

    const handleSaveTool = (tool: CustomTool) => {
      let updatedTools;
      if (editingTool) {
          updatedTools = customTools.map(t => 
              t.id === editingTool.id ? { ...tool, id: editingTool.id, enabled: editingTool.enabled } : t
          );
      } else {
          updatedTools = [...customTools, { 
              ...tool, 
              id: Date.now().toString(),
              enabled: false,
              schema: tool.schema // Ensure schema is included
          }];
      }
      
      localStorage.setItem('customTools', JSON.stringify(updatedTools));
      setCustomTools(updatedTools);
      setIsBuilding(false);
      setEditingTool(null);
      window.dispatchEvent(new Event('customToolsUpdated'));
  };
    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Settings2 size={18} className="text-gray-400" />
                    <h3 className="font-medium">Custom Tools</h3>
                </div>
                <button
                    onClick={() => {
                        setEditingTool(null);
                        setIsBuilding(true);
                    }}
                    className="p-1.5 rounded-md hover:bg-gray-700 text-gray-400 hover:text-white"
                >
                    <Plus size={18} />
                </button>
            </div>

            {/* Tool Builder/Editor */}
            {(isBuilding || editingTool) && (
                <SchemaBuilder
                    initialTool={editingTool}
                    onCancel={() => {
                        setIsBuilding(false);
                        setEditingTool(null);
                    }}
                    onSave={handleSaveTool}
                />
            )}

            {/* Tool List */}
            <div className="space-y-2">
                {customTools.map((tool) => (
                    <div
                        key={tool.id}
                        className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50 hover:bg-gray-800"
                    >
                        <div className="flex-1">
                            <h4 className="text-sm font-medium">{tool.name}</h4>
                            <p className="text-xs text-gray-400">{tool.description}</p>
                            <div className="flex gap-2 mt-1">
                                {Object.keys(tool.schema?.properties || {}).map((fieldName) => (
                                    <span key={fieldName} className="text-xs bg-gray-700 px-2 py-0.5 rounded">
                                        {fieldName}
                                    </span>
                                ))}
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => setViewingTool(tool)}
                                className="p-1.5 rounded hover:bg-gray-700 text-blue-400"
                                title="View details"
                            >
                                <Eye size={16} />
                            </button>
                            <button
                                onClick={() => {
                                    setEditingTool(tool);
                                    setIsBuilding(true);
                                }}
                                className="p-1.5 rounded hover:bg-gray-700 text-green-400"
                                title="Edit tool"
                            >
                                <Edit2 size={16} />
                            </button>
                            <Switch
                                checked={tool.enabled || false}
                                onCheckedChange={(checked) => handleToggle(tool.id, checked)}
                                className="data-[state=checked]:bg-blue-600"
                            />
                            <button
                                onClick={() => handleDelete(tool.id)}
                                className="p-1.5 rounded hover:bg-gray-700 text-red-400"
                                title="Delete tool"
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {/* View Tool Modal */}
            {viewingTool && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
                    <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-medium">{viewingTool.name}</h3>
                            <button
                                onClick={() => setViewingTool(null)}
                                className="text-gray-400 hover:text-gray-300"
                            >
                                <X size={20} />
                            </button>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <h4 className="text-sm font-medium text-gray-400">Description</h4>
                                <p className="text-sm mt-1">{viewingTool.description}</p>
                            </div>
                            <div>
                                <h4 className="text-sm font-medium text-gray-400 mb-2">Schema Fields</h4>
                                <div className="space-y-2">
                                    {viewingTool.schema?.properties && 
                                        Object.entries(viewingTool.schema.properties).map(([name, field]: [string, any]) => (
                                            <div key={name} className="bg-gray-700 rounded p-2">
                                                <div className="flex justify-between">
                                                    <span className="font-medium">{name}</span>
                                                    <span className="text-sm text-gray-400">{field.type}</span>
                                                </div>
                                            </div>
                                        ))
                                    }
                                </div>
                            </div>
                        </div>
                        <div className="mt-6 flex justify-end">
                            <button
                                onClick={() => setViewingTool(null)}
                                className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};