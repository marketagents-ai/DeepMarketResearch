import { useState, useEffect } from 'react';
import type { CustomTool } from '../types/tools';

export const useCustomTools = () => {
  const [tools, setTools] = useState<CustomTool[]>(() => {
    const savedTools = localStorage.getItem('customTools');
    console.log('Initial tools from localStorage:', savedTools);
    return savedTools ? JSON.parse(savedTools) : [];
  });
  
  const [isBuilding, setIsBuilding] = useState(false);
  const [customTools, setCustomTools] = useState<CustomTool[]>([]);

  useEffect(() => {
    console.log('Current tools state:', tools);
    console.log('Enabled tools:', tools.filter(tool => tool.enabled));
    localStorage.setItem('customTools', JSON.stringify(tools));
  }, [tools]);
  
  const startBuilding = () => setIsBuilding(true);
  const cancelBuilding = () => setIsBuilding(false);
  
  const [isEnabled, setIsEnabled] = useState(() => {
    const savedState = localStorage.getItem('customToolsEnabled');
    return savedState ? JSON.parse(savedState) : false;
  });

  const handleToggle = (toolId: string, enabled: boolean) => {
    console.log('Toggling tool:', toolId, 'to:', enabled);
    const updatedTools = tools.map(tool => 
      tool.id === toolId ? { ...tool, enabled } : tool
    );
    console.log('Updated tools:', updatedTools);
    setTools(updatedTools);
    localStorage.setItem('customTools', JSON.stringify(updatedTools));
    window.dispatchEvent(new Event('customToolsUpdated'));
  };

  const saveTool = (tool: CustomTool) => {
    const newTool = {
      ...tool,
      enabled: false,
      id: Date.now().toString()
    };
    const updatedTools = [...tools, newTool];
    setTools(updatedTools);
    setIsBuilding(false);
    localStorage.setItem('customTools', JSON.stringify(updatedTools));
    window.dispatchEvent(new Event('customToolsUpdated'));
  };

  const deleteTool = async (toolId: string) => {
    try {
      const updatedTools = tools.filter(tool => tool.id !== toolId);
      setTools(updatedTools);
      localStorage.setItem('customTools', JSON.stringify(updatedTools));
      window.dispatchEvent(new Event('customToolsUpdated'));
    } catch (error) {
      console.error('Error deleting tool:', error);
    }
  };

  return {
    tools,
    isBuilding,
    startBuilding,
    cancelBuilding,
    saveTool,
    deleteTool,
    isEnabled,
    setIsEnabled,
    handleToggle
  };
};