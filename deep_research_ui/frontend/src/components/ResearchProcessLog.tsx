import React from 'react';
import { Loader } from 'lucide-react';

interface ResearchProcessLogProps {
  logs: string[];
}

export const ResearchProcessLog: React.FC<ResearchProcessLogProps> = ({ logs }) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
        <div className="flex items-center gap-3 mb-4">
          <Loader className="w-6 h-6 animate-spin text-blue-500" />
          <h3 className="text-lg font-medium text-gray-200">Processing Research</h3>
        </div>
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {logs.map((log, index) => (
            <div
              key={index}
              className="flex items-start gap-2 text-sm text-gray-300 animate-fadeIn"
            >
              <span className="flex-shrink-0">{getLogIcon(log)}</span>
              <span>{log}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const getLogIcon = (log: string) => {
  if (log.includes('❌')) return '❌';
  if (log.includes('✅')) return '✅';
  if (log.includes('🚀')) return '🚀';
  if (log.includes('📊')) return '📊';
  if (log.includes('🔍')) return '🔍';
  return '▶️';
};