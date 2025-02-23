import React from 'react';
import { useResearch } from '../../context/ResearchContext';

interface LogsContainerProps {
  chatId: string;
}

export const LogsContainer: React.FC<LogsContainerProps> = ({ chatId }) => {
  const { logs } = useResearch();
  const currentLogs = logs[chatId] || [];

  return (
    <div className="logs-container">
      {currentLogs.map((log, index) => (
        <div key={`${chatId}-${index}`} className="log-entry">
          {log}
        </div>
      ))}
    </div>
  );
};