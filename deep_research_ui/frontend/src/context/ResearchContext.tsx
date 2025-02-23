import React, { createContext, useContext, useState, useEffect } from 'react';

interface ResearchContextType {
  logs: string[];
}

const ResearchContext = createContext<ResearchContextType>({ logs: [] });

export const useResearch = () => useContext(ResearchContext);

export const ResearchProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    const eventSource = new EventSource('http://localhost:8000/api/research/logs');
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.message) {
          setLogs(prev => [...prev, data.message]);
        }
      } catch (error) {
        console.error('Error parsing log message:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource failed:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <ResearchContext.Provider value={{ logs }}>
      {children}
    </ResearchContext.Provider>
  );
};