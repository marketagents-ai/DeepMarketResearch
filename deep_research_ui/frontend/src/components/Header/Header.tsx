import React from 'react';
import { LayoutDashboard } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="border-b border-gray-700 p-4 bg-gray-800/50 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <LayoutDashboard size={24} className="text-blue-500" />
        <h1 className="text-xl font-bold">Dashboard</h1>
      </div>
    </header>
  );
};