import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 border-b border-gray-700 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl font-semibold text-white">Market Oracle</span>
        </div>
      </div>
    </header>
  );
};