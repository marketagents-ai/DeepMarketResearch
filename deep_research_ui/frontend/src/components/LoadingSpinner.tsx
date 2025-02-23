import React from 'react';
import { Loader } from 'lucide-react';

export const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center">
      <div className="animate-spin">
        <Loader size={32} className="text-blue-500" />
      </div>
    </div>
  );
};