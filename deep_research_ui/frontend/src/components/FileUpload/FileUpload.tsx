import React from 'react';
import { Upload } from 'lucide-react';
import { useFileUpload } from '../../hooks/useFileUpload';

interface FileUploadProps {
  onDataReceived: (file: File) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onDataReceived }) => {
  const { handleFileChange, isDragging, dragProps } = useFileUpload({
    onDataProcessed: onDataReceived,
    allowedTypes: ['.csv', '.json', '.xlsx', '.xls']
  });

  return (
    <div className="relative">
      <label
        {...dragProps}
        className={`
          flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
          ${isDragging 
            ? 'bg-blue-600 bg-opacity-20 border-2 border-blue-500' 
            : 'bg-gray-700 hover:bg-gray-600'
          }
          transition-colors
        `}
      >
        <input
          type="file"
          onChange={handleFileChange}
          accept=".csv,.json,.xlsx,.xls"
          className="hidden"
        />
        <Upload size={18} className="text-blue-400" />
        <span className="text-sm">Upload Data</span>
      </label>
    </div>
  );
};