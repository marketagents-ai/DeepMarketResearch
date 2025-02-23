import { useState, useCallback } from 'react';
import { parseFile } from '../utils/fileParser';

interface UseFileUploadProps {
  onDataProcessed: (data: any[]) => void;
  allowedTypes: string[];
}

export const useFileUpload = ({ onDataProcessed, allowedTypes }: UseFileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
      alert('Unsupported file type. Please upload CSV, XLSX, or JSON files.');
      return;
    }

    try {
      const data = await parseFile(file);
      onDataProcessed(data);
    } catch (error) {
      console.error('Error parsing file:', error);
      alert('Error processing file. Please try again.');
    }
  };

  const dragProps = {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(true);
    },
    onDragLeave: () => setIsDragging(false),
    onDrop: async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      
      const file = e.dataTransfer.files[0];
      if (!file) return;

      if (!allowedTypes.includes(file.type)) {
        alert('Unsupported file type. Please upload CSV, XLSX, or JSON files.');
        return;
      }

      try {
        const data = await parseFile(file);
        onDataProcessed(data);
      } catch (error) {
        console.error('Error parsing file:', error);
        alert('Error processing file. Please try again.');
      }
    }
  };

  return {
    handleFileChange,
    isDragging,
    dragProps
  };
};