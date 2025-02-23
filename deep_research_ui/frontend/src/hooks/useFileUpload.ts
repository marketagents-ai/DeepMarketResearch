import { useState, useCallback } from 'react';

interface UseFileUploadProps {
  onDataProcessed: (file: File) => void;
  allowedTypes: string[];
}

export const useFileUpload = ({ onDataProcessed, allowedTypes }: UseFileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;
    if (!allowedTypes.includes(fileExtension)) {
      alert(`Unsupported file type. Please use: ${allowedTypes.join(', ')}`);
      return;
    }

    try {
      onDataProcessed(file);
    } catch (error) {
      console.error('Error handling file:', error);
      alert('Error handling file. Please try again.');
    }
  };

  const dragProps = {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(true);
    },
    onDragEnter: (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(true);
    },
    onDragLeave: (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
    },
    onDrop: async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      
      const file = e.dataTransfer.files[0];
      if (!file) return;

      const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;
      if (!allowedTypes.includes(fileExtension)) {
        alert(`Unsupported file type. Please use: ${allowedTypes.join(', ')}`);
        return;
      }

      try {
        onDataProcessed(file);
      } catch (error) {
        console.error('Error handling file:', error);
        alert('Error handling file. Please try again.');
      }
    }
  };

  return {
    handleFileChange,
    isDragging,
    dragProps
  };
};