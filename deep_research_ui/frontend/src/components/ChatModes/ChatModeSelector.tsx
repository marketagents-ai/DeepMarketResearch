import React from 'react';
import { Database, Search } from 'lucide-react';
import type { ChatMode } from '../../types/chat';

interface ChatModeSelectorProps {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  disabled?: boolean;
}

export const ChatModeSelector: React.FC<ChatModeSelectorProps> = ({ 
  mode, 
  onModeChange,
  disabled = false 
}) => {
  return (
    <div className="flex gap-2">
      <button
        onClick={() => onModeChange('custom')}
        disabled={disabled}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg transition-colors
          ${mode === 'custom' 
            ? 'bg-blue-600 text-white' 
            : 'bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-800'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <Database size={18} />
        Custom Mode
      </button>

      <button
        onClick={() => onModeChange('research')}
        disabled={disabled}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg transition-colors
          ${mode === 'research' 
            ? 'bg-blue-600 text-white' 
            : 'bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-800'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <Search size={18} />
        Research Mode
      </button>
    </div>
  );
};


// import React from 'react';
// import { Database, Search } from 'lucide-react';
// import { ChatMode } from '../../types/chat';

// interface ChatModeSelectorProps {
//   mode: ChatMode;
//   onModeChange: (mode: ChatMode) => void;
// }

// export const ChatModeSelector: React.FC<ChatModeSelectorProps> = ({ mode, onModeChange }) => {
//   return (
//     <div className="bg-[#1a1b1e] p-1 rounded-full inline-flex">
//       <button
//         onClick={() => onModeChange('custom')}
//         className={`flex items-center gap-2 px-4 py-2 rounded-full transition-colors ${
//           mode === 'custom'
//             ? 'bg-gray-800 text-white'
//             : 'text-gray-400 hover:text-white'
//         }`}
//       >
//         <Database size={18} />
//         Custom Mode
//       </button>
//       <button
//         onClick={() => onModeChange('research')}
//         className={`flex items-center gap-2 px-4 py-2 rounded-full transition-colors ${
//           mode === 'research'
//             ? 'bg-gray-800 text-white'
//             : 'text-gray-400 hover:text-white'
//         }`}
//       >
//         <Search size={18} />
//         Research Mode
//       </button>
//     </div>
//   );
// };