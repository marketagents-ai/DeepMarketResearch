// import React from 'react';
// import { Bot, User } from 'lucide-react';

// interface AvatarProps {
//   isUser: boolean;
// }

// export const Avatar: React.FC<AvatarProps> = ({ isUser }) => {
//   return (
//     <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
//       isUser ? 'bg-blue-600' : 'bg-purple-600'
//     }`}>
//       {isUser ? (
//         <User size={18} className="text-white" />
//       ) : (
//         <Bot size={18} className="text-white" />
//       )}
//     </div>
//   );
// };


import React from 'react';
import { Bot, User } from 'lucide-react';

interface AvatarProps {
  isUser: boolean;
}

export const Avatar: React.FC<AvatarProps> = ({ isUser }) => {
  return (
    <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
      isUser ? 'bg-blue-600' : 'bg-purple-600'
    }`}>
      {isUser ? (
        <User size={18} className="text-white" />
      ) : (
        <Bot size={18} className="text-white" />
      )}
    </div>
  );
};