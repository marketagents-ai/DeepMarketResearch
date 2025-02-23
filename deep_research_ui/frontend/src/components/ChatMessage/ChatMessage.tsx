// src/components/ChatMessage/ChatMessage.tsx

import React from 'react';
import { Avatar } from '../Avatar/Avatar';
import { ResearchResult } from './ResearchResult';
import type { ResearchData } from '../../types/research';

interface ChatMessageProps {
  isUser: boolean;
  content: string | ResearchData[];
  timestamp: Date;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  isUser,
  content,
  timestamp
}) => {
  // Add debug logging
  console.log('ChatMessage content:', content);

  // Check if content is research data
  const isResearchData = Array.isArray(content) && content.length > 0 && 'url' in content[0];

  if (isResearchData) {
    return (
      <div className="flex gap-3">
        <Avatar isUser={isUser} />
        <div className="flex-1">
          <ResearchResult data={content as ResearchData[]} />
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <Avatar isUser={isUser} />
      <div className={`p-3 rounded-lg max-w-[80%] ${
        isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
      }`}>
        {content as string}
      </div>
    </div>
  );
};

// import React from 'react';
// import { Avatar } from '../Avatar/Avatar';
// import { ResearchResult } from './ResearchResult';
// import type { ResearchData } from '../../types/research';

// interface ChatMessageProps {
//   isUser?: boolean;
//   content: string | ResearchData[];
// }

// export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser = false, content }) => {
//   // Clean and format the content if it's a string containing JSON
//   const formatContent = (rawContent: string | ResearchData[]) => {
//     if (typeof rawContent === 'string') {
//       try {
//         // Check if the string is JSON
//         if (rawContent.trim().startsWith('{')) {
//           const parsed = JSON.parse(rawContent);
//           if (parsed.content) {
//             return parsed.content;
//           }
//         }
//         return rawContent;
//       } catch {
//         // If parsing fails, return original content
//         return rawContent;
//       }
//     }
//     return rawContent;
//   };

//   const cleanContent = formatContent(content);

//   // Handle research data array
//   if (Array.isArray(cleanContent) && cleanContent.length > 0 && 'url' in cleanContent[0]) {
//     return <ResearchResult data={cleanContent as ResearchData[]} />;
//   }

//   // Handle regular chat message
//   return (
//     <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
//       <Avatar isUser={isUser} />
//       <div className={`p-3 rounded-lg max-w-[80%] ${
//         isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
//       }`}>
//         {typeof content === 'string' ? content : JSON.stringify(content)}
//       </div>
//     </div>
//   );
// };

// // import React from 'react';
// // import { Avatar } from '../Avatar/Avatar';
// // import { ResearchResult } from './ResearchResult';
// // import type { ResearchData } from '../../types/research';

// // interface ChatMessageProps {
// //   isUser?: boolean;
// //   content: string | ResearchData[];
// // }

// // export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser = false, content }) => {
// //   const messageClass = isUser 
// //     ? 'ml-auto flex-row-reverse' 
// //     : 'flex-row';

// //   return (
// //     <div className={`flex items-start gap-3 mb-4 ${messageClass}`}>
// //       <Avatar isUser={isUser} />
      
// //       <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : 'text-left'}`}>
// //         {typeof content === 'string' ? (
// //           <div className={`rounded-lg p-3 inline-block ${
// //             isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
// //           }`}>
// //             <p>{content}</p>
// //           </div>
// //         ) : (
// //           <div className="space-y-4">
// //             {content.map((result, index) => (
// //               <ResearchResult key={`${result.agent_id}-${index}`} data={result} />
// //             ))}
// //           </div>
// //         )}
// //       </div>
// //     </div>
// //   );
// // };

// // import React from 'react';
// // import type { ResearchData } from '../types/research';
// // import { ResearchResult } from './ChatMessage/ResearchResult';

// // interface ChatMessageProps {
// //   isUser?: boolean;
// //   content: string | ResearchData | ResearchData[];
// // }

// // export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser, content }) => {
// //   const messageClass = isUser 
// //     ? 'bg-blue-600 text-white ml-auto' 
// //     : 'bg-gray-700 text-white';

// //   // Handle string content (including error messages)
// //   if (typeof content === 'string') {
// //     return (
// //       <div className={`max-w-[80%] rounded-lg p-3 mb-4 ${messageClass}`}>
// //         <p>{content}</p>
// //       </div>
// //     );
// //   }

// //   // Handle array of research data
// //   if (Array.isArray(content)) {
// //     return (
// //       <div className="space-y-4">
// //         {content.map((item, index) => (
// //           <ResearchResult key={index} data={item} />
// //         ))}
// //       </div>
// //     );
// //   }

// //   // Handle single research data object
// //   return <ResearchResult data={content} />;
// // };



// import React from 'react';
// import { Avatar } from '../Avatar/Avatar';
// import { ResearchResult } from './ResearchResult';
// import type { ResearchData } from '../../types/research';

// interface ChatMessageProps {
//   isUser?: boolean;
//   content: string | ResearchData[];
// }

// export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser = false, content }) => {
//   // Handle array of research data
//   if (Array.isArray(content)) {
//     return (
//       <div className="flex gap-3">
//         <Avatar isUser={isUser} />
//         <div className="flex-1">
//           <ResearchResult data={content} />
//         </div>
//       </div>
//     );
//   }

//   // Handle string content (user messages)
//   return (
//     <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
//       <Avatar isUser={isUser} />
//       <div className={`p-3 rounded-lg max-w-[80%] ${
//         isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
//       }`}>
//         {content}
//       </div>
//     </div>
//   );
// };