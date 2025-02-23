import React from 'react';
import { Avatar } from './Avatar/Avatar';
import { ResearchResult } from './ChatMessage/ResearchResult';
import type { ResearchData } from './../types/research';

interface ChatMessageProps {
  isUser?: boolean;
  content: string | ResearchData[];
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser = false, content }) => {
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
        {typeof content === 'string' ? content : JSON.stringify(content)}
      </div>
    </div>
  );
};
// export const ChatMessage: React.FC<ChatMessageProps> = ({ isUser, content }) => {
//   const messageClass = isUser 
//     ? 'bg-blue-600 text-white ml-auto' 
//     : 'bg-gray-700 text-white';

//   if (typeof content === 'string') {
//     return (
//       <div className={`max-w-[80%] rounded-lg p-3 mb-4 ${messageClass}`}>
//         <p>{content}</p>
//       </div>
//     );
//   }

//   return (
//     <div className="bg-gray-700 text-white rounded-lg p-4 mb-4 max-w-[80%]">
//       <h3 className="font-medium mb-2">{content.title.split('#').pop()}</h3>
//       <p className="mb-2">{content.summary.summary}</p>
//       <div className="space-y-2">
//         <div className="bg-gray-600 p-2 rounded">
//           <span className="font-medium">Risk Level:</span> {content.summary.risk_assessment.risk_level}
//         </div>
//         <div className="bg-gray-600 p-2 rounded">
//           <span className="font-medium">Current Price:</span> {content.summary.price_analysis.current_price}
//         </div>
//         <div className="bg-gray-600 p-2 rounded">
//           <span className="font-medium">Sentiment:</span> {content.summary.sentiment_analysis.overall_sentiment}
//         </div>
//       </div>
//       <div className="mt-2 text-sm text-gray-400">
//         <a href={content.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
//           View Source
//         </a>
//       </div>
//     </div>
//   );
// };