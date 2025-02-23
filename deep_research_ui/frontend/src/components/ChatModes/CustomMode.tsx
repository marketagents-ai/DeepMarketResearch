// import React, { useState } from 'react';
// import { ChatInputCustom } from '../ChatInput/ChatInputCustom';
// import { ChatHistory } from '../ChatHistory';
// import { sendCustomMessage } from '../../services/api';

// const handleSubmit = async (message: string, file?: File) => {
//     try {
//       setIsLoading(true);
//       console.log('CustomMode - Preparing request:', message, file); // Debug log
  
//       const formData = new FormData();
//       formData.append('message', message);
//       if (file) {
//         formData.append('file', file);
//       }
  
//       const response = await sendCustomMessage(formData);
//       console.log('CustomMode - Response received:', response); // Debug log
  
//       setMessages(prev => [
//         ...prev,
//         {
//           isUser: true,
//           content: message,
//           timestamp: new Date()
//         },
//         {
//           isUser: false,
//           content: response.message || 'Message processed successfully',
//           timestamp: new Date()
//         }
//       ]);
//     } catch (error) {
//       console.error('Error in custom mode:', error);
//       // Add error message to chat
//       setMessages(prev => [
//         ...prev,
//         {
//           isUser: false,
//           content: 'An error occurred while processing your message.',
//           timestamp: new Date()
//         }
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="flex flex-col h-full">
//       <div className="flex-1 overflow-y-auto p-4">
//         <ChatHistory messages={messages} />
//       </div>
//       <div className="border-t border-gray-800 p-4">
//         <ChatInputCustom
//           onSubmit={handleSubmit}
//           isLoading={isLoading}
//         />
//       </div>
//     </div>
//   );
// };


// // src/components/ChatModes/CustomMode.tsx

// import React, { useState } from 'react';
// import { ChatInput } from '../ChatInput/ChatInputCustom';

// export const CustomMode: React.FC = () => {
//   const [messages, setMessages] = useState<Array<{ text: string; isUser: boolean }>>([]);
//   const [uploadedData, setUploadedData] = useState<any[]>([]);

//   const handleSubmit = (message: string) => {
//     // Add user message
//     setMessages(prev => [...prev, { text: message, isUser: true }]);
    
//     // Add system response
//     setMessages(prev => [...prev, { 
//       text: `Analyzing your message: ${message}`, 
//       isUser: false 
//     }]);
//   };

//   const handleFileUpload = (data: any[]) => {
//     setUploadedData(data);
//     setMessages(prev => [...prev, {
//       text: `Received data with ${data.length} entries`,
//       isUser: false
//     }]);
//   };

//   return (
//     <div className="flex flex-col h-full">
//       <div className="flex-1 overflow-y-auto p-4">
//         {/* Display Messages */}
//         {messages.map((msg, index) => (
//           <div 
//             key={index}
//             className={`mb-4 ${msg.isUser ? 'text-right' : 'text-left'}`}
//           >
//             <div className={`inline-block p-3 rounded-lg ${
//               msg.isUser ? 'bg-blue-500' : 'bg-gray-700'
//             }`}>
//               {msg.text}
//             </div>
//           </div>
//         ))}

//         {/* Display Uploaded Data */}
//         {uploadedData.length > 0 && (
//           <div className="mt-4 p-4 bg-gray-800 rounded-lg">
//             <h3 className="text-lg font-medium mb-2">Uploaded Data Analysis</h3>
//             <div className="max-h-60 overflow-y-auto">
//               <pre className="text-sm">
//                 {JSON.stringify(uploadedData, null, 2)}
//               </pre>
//             </div>
//           </div>
//         )}
//       </div>

//       <div className="p-4 border-t border-gray-700">
//         <ChatInput 
//           onSubmit={handleSubmit}
//           onFileUpload={handleFileUpload}
//           isLoading={false}
//         />
//       </div>
//     </div>
//   );
// };









// // woking at Jan 10, Friday


// // // frontend/src/components/ChatModes/CustomMode.tsx

import React, { useState } from 'react';
import { ChatInput } from '../ChatInput/ChatInputCustom';
import { ChatHistory } from '../ChatHistory';
import { useSystem } from '../../hooks/useSystem';
import { sendCustomMessage } from '../../services/api';
import type { ChatItem } from '../../types/chat';

export const CustomMode: React.FC = () => {
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { activeMessage } = useSystem();

  const handleSubmit = async (message: string) => {
    try {
      setIsLoading(true);
      
      // Add user message
      setMessages(prev => [...prev, { isUser: true, content: message }]);
      
      // Get response from API
      const response = await sendCustomMessage(message, activeMessage?.content);
      
      // Add AI response
      setMessages(prev => [...prev, { isUser: false, content: response }]);
    } catch (error) {
      console.error('Error:', error);
      // Handle error appropriately
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4">
        <ChatHistory messages={messages} />
      </div>
      <div className="border-t border-gray-800 p-4">
        <ChatInput
          onSubmit={handleSubmit}
          isLoading={isLoading}
          disabled={!activeMessage}
        />
      </div>
    </div>
  );
};