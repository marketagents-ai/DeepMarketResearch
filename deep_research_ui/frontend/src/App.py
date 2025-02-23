import React, { useState } from 'react';
import { ChatInput } from './components/ChatInput/ChatInput';
import { ChatHistory } from './components/ChatHistory';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Header } from './components/Header/Header';
import { ToolsPanel } from './components/ToolsPanel/ToolsPanel';
import { ChatModeSelector } from './components/ChatModes/ChatModeSelector';
import { useChats } from './hooks/useChats';
import { fetchResearch, sendCustomMessage } from './services/api';
import { handleAPIError } from './utils/api';
import type { ChatItem, ChatMode } from './types/chat';

export const App: React.FC = () => {
  const { chats, activeChat, setActiveChat, createChat, updateChat, updateChatMode } = useChats();
  const [isLoading, setIsLoading] = useState(false);

  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
  const activeChatMode = chats.find(chat => chat.id === activeChat)?.mode || 'research';
  const handleSubmit = async (query: string, urls?: string[]) => {
    if (!activeChat) {
      const newChatId = createChat(activeChatMode);
      if (!newChatId) return;
    }
  
    const chatId = activeChat || chats[0].id;
    const userMessage: ChatItem = {
      isUser: true,
      content: query,
      timestamp: new Date(),
      mode: activeChatMode
    };
    
    const newMessages = [...activeMessages, userMessage];
    updateChat(chatId, newMessages);
    
    setIsLoading(true);
    try {
      let responseData;
      
      if (activeChatMode === 'research') {
        responseData = await fetchResearch(query, urls);
      } else {
        // Get the active system message from the ToolsPanel if available
        const systemMessage = activeSystemMessage?.content;
        responseData = await sendCustomMessage(query, systemMessage);
      }
  
      const responseMessage: ChatItem = {
        isUser: false,
        content: responseData,
        timestamp: new Date(),
        mode: activeChatMode
      };
      updateChat(chatId, [...newMessages, responseMessage]);
    } catch (error) {
      const errorMessage: ChatItem = {
        isUser: false,
        content: handleAPIError(error),
        timestamp: new Date(),
        mode: activeChatMode
      };
      updateChat(chatId, [...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // const handleSubmit = async (query: string, urls?: string[]) => {
  //   if (!activeChat) {
  //     const newChatId = createChat(activeChatMode);
  //     if (!newChatId) return;
  //   }

  //   const chatId = activeChat || chats[0].id;
  //   const userMessage: ChatItem = {
  //     isUser: true,
  //     content: query,
  //     timestamp: new Date(),
  //     mode: activeChatMode
  //   };
    
  //   const newMessages = [...activeMessages, userMessage];
  //   updateChat(chatId, newMessages);
    
  //   setIsLoading(true);
  //   try {
  //     let responseData;
      
  //     if (activeChatMode === 'research') {
  //       responseData = await fetchResearch(query, urls);
  //     } else {
  //       // Custom mode using OpenAI
  //       responseData = await sendCustomMessage(query);
  //     }

  //     const responseMessage: ChatItem = {
  //       isUser: false,
  //       content: responseData,
  //       timestamp: new Date(),
  //       mode: activeChatMode
  //     };
  //     updateChat(chatId, [...newMessages, responseMessage]);
  //   } catch (error) {
  //     const errorMessage: ChatItem = {
  //       isUser: false,
  //       content: handleAPIError(error),
  //       timestamp: new Date(),
  //       mode: activeChatMode
  //     };
  //     updateChat(chatId, [...newMessages, errorMessage]);
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  const handleModeChange = (mode: ChatMode) => {
    if (activeChat) {
      updateChatMode(activeChat, mode);
    } else {
      createChat(mode);
    }
  };
  return (
    <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
      <Sidebar
        chats={chats}
        activeChat={activeChat}
        onNewChat={() => createChat(activeChatMode)}
        onSelectChat={setActiveChat}
      />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <ChatModeSelector 
          mode={activeChatMode} 
          onModeChange={handleModeChange}
          disabled={isLoading} 
        />
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatHistory messages={activeMessages} />
          <div className="border-t border-gray-800 p-4">
            <ChatInput
              mode={activeChatMode}
              onSubmit={handleSubmit}
              isLoading={isLoading}
              disabled={!activeChat}
            />
          </div>
        </div>
      </main>
      <ToolsPanel />
    </div>
  );

//   return (
//     <div className="flex h-screen bg-gray-900 text-white">
//       <Sidebar
//         chats={chats}
//         activeChat={activeChat}
//         onNewChat={() => createChat(activeChatMode)}
//         onSelectChat={setActiveChat}
//       />
//       <main className="flex-1 flex flex-col">
//         <Header />
//         <ChatModeSelector 
//           mode={activeChatMode} 
//           onModeChange={handleModeChange}
//           disabled={isLoading} 
//         />
//         <div className="flex-1 flex flex-col">
//           <ChatHistory messages={activeMessages} />
//           <div className="border-t border-gray-800 p-4">
//             <ChatInput
//               mode={activeChatMode}
//               onSubmit={handleSubmit}
//               isLoading={isLoading}
//               disabled={!activeChat}
//             />
//           </div>
//         </div>
//       </main>
//       <ToolsPanel />
//     </div>
//   );
// };





// import React, { useState } from 'react';
// import { ChatInput } from './components/ChatInput/ChatInput';
// import { ChatHistory } from './components/ChatHistory';
// import { Sidebar } from './components/Sidebar/Sidebar';
// import { Header } from './components/Header/Header';
// import { ToolsPanel } from './components/ToolsPanel/ToolsPanel';
// import { ChatModeSelector } from './components/ChatModes/ChatModeSelector';
// import { useChats } from './hooks/useChats';
// import { fetchResearch } from './services/api';
// import { handleAPIError } from './utils/api';
// import type { ChatItem, ChatMode } from './types/chat';
// import { CustomMode } from './components/ChatModes/CustomMode';
// import { ResearchMode } from './components/ChatModes/ResearchMode';
// import type { ChatMode } from './types/chat';

// export const App: React.FC = () => {
//   const { chats, activeChat, setActiveChat, createChat, updateChat, updateChatMode } = useChats();
//   const [isLoading, setIsLoading] = useState(false);

//   const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
//   const activeChatMode = chats.find(chat => chat.id === activeChat)?.mode || 'research';

//   const handleSubmit = async (query: string, urls?: string[]) => {
//     if (!activeChat) {
//       const newChatId = createChat(activeChatMode);
//       if (!newChatId) return;
//     }

//     const chatId = activeChat || chats[0].id;
//     const userMessage: ChatItem = {
//       isUser: true,
//       content: query,
//       timestamp: new Date(),
//       mode: activeChatMode
//     };
    
//     const newMessages = [...activeMessages, userMessage];
//     updateChat(chatId, newMessages);
    
//     setIsLoading(true);
//     try {
//       const data = await fetchResearch(query, urls);
//       const responseMessage: ChatItem = {
//         isUser: false,
//         content: data,
//         timestamp: new Date(),
//         mode: activeChatMode
//       };
//       updateChat(chatId, [...newMessages, responseMessage]);
//     } catch (error) {
//       const errorMessage: ChatItem = {
//         isUser: false,
//         content: handleAPIError(error),
//         timestamp: new Date(),
//         mode: activeChatMode
//       };
//       updateChat(chatId, [...newMessages, errorMessage]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleModeChange = (mode: ChatMode) => {
//     if (activeChat) {
//       updateChatMode(activeChat, mode);
//     } else {
//       createChat(mode);
//     }
//   };

//   return (
//     <div className="flex h-screen bg-gray-900 text-white">
//       <Sidebar
//         chats={chats}
//         activeChat={activeChat}
//         onNewChat={() => createChat(activeChatMode)}
//         onSelectChat={setActiveChat}
//       />
      
//       <div className="flex-1 flex flex-col">
//         <Header />
//         <div className="p-4">
//           <ChatModeSelector
//             mode={activeChatMode}
//             onModeChange={handleModeChange}
//           />
//         </div>
//         <ChatHistory messages={activeMessages} />
//         <ChatInput 
//           mode={activeChatMode}
//           onSubmit={handleSubmit}
//           isLoading={isLoading}
//           disabled={false}
//         />
//       </div>

//       <ToolsPanel />
//     </div>
//   );
// };


 
/////////////////////////////////////////////////////////////////

// import React, { useState } from 'react';
// import { ChatInput } from './components/ChatInput/ChatInput';
// import { ChatHistory } from './components/ChatHistory';
// import { Sidebar } from './components/Sidebar/Sidebar';
// import { Header } from './components/Header/Header';
// import { ToolsPanel } from './components/ToolsPanel/ToolsPanel';
// import { ChatModeSelector } from './components/ChatModes/ChatModeSelector';
// import { useChats } from './hooks/useChats';
// import { fetchResearch } from './services/api';
// import { handleAPIError } from './utils/api';
// import type { ChatItem, ChatMode } from './types/chat';

// export const App: React.FC = () => {
//   const { chats, activeChat, setActiveChat, createChat, updateChat, updateChatMode } = useChats();
//   const [isLoading, setIsLoading] = useState(false);

//   const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
//   const activeChatMode = chats.find(chat => chat.id === activeChat)?.mode || 'research';

//   const handleNewChat = () => {
//     const newChatId = createChat(activeChatMode);
//     setActiveChat(newChatId);  // Make sure to set the active chat after creation
//     const handleModeChange = (newMode: ChatMode) => {
//       setChatMode(newMode);
//       if (activeChat) {
//         updateChatMode(activeChat, newMode);
//       }
//   };

//   const handleSubmit = async (query: string, urls?: string[]) => {
//     try {
//       if (!activeChat) {
//         const newChatId = createChat(activeChatMode);
//         if (!newChatId) return;
//       }

//       // Add user message
//       updateChat(activeChat!, [
//         ...activeMessages,
//         { isUser: true, content: query }
//       ]);

//       setIsLoading(true);
//       const results = await fetchResearch(query, urls);
      
//       // Add research results
//       updateChat(activeChat!, [
//         ...activeMessages,
//         { isUser: true, content: query },
//         { isUser: false, content: results }
//       ]);
//     } catch (error) {
//       const errorMessage = handleAPIError(error);
//       // Add error message to chat
//       updateChat(activeChat!, [
//         ...activeMessages,
//         { isUser: true, content: query },
//         { isUser: false, content: `Error: ${errorMessage}` }
//       ]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleModeChange = (mode: ChatMode) => {
//     if (activeChat) {
//       updateChatMode(activeChat, mode);
//     }
//   };
//   return (
//     <div className="h-screen flex flex-col bg-gray-900 text-white">
//       <Header />
//       <div className="flex-1 flex overflow-hidden">
//         <Sidebar
//           chats={chats}
//           activeChat={activeChat}
//           onNewChat={handleNewChat}
//           onSelectChat={setActiveChat}
//         />
        
//         <main className="flex-1 flex flex-col">
//           <div className="p-4 border-b border-gray-700">
//             <ChatModeSelector
//               mode={activeChatMode}
//               onModeChange={handleModeChange}
//             />
//           </div>
          
//           <ChatHistory messages={activeMessages} />
          
//           <ChatInput
//             mode={activeChatMode}
//             onSubmit={handleSubmit}
//             isLoading={isLoading}
//             disabled={!activeChat}
//           />
//         </main>
        
//         <ToolsPanel />
//       </div>
//     </div>
//   );
// };