export const ResearchMode: React.FC = () => {
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { activeMessage } = useSystem();
  const [customSchemas, setCustomSchemas] = useState<any[]>([]);

  useEffect(() => {
    const loadCustomTools = () => {
      try {
        const savedTools = localStorage.getItem('customTools');
        if (savedTools) {
          const tools = JSON.parse(savedTools);
          const enabledSchemas = tools
            .filter((tool: any) => tool.enabled)
            .map((tool: any) => tool.schema);
          setCustomSchemas(enabledSchemas);
        }
      } catch (error) {
        console.error('Error loading custom tools:', error);
        setCustomSchemas([]);
      }
    };

    loadCustomTools();
    window.addEventListener('customToolsUpdated', loadCustomTools);
    return () => window.removeEventListener('customToolsUpdated', loadCustomTools);
  }, []);

  const handleSubmit = async (query: string, urls?: string[]) => {
    setIsLoading(true);
    try {
      // Add user message
      setMessages(prev => [...prev, { 
        isUser: true, 
        content: query,
        timestamp: new Date(),
        customSchemas
      }]);

      // Send request with custom schemas
      const response = await sendResearchMessage({
        query,
        urls,
        customSchemas,
        systemMessage: activeMessage?.content
      });

      // Add AI response with custom schemas
      setMessages(prev => [...prev, { 
        isUser: false, 
        content: response,
        timestamp: new Date(),
        customSchemas
      }]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4">
        <ChatHistory 
          messages={messages} 
          customSchemas={customSchemas} 
        />
      </div>
      <div className="border-t border-gray-800 p-4">
        <ChatInputResearch
          onSubmit={handleSubmit}
          isLoading={isLoading}
          disabled={!activeMessage}
        />
      </div>
    </div>
  );
};