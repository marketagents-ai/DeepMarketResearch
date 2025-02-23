
import { APIError } from '../utils/api';
import type { ResearchData } from '../types/research';

// Change the API_URL definition
const API_URL = 'http://localhost:8000'; 



export const fetchResearch = async (
  query: string, 
  customSchemas: any[] = []
): Promise<ResearchData[]> => {
  try {
    // Get enabled tools from localStorage
    const savedTools = localStorage.getItem('customTools');
    const tools = savedTools ? JSON.parse(savedTools) : [];
    const enabledTools = tools.filter((tool: any) => tool.enabled).map((tool: any) => ({
      name: tool.name,
      description: tool.description,
      schema_definition: tool.schema
    }));

    console.log('Query being sent:', query);
    console.log('Enabled tools from localStorage:', enabledTools);
    console.log('Custom schemas before request:', JSON.stringify(customSchemas, null, 2));
    console.log('Custom schemas type:', typeof customSchemas);

    const requestBody = {
      query,
      custom_schemas: enabledTools
    };

    console.log('Full request body:', JSON.stringify(requestBody, null, 2));

    const response = await fetch(`${API_URL}/api/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new APIError(errorData.detail || `Error: ${response.status}`, response.status);
    }

    const data = await response.json();
    
    // Add error handling for malformed response
    if (!data) {
      console.error('Empty response from research service');
      return [];
    }

    console.log('Research API response:', data);
    return data;
    
  } catch (error) {
    console.error('Research API Error:', error);
    throw error instanceof APIError ? error : new APIError('Failed to connect to research service');
  }
};
export const sendCustomMessage = async (message: string): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/custom`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      throw new APIError('Failed to send custom message');
    }

    return await response.json();
  } catch (error) {
    console.error('Custom API Error:', error);
    throw error instanceof APIError ? error : new APIError('Failed to send message');
  }
};

export const createTool = async (tool: { name: string; description: string }) => {
  try {
    const response = await fetch(`${API_URL}/api/tools`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tool),
    });

    if (!response.ok) {
      throw new APIError('Failed to create tool');
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating tool:', error);
    throw error instanceof APIError ? error : new APIError('Failed to create tool');
  }
};
export const deleteCustomTool = async (toolId: string): Promise<void> => {
  const response = await fetch(`${API_URL}/tools/${toolId}`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new APIError(error.detail || 'Failed to delete tool');
  }
};

export const getCustomTools = async (): Promise<CustomTool[]> => {
  const response = await fetch(`${API_URL}/tools`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new APIError(error.detail || 'Failed to fetch tools');
  }

  return response.json();
};
export const fetchTools = async () => {
  try {
    const response = await fetch(`${API_URL}/api/tools`);
    if (!response.ok) {
      throw new APIError('Failed to fetch tools');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching tools:', error);
    throw error instanceof APIError ? error : new APIError('Failed to fetch tools');
  }
};
export const submitResearch = async (query: string): Promise<ResearchData[]> => {
  try {
    const enabledSchemas = getEnabledSchemas();
    
    const response = await fetch(`${API_URL}/api/research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        customSchemas: enabledSchemas
      }),
    });

    if (!response.ok) {
      throw new APIError('Research request failed');
    }

    const data = await response.json();
    return data;
    
  } catch (error) {
    throw error instanceof APIError ? error : new APIError('Failed to connect to research service');
  }
};

// import { APIError } from '../utils/api';
// import type { ResearchData } from '../types/research';

// const API_URL = 'http://0.0.0.0:8000';

// export const fetchResearch = async (query: string, urls?: string[]): Promise<ResearchData[]> => {
//   try {
//     const response = await fetch(`${API_URL}/api/research`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         query,
//         urls: urls || [],
//       }),
//     });

//     if (!response.ok) {
//       const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch research data' }));
//       throw new APIError(errorData.detail || `Error: ${response.status}`, response.status);
//     }

//     const data = await response.json();
//     console.log('API Response:', data); // Add this debug log

//     // Check if data is an array
//     if (!Array.isArray(data)) {
//       console.error('Response is not an array:', data);
//       throw new APIError('Invalid response format: expected array');
//     }

//     // Validate each item in the array
//     const isValidResearchData = (item: any): item is ResearchData => {
//       const hasRequiredFields = item &&
//         typeof item.url === 'string' &&
//         typeof item.title === 'string' &&
//         typeof item.content === 'string' &&
//         typeof item.timestamp === 'string' &&
//         typeof item.status === 'string' &&
//         item.summary &&
//         item.agent_id &&
//         item.extraction_method;

//       if (!hasRequiredFields) {
//         console.error('Invalid item format:', item);
//         return false;
//       }
//       return true;
//     };

//     if (!data.every(isValidResearchData)) {
//       throw new APIError('Invalid response format: items missing required fields');
//     }

//     return data;
//   } catch (error) {
//     console.error('Research API Error:', error);
//     throw error instanceof APIError ? error : new APIError('Failed to connect to research service');
//   }
// };



/////////////////////////////////////////////////////////////

// export const fetchResearch = async (query: string, urls?: string[]): Promise<ResearchData[]> => {
//   try {
//     const response = await fetch(`${API_URL}/api/research`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ query, urls }),
//     });

//     if (!response.ok) {
//       const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch research data' }));
//       throw new APIError(errorData.detail || `Error: ${response.status}`, response.status);
//     }

//     const data = await response.json();
//     return data;
//   } catch (error) {
//     console.error('Research API Error:', error);
//     throw new APIError('Failed to connect to research service');
//   }
// };


///WORKED
// export const fetchResearch = async (query: string, urls?: string[]): Promise<ResearchData[]> => {
//   try {
//     const response = await fetch(`${API_URL}/api/research`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         query,
//         urls: urls || [],

//       }),
//     });

//     if (!response.ok) {
//       const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch research data' }));
//       throw new APIError(errorData.detail || `Error: ${response.status}`, response.status);
//     }

//     const data = await response.json();
//     if (!Array.isArray(data)) {
//       throw new APIError('Invalid response format from research service');
//     }

//     return data;
//   } catch (error) {
//     console.error('Research API Error:', error);
//     throw error instanceof APIError ? error : new APIError('Failed to connect to research service');
//   }
// };