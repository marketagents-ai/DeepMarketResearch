import os
import subprocess
import json
from pathlib import Path

def create_backend_structure():
    # Base backend directory
    backend_dir = Path("backend")
    backend_dir.mkdir(exist_ok=True)
    
    # Create API directory structure
    api_dir = backend_dir / "api"
    routes_dir = api_dir / "routes"
    api_dir.mkdir(exist_ok=True)
    routes_dir.mkdir(exist_ok=True)

    # Create __init__.py files
    (backend_dir / "__init__.py").touch()
    (api_dir / "__init__.py").touch()
    (routes_dir / "__init__.py").touch()

    # Create requirements.txt
    requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
python-multipart==0.0.6
"""
    with open(backend_dir / "requirements.txt", "w") as f:
        f.write(requirements)

    # Create main.py
    main_py = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import tools

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tools.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    with open(backend_dir / "main.py", "w") as f:
        f.write(main_py)

    # Create tools.py
    tools_py = """from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

# Pydantic models
class ToolBase(BaseModel):
    schema_name: str
    schema_description: str
    instruction_string: str
    json_schema: Dict[str, Any]
    strict_schema: bool = True

class CallableToolBase(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    is_callable: bool = True

class Tool(ToolBase):
    id: int
    created_at: datetime

class AutoToolsUpdate(BaseModel):
    tool_ids: List[int]

# Global state (replace with database in production)
tools: List[Dict] = []
auto_tools_ids: List[int] = []
stop_tool_id: Optional[int] = None
tool_counter = 0

@router.get("/tools")
async def get_tools():
    return {
        "tools": tools,
        "autoToolsIds": auto_tools_ids,
        "stopToolId": stop_tool_id
    }

@router.post("/tools")
async def create_tool(tool: ToolBase | CallableToolBase):
    global tool_counter
    tool_counter += 1
    tool_dict = tool.dict()
    tool_dict["id"] = tool_counter
    tool_dict["created_at"] = datetime.now()
    tools.append(tool_dict)
    return tool_dict
"""
    with open(routes_dir / "tools.py", "w") as f:
        f.write(tools_py)

def create_frontend_structure():
    # Create base directory for frontend
    base_dir = Path("chat-frontend")
    base_dir.mkdir(exist_ok=True)

    # Create src directories
    directories = [
        "src/api",
        "src/types",
        "src/components",
        "src/contexts",
        "src/styles"
    ]

    # Create directories
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

    # Create tools.ts
    tools_api = """import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const toolsApi = {
  getAllTools: async () => {
    const response = await axios.get(`${API_BASE_URL}/tools`);
    return response.data;
  },

  createTool: async (tool) => {
    const response = await axios.post(`${API_BASE_URL}/tools`, tool);
    return response.data;
  },

  updateTool: async (toolId, tool) => {
    const response = await axios.put(`${API_BASE_URL}/tools/${toolId}`, tool);
    return response.data;
  },

  deleteTool: async (toolId) => {
    const response = await axios.delete(`${API_BASE_URL}/tools/${toolId}`);
    return response.data;
  }
};"""
    with open(base_dir / "src/api/tools.ts", "w") as f:
        f.write(tools_api)

    # Create types.ts with updated content
    types_content = """export enum MessageRole {
    SYSTEM = 'system',
    USER = 'user',
    ASSISTANT = 'assistant',
    TOOL = 'tool'
}

export enum ResponseFormat {
    TEXT = 'text',
    JSON = 'json',
    MARKDOWN = 'markdown'
}

export interface Message {
    role: MessageRole;
    content: string;
    name?: string;
    function_call?: any;
}

export interface Chat {
    id: number;
    created_at: string;
    messages: Message[];
    system_prompt_id?: number;
    llm_config_id?: number;
}

export interface ChatState {
    id: number | null;
    new_message: string | null;
    history: Message[];
    system_string: string | null;
}

export interface Tool {
    id: number;
    created_at: string;
    is_callable?: boolean;
    schema_name?: string;
    schema_description?: string;
    instruction_string?: string;
    json_schema?: any;
    strict_schema?: boolean;
    name?: string;
    description?: string;
    input_schema?: any;
    output_schema?: any;
}

export interface TypedTool extends Tool {
    is_callable: false;
    schema_name: string;
    schema_description: string;
    instruction_string: string;
    json_schema: any;
    strict_schema: boolean;
}

export interface CallableTool extends Tool {
    is_callable: true;
    name: string;
    description: string;
    input_schema: any;
    output_schema: any;
}

export interface ToolCreate {
    is_callable: boolean;
    schema_name?: string;
    schema_description?: string;
    instruction_string?: string;
    json_schema?: any;
    strict_schema?: boolean;
    name?: string;
    description?: string;
    input_schema?: any;
    output_schema?: any;
}

export interface SystemPrompt {
    id: number;
    created_at: string;
    content: string;
    name: string;
    description: string;
}

export interface SystemPromptCreate {
    content: string;
    name: string;
    description: string;
}

export interface LLMConfig {
    id: number;
    created_at: string;
    temperature: number;
    top_p: number;
    frequency_penalty: number;
    presence_penalty: number;
    max_tokens: number;
    stop: string[];
    response_format: ResponseFormat;
}

export interface LLMConfigUpdate {
    temperature?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    max_tokens?: number;
    stop?: string[];
    response_format?: ResponseFormat;
}

export interface ActivityState {
    isTyping: boolean;
    isProcessing: boolean;
    error: string | null;
}"""
    with open(base_dir / "src/types/index.ts", "w") as f:
        f.write(types_content)

def create_project_structure():
    # Create frontend structure
    create_frontend_structure()
    
    # Create backend structure
    create_backend_structure()

    print("\nProject setup complete! Follow these steps to start developing:")
    print("\n1. Set up the frontend:")
    print("   cd chat-frontend")
    print("   npm install")
    print("   npm start")
    
    print("\n2. Set up the backend:")
    print("   cd ../backend")
    print("   pip install -r requirements.txt")
    print("   uvicorn main:app --reload")
    
    print("\nThe development servers will start at:")
    print("- Frontend: http://localhost:3000")
    print("- Backend: http://localhost:8000")
    print("\nAPI Documentation will be available at:")
    print("- http://localhost:8000/docs")

if __name__ == "__main__":
    create_project_structure()