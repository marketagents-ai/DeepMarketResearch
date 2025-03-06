from datetime import datetime
import sys
import asyncio
import subprocess
import logging
from typing import Dict, Any, List, Union, Optional, Type
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp import ClientSession

import sys

from pydantic import Field, BaseModel

from market_agents.environments.environment import (
    EnvironmentHistory,
    MultiAgentEnvironment, 
    Mechanism,
    LocalAction,
    LocalObservation,
    GlobalAction,
    GlobalObservation,
    LocalEnvironmentStep,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace,
    StrAction
)
from minference.lite.models import CallableMCPTool, CallableTool, StructuredTool
from minference.caregistry import CallableRegistry

logger = logging.getLogger(__name__)
CallableRegistry._logger = logger

class MCPServerResult(BaseModel):
    """Structure for a single MCP server tool result"""
    tool_name: str
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MCPServerLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: Dict[str, Any]
    status: str = "pending"
    tool_results: Optional[List[MCPServerResult]] = None

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation
        return d

class MCPServerGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, MCPServerLocalObservation]

class MCPToolAction(LocalAction):
    """Action for invoking an MCP server tool"""
    tool_name: str = Field(..., description="Name of the tool to invoke")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")

    @classmethod
    def sample(cls, agent_id: str) -> 'MCPToolAction':
        """Sample a random tool action (not implemented)"""
        # This would require knowledge of the available tools and their parameters
        # For now, just return a placeholder
        return cls(agent_id=agent_id, tool_name="sample_tool", tool_args={})

class MCPServerMechanism(Mechanism):
    """Mechanism that manages MCP server tool interactions"""
    current_round: int = Field(default=0, description="Current interaction round")
    max_rounds: int = Field(default=10, description="Maximum interaction rounds")
    tool_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of tool invocations"
    )
    available_tools: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Available tools from the MCP server"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize session-related attributes
        self._active_session = None
        self._session = None
        self._read_write = None
        self.client_initialized = False
        
        # Handle different ways of providing the MCP server
        if "mcp_server" in data:
            # Direct server instance provided
            self.mcp_server = data["mcp_server"]
            self.server_process = None
            self.is_external_server = True
            print(f"Using provided MCP server instance: {type(self.mcp_server)}")
        elif "server_path" in data:
            # Path to server script provided - we'll start it ourselves
            self.server_path = data["server_path"]
            self.server_process = None
            self.is_external_server = False
            # Start the server process
            self._start_server_process()
            # Initialize client connection parameters
            self._initialize_client_connection()
            # The mcp_server attribute will be set after initializing the client
            print(f"Started MCP server from path: {self.server_path}")
        else:
            raise ValueError("Either mcp_server or server_path must be provided")
        
        self.available_tools = {}
    
    def _start_server_process(self):
        """Start the MCP server as a subprocess"""
        try:
            # Check if we should use mcp run or direct python execution
            if self.server_path.endswith('.py'):
                # Use direct Python execution
                self.server_process = subprocess.Popen(
                    [sys.executable, self.server_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Started MCP server process with PID: {self.server_process.pid}")
            else:
                # Use mcp run command
                self.server_process = subprocess.Popen(
                    ["mcp", "run", self.server_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Started MCP server with mcp run, PID: {self.server_process.pid}")
                
            # Give the server a moment to start up
            import time
            time.sleep(2)
        except Exception as e:
            print(f"Error starting MCP server: {str(e)}")
            raise

    def _initialize_client_connection(self):
        """Initialize client connection to the MCP server"""
        
        try:
            # Create server parameters for the client
            self.server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_path],
                env=None
            )
            print(f"Initialized client connection parameters for server: {self.server_path}")
            
            
            # Set the mcp_server attribute to the server parameters
            # This will be used to create sessions when needed
            self.mcp_server = self.server_params
            print("Client connection parameters initialized")
            
        except Exception as e:
            print(f"Error initializing client connection: {str(e)}")
            raise
    
    async def _get_or_create_session(self):
        """Get an existing session or create a new one if needed"""
        
        # Check if we already have an active session
        if hasattr(self, '_active_session') and self._active_session:
            return self._active_session
        
        # Create a new session
        self._read_write = await stdio_client(self.server_params).__aenter__()
        self._session = ClientSession(*self._read_write)
        self._active_session = await self._session.__aenter__()
        
        # Initialize the session
        await self._active_session.initialize()
        
        return self._active_session

    async def _close_session(self):
        """Close the active session if one exists"""
        if hasattr(self, '_active_session') and self._active_session:
            try:
                await self._session.__aexit__(None, None, None)
                await self._read_write[0].__aexit__(None, None, None)
                self._active_session = None
                self._session = None
                self._read_write = None
            except Exception as e:
                print(f"Error closing session: {str(e)}")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool using an MCP client session"""
        try:
            # Get or create a session
            session = await self._get_or_create_session()
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments=arguments)
            
            # Record the tool execution in history
            self.tool_history.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        except Exception as e:
            print(f"Error executing tool {tool_name}: {str(e)}")
            # Close the session on error to ensure a fresh start next time
            await self._close_session()
            raise

    async def initialize(self):
        """Initialize the mechanism by extracting available tools"""
        try:
            async with stdio_client(self.mcp_server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print("Connected to MCP Server")
                    
                    tools_result = await session.list_tools()
                    
                    if hasattr(tools_result, 'tools'):
                        self.available_tools = {}
                        for tool_info in tools_result.tools:
                            self.available_tools[tool_info.name] = {
                                "name": tool_info.name,
                                "description": tool_info.description,
                                "input_schema": tool_info.inputSchema
                            }
                        print(f"Found {len(self.available_tools)} tools")
                    else:
                        print("No tools attribute found in result")
                        self.available_tools = {}
                        
        except Exception as e:
            print(f"Error initializing mechanism: {str(e)}")
            import traceback
            traceback.print_exc()
            self.available_tools = {}
            
    async def step(
        self,
        action: Union[GlobalAction, str]
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """
        Process agent actions and track the current round.
        
        Note: This method doesn't execute tools directly - tools are executed
        by the CallableMCPTool instances when used by agents.
        """
        self.current_round += 1
        done = (self.current_round >= self.max_rounds)

        if isinstance(action, GlobalAction):
            observations = {}
            
            for agent_id, agent_action in action.actions.items():
                obs_data = {
                    "action": agent_action.model_dump() if hasattr(agent_action, 'model_dump') else str(agent_action),
                    "round": self.current_round,
                    "status": "success"
                }
                
                # Create the local observation
                observations[agent_id] = MCPServerLocalObservation(
                    agent_id=agent_id,
                    observation=obs_data,
                    status=obs_data["status"]
                )
            
            # Create global observation
            global_obs = MCPServerGlobalObservation(observations=observations)
            
            # Return environment step
            return EnvironmentStep(
                global_action=action,
                global_observation=global_obs,
                done=done
            )
        
        elif isinstance(action, LocalAction):
            # Handle single agent action
            agent_id = action.agent_id
            obs_data = {
                "action": action.model_dump() if hasattr(action, 'model_dump') else str(action),
                "round": self.current_round,
                "status": "success"
            }
            
            # Create the local observation
            local_obs = MCPServerLocalObservation(
                agent_id=agent_id,
                observation=obs_data,
                status=obs_data["status"]
            )
            
            # Return local environment step
            return LocalEnvironmentStep(
                observation=local_obs,
                done=done
            )
        
        else:
            # Handle string actions or other types
            return LocalEnvironmentStep(
                observation=MCPServerLocalObservation(
                    agent_id="system",
                    observation={"action": str(action), "round": self.current_round},
                    status="success"
                ),
                done=done
            )

    def get_global_state(self) -> Dict[str, Any]:
        """Get the current global state of the mechanism"""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "tool_history": self.tool_history,
            "available_tools": list(self.available_tools.keys())
        }
    
    async def cleanup(self):
        """Clean up resources when the mechanism is no longer needed"""
        await self._close_session()
        
        if hasattr(self, 'server_process') and self.server_process:
            try:
                self.server_process.terminate()
                print(f"Terminated MCP server process with PID: {self.server_process.pid}")
            except Exception as e:
                print(f"Error terminating server process: {str(e)}")

    def __del__(self):
        """Clean up resources when the mechanism is destroyed"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            print(f"Error in __del__: {str(e)}")

class MCPServerActionSpace(ActionSpace):
    """Action space that handles MCP server tool invocations"""
    mechanism: MCPServerMechanism = Field(
        ..., 
        description="Mechanism that handles MCP server operations"
    )
    
    def __init__(self, mechanism: MCPServerMechanism, **data):
        data.update({
            "mechanism": mechanism
        })
        super().__init__(**data)
        
        self.allowed_actions = []
        
        print(f"Available tools in mechanism: {list(mechanism.available_tools.keys())}")
        
        # Create a tool for each available MCP server tool
        for tool_name, tool_info in mechanism.available_tools.items():
            print(f"Creating CallableMCPTool for {tool_name}")
            try:
                from minference.lite.models import CallableMCPTool
                
                # Check if we have the function object
                if "input_schema" in tool_info and tool_info["input_schema"] is not None:
                    # Create the tool from the function
                    mcp_tool = CallableMCPTool.from_callable(
                        name=tool_name,
                        description=tool_info.get("description"),
                        input_schema=tool_info.get("input_schema")
                    )
                else:
                    # Skip this tool since we don't have a callable function
                    print(f"Skipping tool {tool_name} - no callable function provided")
                    continue
                
                # Set the MCP server
                mcp_tool.mcp_mechanism = mechanism
                
                self.allowed_actions.append(mcp_tool)
                print(f"Successfully added tool: {tool_name}")
            except Exception as e:
                print(f"Error creating tool {tool_name}: {str(e)}")
        
        print(f"Total allowed_actions: {len(self.allowed_actions)}")

    
    def get_action_schema(self):
        """Return JSON schema for all available tools"""
        schemas = {}
        for tool in self.allowed_actions:
            schemas[tool.name] = tool.json_schema()
        return schemas

class MCPServerEnvironment(MultiAgentEnvironment):
    """Environment that manages MCP server operations"""
    name: str = Field(
        default="MCP Server Environment",
        description="Name of the environment"
    )
    mechanism: MCPServerMechanism = Field(
        ...,
        description="Mechanism that handles MCP server operations"
    )
    action_space: MCPServerActionSpace = None
    internal_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Internal storage for global state"
    )
    
    def __init__(
        self,
        name: str = "MCP Server Environment",
        mechanism: Optional[MCPServerMechanism] = None,
        **data
    ):
        """Initialize the MCP server environment."""
        if not mechanism:
            raise ValueError("mechanism must be provided")

        model_data = {
            "name": name,
            "mechanism": mechanism,
            **data
        }
        super().__init__(**model_data)
        
        self.action_space = None
        
        self.internal_state = {}

    async def initialize(self):
        """Initialize the environment by setting up mechanism and action space"""
        # Initialize mechanism first
        await self.mechanism.initialize()
        
        # Create action space with available tools
        print(f"Creating action space with {len(self.mechanism.available_tools)} available tools")
        self.action_space = MCPServerActionSpace(mechanism=self.mechanism)
    
    def get_global_state(self):
        """Get current global state combining mechanism and environment state"""
        state = self.mechanism.get_global_state()
        state.update(self.internal_state)
        return state
    
    async def step(self, action: Union[GlobalAction, LocalAction, str]) -> Union[EnvironmentStep, LocalEnvironmentStep]:
        """Process an action and return the resulting step"""
        # Delegate to the mechanism's step method
        return await self.mechanism.step(action)
    
    def reset(self):
        """Reset environment state"""
        self.mechanism.current_round = 0
        self.mechanism.tool_history = []
        self.internal_state = {}
        self.current_step = 0
        
        # Create initial observations for all agents
        observations = {}
        
        # Return global observation
        return MCPServerGlobalObservation(observations=observations)
    
    async def cleanup(self):
        """Clean up resources when the environment is no longer needed"""
        await self.mechanism.cleanup()