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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
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
            self.logger.info(f"Using provided MCP server instance: {type(self.mcp_server)}")
        elif "server_path" in data:
            # Path to server script provided - we'll start it ourselves
            self.server_path = data["server_path"]
            self.server_process = None
            self.is_external_server = False
            # Start the server process
            self._start_server_process()
            # Initialize client connection parameters
            self._initialize_client_connection()
            self.logger.info(f"Started MCP server from path: {self.server_path}")
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

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool using a fresh MCP client session"""
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Create a fresh session for each tool execution
                async with asyncio.timeout(30):  # 30 second timeout
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, arguments=arguments)
                            
                            # Convert CallToolResult to dict
                            if hasattr(result, 'model_dump'):
                                result = result.model_dump()
                            elif hasattr(result, 'dict'):
                                result = result.dict()
                            elif hasattr(result, '__dict__'):
                                result = result.__dict__
                            
                            print(f"tool result:\n{result}")
                            # Record successful execution
                            self.tool_history.append({
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "result": result,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            return result
                    
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                self.logger.error(f"Tool execution timed out or was cancelled: {str(e)}")
                last_error = e
                # Don't retry on explicit cancellation
                if isinstance(e, asyncio.CancelledError):
                    break
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                last_error = e
            
            # Increment retry counter and wait before retrying
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait before retrying
        
        # If we've exhausted retries or got cancelled, raise the last error
        if isinstance(last_error, asyncio.CancelledError):
            raise last_error
        raise last_error or Exception(f"Failed to execute tool {tool_name} after {max_retries} retries")

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
            
            # Return environment step with required info field
            return EnvironmentStep(
                global_action=action,
                global_observation=global_obs,
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
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
            
            # Return local environment step with required info field
            return LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
            )
        
        else:
            # Handle string actions or other types
            return LocalEnvironmentStep(
                observation=MCPServerLocalObservation(
                    agent_id="system",
                    observation={"action": str(action), "round": self.current_round},
                    status="success"
                ),
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
            )
    def get_global_state(self) -> Dict[str, Any]:
        """Get the current global state of the mechanism"""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "tool_history": self.tool_history,
            "available_tools": list(self.available_tools.keys())
        }

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