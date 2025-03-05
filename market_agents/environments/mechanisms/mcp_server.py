from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

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

class MCPServerMechanism(Mechanism):
    """Mechanism that manages MCP server tool interactions"""
    mcp_server: Any = Field(default=None, exclude=True)
    current_round: int = Field(default=0, description="Current interaction round")
    max_rounds: int = Field(default=10, description="Maximum interaction rounds")
    tool_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of tool invocations"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data):
        super().__init__(**data)
        if not data.get("mcp_server"):
            raise ValueError("mcp_server must be provided")
            
        self.mcp_server = data["mcp_server"]
        # Extract available tools from the MCP server
        self.available_tools = self._extract_tools_from_server()
    
    def _extract_tools_from_server(self):
        """Extract available tools from the MCP server using the official MCP API"""
        tools = {}
        
        print(f"MCP Server type: {type(self.mcp_server)}")
        
        # Use the official list_tools method to get tool schemas
        if hasattr(self.mcp_server, "list_tools") and callable(self.mcp_server.list_tools):
            try:
                # Get the current event loop
                import asyncio
                loop = asyncio.get_event_loop()
                
                # Check if the loop is already running
                if loop.is_running():
                    print("Event loop is already running, using direct method access")
                    
                    # Try to access tools directly from the MCP server
                    if hasattr(self.mcp_server, "_tool_manager"):
                        tool_manager = self.mcp_server._tool_manager
                        print(f"Found tool manager: {type(tool_manager)}")
                        
                        # Try to access tools from the tool manager
                        if hasattr(tool_manager, "_tools") and tool_manager._tools:
                            for name, tool_obj in tool_manager._tools.items():
                                # Print tool object details for debugging
                                print(f"Tool object for {name}: {type(tool_obj)}")
                                
                                # Extract tool info
                                description = getattr(tool_obj, "description", f"Execute {name}")

                                # Get the function directly from the 'fn' attribute
                                func = None
                                if hasattr(tool_obj, "fn"):
                                    func = getattr(tool_obj, "fn")
                                    print(f"function: {func} for fn")
                                    if callable(func):
                                        print(f"Found callable function in attribute 'fn'")
                                        import inspect
                                        print(f"Function signature: {inspect.signature(func)}")
                                        print(f"Function is callable: {callable(func)}")
                                    else:
                                        print(f"'fn' attribute exists but is not callable")

                                # Store tool info
                                tools[name] = {
                                    "name": name,
                                    "description": description,
                                    "func": func
                                }
                                
                                # Print function info for debugging
                                if func:
                                    import inspect
                                    print(f"Function signature: {inspect.signature(func)}")
                                    print(f"Function is callable: {callable(func)}")
                                else:
                                    print(f"No callable function found for tool {name}")
                                
                                print(f"Found tool: {name}")
                    else:
                        # Create a coroutine for list_tools
                        coro = self.mcp_server.list_tools()
                        
                        # Execute the coroutine
                        tool_list_response = loop.run_until_complete(coro)
                        
                        if hasattr(tool_list_response, 'tools'):
                            tool_list = tool_list_response.tools
                            print(f"Found {len(tool_list)} tools using list_tools()")
                            
                            # Store tool schemas for each tool
                            for tool_info in tool_list:
                                tool_name = tool_info.name
                                tool_description = tool_info.description
                                
                                # Store the tool info in our tools dictionary
                                tools[tool_name] = {
                                    "name": tool_name,
                                    "description": tool_description
                                }
                                print(f"Found tool: {tool_name}")
                        else:
                            print("Tool list response doesn't have 'tools' attribute")
            except Exception as e:
                print(f"Error accessing tools through list_tools: {str(e)}")
        else:
            print("MCP Server doesn't have list_tools method")
        
        print(f"Total tools found: {len(tools)}")
        if tools:
            print(f"Tool names: {list(tools.keys())}")
        else:
            print("No tools found in MCP server")
        
        return tools
    
    def step(
        self,
        action: Union[GlobalAction, str]
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Process agent actions and execute MCP server tools"""
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
                
                # Execute the tool if it's a valid tool request
                if hasattr(agent_action, 'tool_name') and agent_action.tool_name in self.available_tools:
                    try:
                        tool_func = self.available_tools[agent_action.tool_name]
                        tool_args = agent_action.tool_args if hasattr(agent_action, 'tool_args') else {}
                        
                        # Execute the tool
                        result = tool_func(**tool_args)
                        
                        # Record the tool execution
                        tool_result = MCPServerResult(
                            tool_name=agent_action.tool_name,
                            result=result
                        )
                        
                        # Add to history
                        self.tool_history.append({
                            "agent_id": agent_id,
                            "tool_name": agent_action.tool_name,
                            "tool_args": tool_args,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Create observation with tool result
                        obs_data["tool_result"] = tool_result.model_dump()
                        
                    except Exception as e:
                        obs_data["status"] = "error"
                        obs_data["error"] = str(e)
                
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
                global_observation=global_obs,
                done=done,
                info={"round": self.current_round, "max_rounds": self.max_rounds}
            )
        
        elif isinstance(action, LocalAction):
            # Handle single agent action
            agent_id = action.agent_id
            obs_data = {
                "action": action.model_dump() if hasattr(action, 'model_dump') else str(action),
                "round": self.current_round,
                "status": "success"
            }
            
            # Execute the tool if it's a valid tool request
            if hasattr(action, 'tool_name') and action.tool_name in self.available_tools:
                try:
                    tool_func = self.available_tools[action.tool_name]
                    tool_args = action.tool_args if hasattr(action, 'tool_args') else {}
                    
                    # Execute the tool
                    result = tool_func(**tool_args)
                    
                    # Record the tool execution
                    tool_result = MCPServerResult(
                        tool_name=action.tool_name,
                        result=result
                    )
                    
                    # Add to history
                    self.tool_history.append({
                        "agent_id": agent_id,
                        "tool_name": action.tool_name,
                        "tool_args": tool_args,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Create observation with tool result
                    obs_data["tool_result"] = tool_result.model_dump()
                    
                except Exception as e:
                    obs_data["status"] = "error"
                    obs_data["error"] = str(e)
            
            # Create the local observation
            local_obs = MCPServerLocalObservation(
                agent_id=agent_id,
                observation=obs_data,
                status=obs_data["status"]
            )
            
            # Return local environment step
            return LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={"round": self.current_round, "max_rounds": self.max_rounds}
            )
        
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get the current global state of the mechanism"""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "tool_history": self.tool_history,
            "available_tools": list(self.available_tools.keys())
        }

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
                if "func" in tool_info and tool_info["func"] is not None:
                    # Create the tool from the function
                    mcp_tool = CallableMCPTool.from_callable(
                        func=tool_info["func"],
                        name=tool_name,
                        docstring=tool_info.get("description", ""),
                        strict_schema=False
                    )
                else:
                    # Skip this tool since we don't have a callable function
                    print(f"Skipping tool {tool_name} - no callable function provided")
                    continue
                
                # Set the MCP server
                mcp_tool.mcp_server = mechanism.mcp_server
                
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
        
        self.action_space = MCPServerActionSpace(
            mechanism=self.mechanism
        )
        
        self.internal_state = {}
    
    def get_global_state(self):
        """Get current global state combining mechanism and environment state"""
        state = self.mechanism.get_global_state()
        state.update(self.internal_state)
        return state
    
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
