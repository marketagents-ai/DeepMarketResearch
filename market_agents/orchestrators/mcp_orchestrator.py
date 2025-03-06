
from datetime import datetime, timezone
import importlib
import logging
from typing import List, Dict, Any, Optional, Type, Union
import inspect

from pydantic import BaseModel, Field

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import MCPServerConfig, OrchestratorConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter, serialize_for_json
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor
from market_agents.environments.environment import EnvironmentStep, GlobalAction, LocalEnvironmentStep, StrAction
from market_agents.environments.mechanisms.mcp_server import (
    MCPServerEnvironment,
    MCPServerMechanism,
    MCPServerActionSpace,
    MCPToolAction
)
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.orchestrators.logger_utils import log_action, log_perception, log_persona, log_reflection
from minference.lite.models import CallableMCPTool

logger = logging.getLogger(__name__)

class MCPOrchestrator(BaseEnvironmentOrchestrator):
    """Orchestrator for MCP server environments"""
    
    def __init__(
        self,
        config: MCPServerConfig,
        agents: List[MarketAgent],
        storage_service: StorageService,
        orchestrator_config: OrchestratorConfig,
        logger=None,
        ai_utils=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            orchestrator_config=orchestrator_config,
            agents=agents,
            storage_service=storage_service,
            logger=logger,
            environment_name=config.name,
            ai_utils=ai_utils,
            **kwargs
        )
        
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        self.cognitive_processor = ParallelCognitiveProcessor(
            ai_utils=self.ai_utils,
            storage_service=storage_service,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )
        
        # Determine the server path from the module path
        server_module_path = self.config.mcp_server_module.replace('.', '/')
        server_path = f"{server_module_path}.py"
        
        # Check if the file exists
        import os
        if not os.path.exists(server_path):
            # Try to find the actual path
            import importlib.util
            import sys
            
            spec = importlib.util.find_spec(self.config.mcp_server_module)
            if spec and spec.origin:
                server_path = spec.origin
                self.logger.info(f"Found server script at: {server_path}")
            else:
                self.logger.error(f"Could not find server script for module: {self.config.mcp_server_module}")
                raise ValueError(f"Server script not found for {self.config.mcp_server_module}")
        
        # Create the MCP server mechanism with the server path
        mechanism = MCPServerMechanism(
            server_path=server_path,
            max_rounds=self.config.sub_rounds
        )
                
        self.environment = MCPServerEnvironment(
            name=self.config.name,
            mechanism=mechanism
        )
        
        # Register the environment with each agent
        for agent in self.agents:
            agent.environments[self.config.name] = self.environment
            agent.task = f"Interact with the MCP server: {self.config.task_prompt}"
            agent._refresh_prompts()
        
        self.logger.info(f"Initialized MCPOrchestrator for environment: {self.config.name}")
    
    async def setup_environment(self):
        """Setup or reset the MCP server environment."""
        self.logger.info("Setting up the MCP Server Environment...")
        await self.environment.initialize()
        self.environment.reset()
        self.logger.info("MCP Server Environment setup complete.")
    
    async def run_environment(self, round_num: int = None):
        """Run the environment for specified rounds."""
        if round_num is not None:
            await self.run_mcp_round(round_num)
        else:
            for r in range(1, self.orchestrator_config.max_rounds + 1):
                await self.run_mcp_round(r)
    
    async def run_mcp_round(self, round_num: int):
        """Orchestrates a single main round with multiple sub-rounds of MCP server interaction."""
        self.logger.info(f"=== Running MCP Server Round {round_num} ===")
        # Run each sub-round
        for sub_round in range(1, self.config.sub_rounds + 1):
            self.logger.info(f"=== Starting Sub-round {sub_round}/{self.config.sub_rounds} of Round {round_num} ===")

            try:
                # Run perception phase
                perceptions = await self._run_perception_phase(round_num, sub_round)
                
                # Run action phase
                step_result = await self._run_action_phase(round_num, sub_round)
                
                # Run reflection phase
                reflections = await self._run_reflection_phase(round_num, sub_round)
                
                # Process and store results
                await self.process_round_results(round_num, step_result, sub_round)
            except Exception as e:
                self.logger.error(f"Error in round {round_num}, sub-round {sub_round}: {e}")
                self.logger.exception("Sub-round failed")
                raise
        
        self.logger.info(f"Round {round_num} complete with {self.config.sub_rounds} sub-rounds.\n")
    
    async def _run_perception_phase(self, round_num: int, sub_round: int):
        """Handles the perception phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents perceiving MCP server state...")
        perceptions = await self.cognitive_processor.run_parallel_perception(
            self.agents, 
            self.config.name
        )
        
        for agent, perception in zip(self.agents, perceptions or []):
            log_persona(self.logger, agent.id, agent.persona)
            log_perception(
                self.logger, 
                agent.id, 
                perception.json_object.object if perception and perception.json_object else None
            )
        
        return perceptions
    
    async def _run_action_phase(self, round_num: int, sub_round: int):
        """Handles the action phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Executing agent actions with MCP server...")
        
        try:
            # Get agent actions
            actions = await self.cognitive_processor.run_parallel_action(
                self.agents,
                self.config.name
            )
            
            # Process actions into global actions
            agent_results = await self._process_agent_actions(actions)
            self.logger.info(f"Processed action results: {agent_results}")
            
            # Create global action object
            global_action = GlobalAction(actions=agent_results)
            
            # Execute step and await the result
            step_result = await self.environment.step(global_action)
            
            # Update agent observations if we have results
            if step_result:
                if isinstance(step_result, EnvironmentStep) and step_result.global_observation:
                    await self._update_agent_observations(step_result)
                elif isinstance(step_result, LocalEnvironmentStep) and step_result.observation:
                    # Handle local step result
                    for agent in self.agents:
                        if step_result.observation.agent_id == agent.id:
                            agent.last_observation = step_result.observation
            
            return step_result
            
        except Exception as e:
            self.logger.error(f"Error in action phase: {str(e)}")
            raise

    async def _update_agent_observations(self, step_result):
        """Update agent observations based on step results."""
        try:
            if isinstance(step_result, LocalEnvironmentStep):
                # Handle local step result
                for agent in self.agents:
                    if step_result.observation.agent_id == agent.id:
                        agent.last_observation = step_result.observation
                        break
            elif isinstance(step_result, EnvironmentStep) and step_result.global_observation:
                # Handle global step result
                for agent in self.agents:
                    if agent.id in step_result.global_observation.observations:
                        agent.last_observation = step_result.global_observation.observations[agent.id]
        except Exception as e:
            self.logger.error(f"Error updating agent observations: {str(e)}")
    
    async def _process_agent_actions(self, actions):
        """Process individual agent actions and create global actions."""
        global_actions = {}
        
        for agent, action in zip(self.agents, actions or []):
            try:
                # Debug the action object
                print(f"Action for agent {agent.id}: {type(action)}")
                if hasattr(action, 'tool_calls') and action.tool_calls:
                    print(f"Tool calls: {action.tool_calls}")
                    # Handle tool calls
                    tool_call = action.tool_calls[0]  # Take the first tool call
                    global_actions[agent.id] = MCPToolAction(
                        agent_id=agent.id,
                        tool_name=tool_call.name,
                        tool_input=tool_call.arguments
                    )
                    agent.last_action = {
                        "tool_name": tool_call.name,
                        "tool_input": tool_call.arguments
                    }
                    log_action(self.logger, agent.id, f"Tool: {tool_call.name}, Args: {tool_call.arguments}")
                elif hasattr(action, 'content') and action.content:
                    # Handle text content
                    content = action.content
                    agent.last_action = content
                    global_actions[agent.id] = StrAction(
                        agent_id=agent.id,
                        action=content
                    )
                    log_action(self.logger, agent.id, content)
                elif hasattr(action, 'json_object') and action.json_object and action.json_object.object:
                    # Handle structured JSON output
                    content = action.json_object.object
                    if isinstance(content, dict) and 'tool_name' in content and 'tool_input' in content:
                        global_actions[agent.id] = MCPToolAction(
                            agent_id=agent.id,
                            tool_name=content['tool_name'],
                            tool_input=content['tool_input']
                        )
                    else:
                        # Default to string action if structure doesn't match
                        global_actions[agent.id] = StrAction(
                            agent_id=agent.id,
                            action=str(content)
                        )
                    agent.last_action = content
                    log_action(self.logger, agent.id, content)
                else:
                    print(f"Unhandled action type: {action}")
                    global_actions[agent.id] = StrAction(
                        agent_id=agent.id,
                        action=""
                    )
                    agent.last_action = ""
            except Exception as e:
                self.logger.error(f"Error creating global action for agent {agent.id}: {str(e)}")
                global_actions[agent.id] = StrAction(
                    agent_id=agent.id,
                    action=""
                )
        
        return global_actions
    
    async def _create_global_actions(self, actions) -> Dict[str, Union[MCPToolAction, StrAction]]:
        """Create global actions from agent actions."""
        global_actions = {}
        
        for agent, action in zip(self.agents, actions or []):
            try:
                if not action:
                    continue
                
                if isinstance(action, dict) and 'tool_name' in action and 'tool_input' in action:
                    # Handle tool call format
                    global_actions[agent.id] = MCPToolAction(
                        agent_id=agent.id,
                        tool_name=action['tool_name'],
                        tool_input=action['tool_input']
                    )
                elif isinstance(action, str):
                    # Handle string actions
                    global_actions[agent.id] = StrAction(
                        agent_id=agent.id,
                        action=action
                    )
                # Handle OpenAI tool calls format
                elif hasattr(action, 'tool_calls') and action.tool_calls:
                    tool_call = action.tool_calls[0]  # Take the first tool call
                    global_actions[agent.id] = MCPToolAction(
                        agent_id=agent.id,
                        tool_name=tool_call.function.name,
                        tool_input=tool_call.function.arguments
                    )
                elif hasattr(action, 'json_object') and action.json_object and action.json_object.object:
                    # Handle structured JSON output
                    content = action.json_object.object
                    if isinstance(content, dict) and 'tool_name' in content and 'tool_input' in content:
                        global_actions[agent.id] = MCPToolAction(
                            agent_id=agent.id,
                            tool_name=content['tool_name'],
                            tool_input=content['tool_input']
                        )
                    else:
                        # Default to string action if structure doesn't match
                        global_actions[agent.id] = StrAction(
                            agent_id=agent.id,
                            action=str(content)
                        )
            except Exception as e:
                self.logger.error(f"Error creating global action for agent {agent.id}: {str(e)}")
                global_actions[agent.id] = StrAction(
                    agent_id=agent.id,
                    action=""
                )
        
        return global_actions
    
    async def _run_reflection_phase(self, round_num: int, sub_round: int):
        """Run parallel reflection for all agents."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents reflecting on MCP server interactions...")
        
        # Only run reflection for agents that have observations
        agents_with_observations = [agent for agent in self.agents if agent.last_observation]
        
        if not agents_with_observations:
            self.logger.warning("No agents had observations to reflect on")
            return
        
        reflections = await self.cognitive_processor.run_parallel_reflection(
            agents_with_observations,
            self.config.name
        )
        
        if not reflections:
            self.logger.warning("No reflections received from agents")
            return
        
        for agent, reflection in zip(agents_with_observations, reflections):
            try:
                if reflection and hasattr(reflection, 'json_object') and reflection.json_object and reflection.json_object.object:
                    content = reflection.json_object.object
                elif hasattr(reflection, 'str_content'):
                    content = reflection.str_content
                else:
                    content = str(reflection)
                
                if content:
                    log_reflection(self.logger, agent.id, content)
                else:
                    self.logger.warning(f"No reflection content for agent {agent.id}")
            except Exception as e:
                self.logger.error(f"Error processing reflection for agent {agent.id}: {str(e)}")
    
    async def process_round_results(self, round_num: int, step_result=None, sub_round: int = None):
        """Process and store results from the round."""
        try:
            # Store environment state
            if step_result:
                await self.data_inserter.insert_environment_state(
                    environment_name=self.config.name,
                    round_num=round_num,
                    state_data={
                        'step_result': serialize_for_json(step_result),
                        'global_state': self.environment.get_global_state()
                    },
                    metadata={
                        'sub_round': sub_round,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            # Store agent actions
            for agent in self.agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    await self.data_inserter.insert_agent_action(
                        agent_id=agent.id,
                        environment_name=self.config.name,
                        round_num=round_num,
                        action_data=serialize_for_json(agent.last_action),
                        metadata={
                            'sub_round': sub_round,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    )
        except Exception as e:
            self.logger.error(f"Error processing round results: {str(e)}")
    
    async def get_round_summary(self, round_num: int) -> dict:
        """Get summary of the specified round."""
        return {
            "round": round_num,
            "environment": self.config.name,
            "global_state": self.environment.get_global_state()
        }
    
    async def print_summary(self):
        """Print final summary of MCP server interactions."""
        self.logger.info("\n=== MCP SERVER INTERACTION SUMMARY ===")
        
        global_state = self.environment.get_global_state()
        self.logger.info(f"Final Environment State: {global_state}")
        
        for agent in self.agents:
            self.logger.info(f"\nAgent {agent.id} final state:")
            if agent.last_action:
                self.logger.info(f"Last action: {agent.last_action}")