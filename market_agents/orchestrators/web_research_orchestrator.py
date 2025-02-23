from datetime import datetime, timezone
import importlib
import logging
from typing import List, Dict, Any, Type, Union

from pydantic import BaseModel

from market_agents.environments.mechanisms.research import ResearchAction
from market_agents.web_search.web_search_config import WebSearchConfig
from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig, WebResearchConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter, serialize_for_json
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor
from market_agents.environments.environment import EnvironmentStep, GlobalAction, LocalEnvironmentStep, StrAction
from market_agents.environments.mechanisms.web_research import (
    WebSearchEnvironment,
    WebSearchLocalObservation,
    WebSearchMechanism
)
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.orchestrators.logger_utils import log_action, log_perception, log_persona, log_reflection

logger = logging.getLogger(__name__)

class WebResearchOrchestrator(BaseEnvironmentOrchestrator):
    def __init__(
        self,
        config: WebResearchConfig,
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

        self.summary_model = self.get_schema_model(self.config.schema_model) if self.config.schema_model else None
        self.logger.info(f"Loaded schema model: {self.summary_model}")
        
        mechanism = WebSearchMechanism(
            search_config=WebSearchConfig.from_yaml(self.config.search_config),
            max_rounds=self.config.sub_rounds,
            current_query=self.config.initial_query
        )
        
        self.environment = WebSearchEnvironment(
            name=self.config.name,
            initial_query=self.config.initial_query,
            summary_model=self.summary_model,
            mechanism=mechanism
        )

        for agent in self.agents:
            agent.environments[self.config.name] = self.environment
            agent.task = f"Research the following topic using web search:\n{self.config.initial_query}"
            agent._refresh_prompts()

        self.logger.info(f"Initialized WebResearchOrchestrator for environment: {self.config.name}")
    async def setup_environment(self):
        """Setup or reset the web search environment."""
        self.logger.info("Setting up the Web Research Environment...")
        self.environment.reset()
        self.logger.info("Web Research Environment setup complete.")

    async def run_environment(self, round_num: int = None):
        """Run the environment for specified rounds."""
        if round_num is not None:
            await self.run_research_round(round_num)
        else:
            for r in range(1, self.orchestrator_config.max_rounds + 1):
                await self.run_research_round(r)

    async def run_research_round(self, round_num: int):
        """Orchestrates a single main round with multiple sub-rounds of web search."""
        self.logger.info(f"=== Running Web Research Round {round_num} ===")

        # Initialize agents for this round
        self._initialize_agents_for_round()

        # Run each sub-round
        for sub_round in range(1, self.config.sub_rounds + 1):
            self.logger.info(f"=== Starting Sub-round {sub_round}/{self.config.sub_rounds} of Round {round_num} ===")
            try:
                step_result = await self._run_sub_round(round_num, sub_round)
                await self.process_round_results(round_num, step_result, sub_round)
            except Exception as e:
                self.logger.error(f"Error in round {round_num}, sub-round {sub_round}: {e}")
                self.logger.exception("Sub-round failed")
                raise

        self.logger.info(f"Round {round_num} complete with {self.config.sub_rounds} sub-rounds.\n")

    def _initialize_agents_for_round(self):
        """Initialize agents with the current search query."""
        for agent in self.agents:
            agent.task = f"Research the following topic using web search:\n{self.environment.mechanism.current_query}"
            agent._refresh_prompts()

    async def _run_sub_round(self, round_num: int, sub_round: int):
        """Executes a single sub-round with both search and summary phases"""
        try:
            # 1. Perception Phase
            perceptions = await self._run_perception_phase(round_num, sub_round)
            
            # 2. Search Phase
            self.environment.switch_phase("search")
            search_result = await self._run_action_phase(round_num, sub_round, "search")
            
            # 3. Summary Phase
            self.environment.switch_phase("summary")
            summary_result = await self._run_action_phase(round_num, sub_round, "summary")
            
            # 4. Reflection Phase
            reflections = await self._run_reflection_phase(round_num, sub_round)
            
            return summary_result
            
        except Exception as e:
            self.logger.error(f"Error in sub-round {sub_round} of round {round_num}: {e}")
            raise

    async def _run_perception_phase(self, round_num: int, sub_round: int):
        """Handles the perception phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents perceiving environment...")
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

    async def _run_action_phase(self, round_num: int, sub_round: int, phase: str):
        """Handles the action phase of the cognitive cycle for either search or summary."""
        self.logger.info(f"Round {round_num}.{sub_round}: Executing agent {phase}...")
        
        actions = await self.cognitive_processor.run_parallel_action(
            self.agents,
            self.config.name
        )
        
        agent_results = await self._process_agent_actions(actions)
        self.logger.info(f"Processed {phase} results: {agent_results}")
        
        global_actions = await self._create_global_actions(actions, phase)
        
        step_result = self.environment.step(GlobalAction(actions=global_actions))
        
        if step_result and step_result.global_observation:
            await self._update_agent_observations(step_result)
        
        return step_result

    async def _create_global_actions(self, actions, phase: str) -> Dict[str, Union[ResearchAction, StrAction]]:
        """Create global actions from individual agent actions."""
        global_actions = {}
        
        for agent, action in zip(self.agents, actions or []):
            try:
                if phase == "summary":
                    if self.summary_model:
                        try:
                            if action and action.json_object and action.json_object.object:
                                content = action.json_object.object
                                validated_content = self.summary_model.model_validate(content)
                                global_actions[agent.id] = ResearchAction(
                                    agent_id=agent.id,
                                    action=validated_content
                                )
                            else:
                                empty_content = self.summary_model.model_construct()
                                global_actions[agent.id] = ResearchAction(
                                    agent_id=agent.id,
                                    action=empty_content
                                )
                        except Exception as e:
                            self.logger.error(f"Error creating ResearchAction: {e}")
                            empty_content = self.summary_model.model_construct()
                            global_actions[agent.id] = ResearchAction(
                                agent_id=agent.id,
                                action=empty_content
                            )
                    else:
                        content = ""
                        if action and action.content:
                            content = action.content
                        elif action and action.json_object and action.json_object.object:
                            content = str(action.json_object.object)
                        elif isinstance(action, str):
                            content = action
                        
                        global_actions[agent.id] = StrAction(
                            agent_id=agent.id,
                            action=content
                        )
                        
            except Exception as e:
                self.logger.error(f"Error creating global action for agent {agent.id}: {str(e)}")
                if phase == "summary":
                    if self.summary_model:
                        empty_content = self.summary_model.model_construct()
                        global_actions[agent.id] = ResearchAction(
                            agent_id=agent.id,
                            action=empty_content
                        )
                    else:
                        global_actions[agent.id] = StrAction(
                            agent_id=agent.id,
                            action=""
                        )
        
        return global_actions

    async def _process_agent_actions(self, actions):
        """Process individual agent actions and create summaries."""
        agent_summaries = {}
        
        for agent, action in zip(self.agents, actions or []):
            try:
                content = None
                if action:
                    if isinstance(action, str):
                        content = action
                    elif action.content and not action.json_object:
                        content = action.content
                    elif action.json_object and action.json_object.object:
                        content = action.json_object.object

                agent.last_action = content
                if content:
                    agent_summaries[agent.id] = content
                
                model_name = agent.llm_config.model if agent.llm_config else None
                log_action(self.logger, agent.id, content, model_name=model_name)

            except Exception as e:
                self.logger.error(f"Error processing action for agent {agent.id}: {str(e)}", exc_info=True)
                agent.last_action = None
        
        return agent_summaries

    async def _update_agent_observations(self, step_result):
        """Update agent observations based on step results."""
        if not step_result:
            return
            
        for agent in self.agents:
            try:
                if isinstance(step_result, LocalEnvironmentStep):
                    if step_result.observation.agent_id == agent.id:
                        agent.last_observation = step_result.observation
                elif isinstance(step_result, EnvironmentStep) and step_result.global_observation:
                    if agent.id in step_result.global_observation.observations:
                        agent.last_observation = step_result.global_observation.observations[agent.id]
                
            except Exception as e:
                self.logger.error(f"Error updating observation for agent {agent.id}: {str(e)}")
                agent.last_observation = None

    async def _run_reflection_phase(self, round_num: int, sub_round: int):
        """Run parallel reflection for all agents."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents reflecting on search results...")
        reflections = await self.cognitive_processor.run_parallel_reflection(
            self.agents,
            self.config.name
        )

    async def process_round_results(self, round_num: int, step_result=None, sub_round: int = None):
        """Process and store results from the round."""
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

    async def get_round_summary(self, round_num: int) -> dict:
        """Get summary of the specified round."""
        return {
            "round": round_num,
            "environment": self.config.name,
            "global_state": self.environment.get_global_state()
        }
    
    def get_schema_model(self, schema_name: str) -> Type[BaseModel]:
        """Load the summary schema model class"""
        try:
            schemas_module = importlib.import_module('market_agents.orchestrators.research_schemas')
            model_class = getattr(schemas_module, schema_name)
            return model_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load schema model '{schema_name}': {e}")

    async def print_summary(self):
        """Print final summary of web research results."""
        self.logger.info("\n=== WEB RESEARCH SUMMARY ===")
        
        global_state = self.environment.get_global_state()
        self.logger.info(f"Final Environment State: {global_state}")
        
        for agent in self.agents:
            self.logger.info(f"\nAgent {agent.id} final search results:")
            if agent.last_observation and isinstance(agent.last_observation, WebSearchLocalObservation):
                if isinstance(agent.last_observation.search_results, list):
                    for result in agent.last_observation.search_results:
                        if isinstance(result, dict):
                            self.logger.info(f"- {result.get('title', 'No title')}: {result.get('url', 'No URL')}")
                        elif hasattr(result, 'title') and hasattr(result, 'url'):
                            self.logger.info(f"- {result.title}: {result.url}")