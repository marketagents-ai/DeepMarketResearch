from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from market_agents.environments.mechanisms.research import ResearchAction
from market_agents.web_search.url_processor import URLFetcher
from market_agents.web_search.web_search_config import WebSearchConfig
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
from minference.lite.models import CallableTool, StructuredTool
from minference.caregistry import CallableRegistry

from market_agents.web_search.web_search_manager import SearchManager
from market_agents.web_search.content_extractor import ContentExtractor

logger = logging.getLogger(__name__)
CallableRegistry._logger = logger

class WebSearchResult(BaseModel):
    """Structure for a single search result"""
    url: str
    title: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class WebSearchLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: Dict[str, Any]
    status: str = "pending"
    search_results: Optional[List[Dict[str, str]]] = None

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation
        return d

class WebSearchGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, WebSearchLocalObservation]

class WebSearchMechanism(Mechanism):
    """Mechanism that manages web search workflow"""
    search_manager: Optional[SearchManager] = Field(default=None, exclude=True)
    content_extractor: Optional[ContentExtractor] = Field(default=None, exclude=True)
    url_fetcher: Optional[URLFetcher] = Field(default=None, exclude=True)
    current_round: int = Field(default=0, description="Current search round")
    max_rounds: int = Field(default=3, description="Maximum search rounds")
    search_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of search results"
    )
    current_query: str = ""

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data):
        super().__init__(**data)
        if not isinstance(data.get("search_config"), WebSearchConfig):
            raise ValueError("search_config must be a WebSearchConfig instance")
            
        self.search_config = data["search_config"]
        self.search_manager = SearchManager(config=self.search_config)
        self.content_extractor = ContentExtractor(config=self.search_config)
        self.url_fetcher = URLFetcher(config=self.search_config, prompts={})
    
    def step(
        self,
        action: Union[GlobalAction, str]
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Process agent actions in workflow sequence"""
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
                
                obs = WebSearchLocalObservation(
                    agent_id=agent_id,
                    observation=obs_data,
                    status="success",
                    search_results=[]
                )
                observations[agent_id] = obs

            step_result = EnvironmentStep(
                global_observation=GlobalObservation(observations=observations),
                reward=1.0,
                done=done,
                info={
                    "round": self.current_round,
                    "actions": {k: str(v) for k, v in action.actions.items()}
                }
            )
            self.last_step = step_result
            return step_result

        return LocalEnvironmentStep(
            observation=WebSearchLocalObservation(
                agent_id="default",
                observation={"action": str(action), "round": self.current_round},
                status="success",
                search_results=[]
            ),
            reward=1.0,
            done=done,
            info={"round": self.current_round}
        )

    def get_global_state(self) -> Dict[str, Any]:
        """Get current global state"""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "search_history": self.search_history,
            "current_query": self.current_query
        }

    def reset(self) -> None:
        """Reset mechanism state"""
        self.current_round = 0
        self.search_history.clear()
        self.current_query = ""

class WebResearchActionSpace(ActionSpace):
    """Action space that handles both web search and research summary actions"""
    summary_model: Optional[Type[BaseModel]] = None
    mechanism: WebSearchMechanism = Field(
        ..., 
        description="Mechanism that handles web search operations"
    )
    workflow: bool = Field(
        default=True,
        description="Whether tools should be executed sequentially as a workflow"
    )

    def __init__(self, mechanism: WebSearchMechanism, summary_model: Type[BaseModel] = None, workflow:bool = True, **data):
        data.update({
            "mechanism": mechanism,
            "workflow": workflow
        })
        super().__init__(**data)

        print(f"WebResearchActionSpace initialized with workflow={workflow}")  # Debug
        
        self.summary_model = summary_model
        
        # Create web search tool
        web_search_tool = CallableTool.from_callable(
            func=self.execute_web_search,
            name="web_search",
            docstring="Execute web search and return results",
            strict_schema=True
        )
        
        # Create summary tool based on model or string action
        if summary_model:
            summary_tool = StructuredTool(
                json_schema=summary_model.model_json_schema(),
                name="research_summary",
                description="Generate research summary from search results"
            )
        else:
            summary_tool = StructuredTool(
                json_schema=StrAction.model_json_schema(),
                name="text_summary",
                description="Generate text summary from search results"
            )

        # Both tools defined in action space
        self.allowed_actions = [
            web_search_tool,
            summary_tool
        ]

    async def execute_web_search(
        self,
        query: str,
        num_results: Optional[int] = None
    ) -> List[WebSearchResult]:
        """Execute web search and return results"""
        try:
            self.mechanism.current_query = query
            
            urls = await self.mechanism.search_manager.get_urls_for_query(
                query,
                self.mechanism.search_config.urls_per_query)
            
            for url in urls:
                self.mechanism.search_manager.query_url_mapping[url] = query
            
            fetched_results = await self.mechanism.url_fetcher.process_urls(urls, self.mechanism.search_manager.query_url_mapping)
            
            search_results = [
                {
                    "url": fr.url,
                    "title": fr.title,
                    "content": fr.content.get('text', ''),
                    "timestamp": datetime.now().isoformat()
                }
                for fr in fetched_results if fr is not None
            ]
            
            return search_results
                
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
        
    def get_action_schema(self) -> Dict[str, Any]:
        """Return JSON schema for both tools"""
        return {
            tool.name: tool.json_schema() 
            for tool in self.allowed_actions
        }
    
class WebSearchEnvironment(MultiAgentEnvironment):
    """Environment that manages web search operations"""
    name: str = Field(
        default="Web Search Environment",
        description="Name of the environment"
    )
    mechanism: WebSearchMechanism = Field(
        ...,
        description="Mechanism that handles web search operations"
    )
    action_space: WebResearchActionSpace = None

    internal_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Internal storage for global state"
    )
    summary_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Optional Pydantic model for structuring research summaries"
    )
    initial_query: str = Field(
        ...,
        description="Initial search query to start the research with"
    )

    def __init__(
        self,
        name: str = "Web Search Environment",
        initial_query: str = None,
        summary_model: Optional[Type[BaseModel]] = None,
        mechanism: Optional[WebSearchMechanism] = None,
        **data
    ):
        """Initialize the web search environment."""
        if not initial_query:
            raise ValueError("initial_query must be provided")

        model_data = {
            "name": name,
            "initial_query": initial_query,
            "summary_model": summary_model,
            "mechanism": mechanism,
            **data
        }
        super().__init__(**model_data)
        
        self.action_space = WebResearchActionSpace(
            mechanism=self.mechanism,
            summary_model=summary_model,
            workflow=True
        )
        
        self.internal_state = {}
        
        if hasattr(self.mechanism, 'current_query'):
            self.mechanism.current_query = initial_query

    def get_global_state(self) -> Dict[str, Any]:
        """Get current global state combining mechanism and environment state"""
        mechanism_state = self.mechanism.get_global_state()
        
        summary_schema = None
        if self.summary_model:
            try:
                summary_schema = self.summary_model.model_json_schema()
            except Exception as e:
                logger.error(f"Error getting summary model schema: {e}")
        
        return {
            **self.internal_state,
            **mechanism_state,
            "initial_query": self.initial_query,
            "summary_model": self.summary_model.__name__ if self.summary_model else None,
            "summary_schema": summary_schema
        }

    def reset(self) -> GlobalObservation:
        """Reset environment state and restore initial query"""
        self.internal_state = {}
        if hasattr(self.mechanism, 'current_query'):
            self.mechanism.current_query = self.initial_query
        self.mechanism.reset()
        return GlobalObservation(observations={})