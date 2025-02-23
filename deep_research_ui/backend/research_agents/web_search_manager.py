import asyncio
import json
import logging
import os
import sys
import random
import threading
import uuid
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from tools_manager import ToolsManager
import yaml
from googlesearch import search
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext, StructuredTool
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.research_agents.utils import parse_ai_response, clean_json_string, load_config
from market_agents.research_agents.research_schemas import SearchQueries



# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from market_agents.research_agent_with_ui.backend.toolsmanager import ToolsManager


# Set up logging
logger = logging.getLogger(__name__)
class WebSearchConfig(BaseSettings):
    query: str = "default search query"
    max_concurrent_requests: int = 50
    rate_limit: float = 0.1
    content_max_length: int = 1000000
    request_timeout: int = 30
    urls_per_query: int = 1
    use_ai_summary: bool = True 
    methods: List[str] = ["selenium", "playwright", "beautifulsoup", "newspaper3k", "scrapy", 
                         "requests_html", "mechanicalsoup", "httpx"]
    default_method: str = "newspaper3k"
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    llm_configs: Dict[str, Dict[str, Any]]
    tools_storage_path: Optional[str] = None

    model_config = SettingsConfigDict(extra='forbid')


    
class SearchManager:
    def __init__(self, ai_utils, config: WebSearchConfig, prompts: Dict):
        self.ai_utils = ai_utils
        self.config = config
        self.prompts = prompts
        self.last_request_time = 0
        self.request_delay = 5
        self.max_retries = 3
        self.headers = config.headers
        self.search_params = {
            'num': self.config.urls_per_query,
            'stop': self.config.urls_per_query,
            'pause': 2.0,
            'user_agent': self.headers['User-Agent']
        }
        self.query_url_mapping = {}
        self.tools_manager = ToolsManager(config.tools_storage_path)
    def add_custom_tool(self, tool_name: str, tool_config: Dict[str, Any]):
        """Add a new custom tool"""
        self.tools_manager.add_tool(tool_name, tool_config)

    def remove_custom_tool(self, tool_name: str):
        """Remove a custom tool"""
        self.tools_manager.remove_tool(tool_name)

    def get_custom_tools(self) -> Dict[str, Any]:
        """Get all custom tools"""
        return self.tools_manager.get_tools()
        
    async def generate_search_queries(self, base_query: str) -> List[str]:
        try:
            # Get config from llm_configs
            llm_config_dict = self.config.llm_configs["search_query_generation"].copy()
            
            # Parse the YAML content properly
            search_query_section = yaml.safe_load(self.prompts["search_query_generation"])
            
            # Get system prompt and template from parsed YAML
            system_prompt = search_query_section["system_prompt"]
            prompt_template = search_query_section["prompt_template"]
            
            # Remove non-LLMConfig fields before creating LLMConfig
            llm_config_dict.pop('system_prompt', None)
            llm_config_dict.pop('prompt_template', None)
            
            # Set response format to json_object
            llm_config = LLMConfig(**llm_config_dict)
            
            # Get current date information
            current_date = datetime.now()
            current_year = current_date.year
            current_month = current_date.strftime("%B")
            
            # Format the prompt
            prompt = prompt_template.format(
                query=base_query,
                current_year=current_year,
                current_month=current_month
            )
            
            # Create structured tool with simplified schema
            structured_tool = StructuredTool(
                json_schema=SearchQueries.model_json_schema(),
                schema_name="generate_search_queries",
                schema_description="Generate a list of search queries based on the base query",
                instruction_string="Generate 2-3 search queries following this schema:"
            )
            
            # Create prompt context
            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=system_prompt,
                new_message=prompt,
                llm_config=llm_config,
                structured_output=structured_tool,
                use_schema_instruction=True,
                use_history=False
            )
            
            logger.info(f"Sending prompt to LLM:\nSystem: {system_prompt}\nPrompt: {prompt}")
            
            responses = await self.ai_utils.run_parallel_ai_completion([context])
                    
            if not responses:
                logger.error("No response received from LLM")
                return [f"{base_query} {current_year} {current_month} latest"]
            
            # Parse response using structured output
            if responses[0].json_object and hasattr(responses[0].json_object, 'object'):
                try:
                    queries = SearchQueries(**responses[0].json_object.object)
                    return queries.queries
                except Exception as e:
                    logger.error(f"Error validating response: {str(e)}")
            
            # Fallback
            logger.warning("Using fallback query")
            return [f"{base_query} {current_year} {current_month} latest"]
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [f"{base_query} {current_year} latest"]
        
    def get_urls_for_query(self, query: str, num_results: int = 2) -> List[str]:
        """Get URLs from Google search with retry logic"""
        for attempt in range(self.max_retries):
            try:
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                urls = list(search(
                    query,
                    num=num_results,
                    stop=num_results,
                    pause=self.search_params['pause']
                ))
                
                self.last_request_time = time.time()
                
                if urls:
                    logger.info(f"\n=== URLs Found ===")
                    logger.info(f"Query: {query}")
                    for i, url in enumerate(urls, 1):
                        logger.info(f"URL {i}: {url}")
                    logger.info("================")
                    
                    # Store query-URL mapping
                    for url in urls:
                        self.query_url_mapping[url] = query
                    return urls
                    
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.request_delay * (attempt + 1)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    
        logger.error(f"All search attempts failed for query: {query}")
        return []

async def main():
    # Load config and prompts
    config_data, prompts = load_config()
    config = WebSearchConfig(**config_data)
    
    # Initialize AI utilities with request limits
    oai_request_limits = RequestLimits(
        max_requests_per_minute=500,
        max_tokens_per_minute=150000
    )
    ai_utils = ParallelAIUtilities(
        oai_request_limits=oai_request_limits,
        anthropic_request_limits=None
    )
    
    # Initialize search manager
    search_manager = SearchManager(ai_utils, config, prompts)
    
    # Test query
    test_query = "latest developments in artificial intelligence"
    logger.info(f"Testing search query generation for: {test_query}")
    
    # Generate search queries
    search_queries = await search_manager.generate_search_queries(test_query)
    
    logger.info(f"""
        === Search Query Generation Results ===
        Base Query: {test_query}
        Generated Queries: 
        {chr(10).join(f'- {query}' for query in search_queries)}
        =====================================
    """)

if __name__ == "__main__":
    asyncio.run(main())