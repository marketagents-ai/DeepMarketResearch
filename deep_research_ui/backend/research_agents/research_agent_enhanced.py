import importlib
import os
from insert_agent_data import SimulationDataInserter
from web_search_manager import SearchManager, WebSearchConfig
from url_processor import URLFetcher, FetchedResult
from utils import (
    clean_response_content,
    load_config,
    logger,
    structure_text_response,
)
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMOutput, LLMPromptContext, LLMConfig, StructuredTool
from datetime import datetime
import json
from pydantic import BaseModel, Field ,validator  
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Type
import asyncio
import logging
from sse_starlette.sse import EventSourceResponse
import asyncio


class WebSearchResult(BaseModel):
    url: str
    title: str
    content: str
    timestamp: datetime
    status: str
    summary: Optional[dict] = {}
    agent_id: str
    extraction_method: str = "unknown"

log_queue = asyncio.Queue()
class WebSearchAgent:
    def __init__(
        self,
        config: WebSearchConfig,
        prompts: Dict[str, str],
        custom_schemas: Optional[List[Dict[str, Any]]] = None,
        log_queue: Optional[asyncio.Queue] = None
    ):
        self.config = config
        self.prompts = prompts
        self.custom_schemas = custom_schemas or []  # Store custom schemas
        self.log_queue = log_queue
        logger.info(f"Initialized WebSearchAgent with custom schemas: {self.custom_schemas}")
        self.results: List[WebSearchResult] = []
        _, self.prompts = load_config()
        self.log_queue = log_queue 
        
        def set_custom_schemas(self, schemas):
            """Add custom schemas to be used alongside default schemas"""
            self.custom_schemas = schemas
        
        # Initialize other components
        oai_request_limits = RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=150000
        )
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=None
        )
        self.llm_configs = config.llm_configs
        self.search_manager = SearchManager(self.ai_utils, config, prompts)
        
        # Instantiate URLFetcher for URL fetching only
        self.url_fetcher = URLFetcher(config, prompts)
    async def log(self, message: str):
        """Helper method to send logs to both logger and queue"""
        logger.info(message)
        if self.log_queue:
            await self.log_queue.put(message)
    async def generate_market_analysis(self, content: str) -> Dict[str, Any]:
        # Combine default and custom fields for analysis
        analysis_fields = {
            "TICKER": "Asset/token symbol",
            "RATING": "Investment rating",
            "TARGET_PRICE": "Price predictions",
            "SENTIMENT": "Market sentiment",
            "ACTION": "Recommended trading action",
            "CATALYSTS": "Key market drivers",
            "KPIS": "Key performance indicators",
            "SOURCES": "Data sources"
        }
        
        # Add custom fields to analysis
        for schema in self.custom_schemas:
            field_name = schema["name"].upper()
            analysis_fields[field_name] = schema["description"]
        
        # Update prompt to include custom fields
        analysis_prompt = self.create_analysis_prompt(content, analysis_fields)
        
        # Generate analysis including custom fields
        analysis_result = await self.generate_ai_analysis(analysis_prompt)
        
        return analysis_result
    def create_analysis_prompt(self, content: str, analysis_fields: Dict[str, str]) -> str:
        # Create prompt that includes all fields
        fields_description = "\n".join(
            f"- {field}: {description}"
            for field, description in analysis_fields.items()
        )
        
        return f"""
        Analyze the following content and provide a structured market analysis.
        Include analysis for each of these fields:
        
        {fields_description}
        
        Content to analyze:
        {content}
        """

    async def process_search_query(self, query: str) -> None:
        try:
            if self.log_queue:
                await self.log_queue.put("🔄 Starting search query processing...")
            search_queries = await self.search_manager.generate_search_queries(query)
            if self.log_queue:
                await self.log_queue.put(f"📊 Generated {len(search_queries)} search queries")
            
            # Format custom tools info using self.custom_schemas
            custom_tools_info = "\n".join([
                f"      - {schema['name']}: {schema['description']}"
                for schema in self.custom_schemas
            ]) if self.custom_schemas else "      None"
            
            log_message = "=== Search Process Starting ===\n"
            log_message += f"Original Query: {query}\n\n"
            log_message += "Active Custom Tools:\n"
            log_message += f"{custom_tools_info}\n\n"
            log_message += f"Generated {len(search_queries)} queries:\n"
            log_message += "".join([f"{i+1}. {q}\n" for i, q in enumerate(search_queries)])
            log_message += "=============================="
            
            logger.info(log_message)

            all_results = []
            
            for idx, search_query in enumerate(search_queries, 1):
                if self.log_queue:
                    await self.log_queue.put(f"🔍 Processing Query {idx}/{len(search_queries)}: {search_query}")
                logger.info(f"""
                                === Processing Query {idx}/{len(search_queries)} ===
                                Query: {search_query}
                                """)
                
                urls = self.search_manager.get_urls_for_query(
                    search_query, 
                    num_results=self.config.urls_per_query
                )
                
                logger.info(f"""
                                URLs found for query "{search_query}":
                                {chr(10).join(f'- {url}' for url in urls)}
                                """)
                if self.log_queue:
                    await self.log_queue.put(f"🌐 Found {len(urls)} URLs to process")
                for i, url in enumerate(urls, 1):
                    await log_queue.put(f"   {i}. {url}")

                for url in urls:
                    self.search_manager.query_url_mapping[url] = search_query
                    if self.log_queue:
                        await self.log_queue.put(f"🔗 Processing URL: {url}")
                
                # Fetch raw content without summary
                fetched_results = await self.url_fetcher.process_urls(urls, self.search_manager.query_url_mapping)

                # For each fetched result, generate summary and create WebSearchResult
                for fr in fetched_results:
                    await log_queue.put(f"📥 Content extracted from: {fr.url}")
                    # Generate summary if enabled
                    summary = {}
                    if self.config.use_ai_summary:
                        # Generate standard summary
                        summary = await self.generate_ai_summary(fr.url, fr.content, 
                                                            "Contains tables/charts" if fr.has_data else "Text only")
                        if self.log_queue:
                            await self.log_queue.put(f"✨ Summary completed for: {fr.url}")
                        
                        # Generate custom field summaries if custom schemas exist
                        if self.custom_schemas and 'assets' in summary and summary['assets']:
                            for asset in summary['assets']:
                                custom_fields = {}
                                for schema in self.custom_schemas:
                                    field_name = schema['name'].lower()
                                    # Generate specific analysis for custom field
                                    custom_analysis = await self._generate_custom_field_analysis(
                                        fr.content.get('text', ''),
                                        schema['name'],
                                        schema.get('description', '')
                                    )
                                    custom_fields[field_name] = custom_analysis
                                
                                # Add custom fields to asset
                                asset['custom_fields'] = custom_fields

                    web_result = WebSearchResult(
                        url=fr.url,
                        title=fr.title,
                        content=fr.content.get('text', '')[:self.config.content_max_length],
                        timestamp=datetime.now(),
                        status="success" if fr.content else "failed",
                        summary=summary,
                        agent_id=str(uuid.uuid4()),
                        extraction_method=fr.extraction_method
                    )
                    all_results.append(web_result)
                    if self.log_queue:
                        await self.log_queue.put(f"✅ Completed processing: {fr.url}")
                    successful = len([r for r in fetched_results if r.content])
                    failed = len(urls) - successful
                    await log_queue.put(f"""
                        📊 Query {idx} Results:
                        - URLs found: {len(urls)}
                        - Successfully processed: {successful}
                        - Failed: {failed}
                        """)

                    self.results = all_results
                    
                    total_successful = len([r for r in self.results if r and r.status == "success"])
                    await log_queue.put(f"""
✨ Search Complete:
   - Total queries processed: {len(search_queries)}
   - Total URLs processed: {sum(len(self.search_manager.get_urls_for_query(q)) for q in search_queries)}
   - Successfully analyzed: {total_successful}
   - Failed: {len(self.results) - total_successful}
""")
                
                logger.info(f"""
                            Query {idx} Results Summary:
                            - URLs processed: {len(urls)}
                            - Successful extractions: {len(fetched_results)}
                            - Failed extractions: {len(urls) - len(fetched_results)}
                            """)

            self.results = all_results
            
            logger.info(f"""
                    === Final Search Summary ===
                    Total Queries Processed: {len(search_queries)}
                    Total URLs Processed: {sum(len(self.search_manager.get_urls_for_query(q)) for q in search_queries)}
                    Total Successful Extractions: {len(self.results)}
                    Custom Fields Analyzed: {[schema['name'] for schema in self.custom_schemas] if self.custom_schemas else 'None'}
                    """)

        except Exception as e:
            logger.error(f"Error processing search query: {str(e)}")
            raise
    async def _generate_custom_field_analysis(self, content: str, field_name: str, description: str) -> str:
        """Generate specific analysis for a custom field"""
        try:
            prompt = f"""
            Based on the following market content, provide a detailed analysis for {field_name}.
            
            Field Description: {description}
            
            Content to analyze:
            {content}
            
            Requirements:
            1. Provide specific numerical predictions or values where applicable
            2. Include market-based justification for your analysis
            3. Reference specific data points from the content
            4. Consider both short-term and long-term implications
            5. Ensure analysis is actionable and concrete
            6. Base all analysis solely on the provided content
            7. Format the response as a single string with clear, structured analysis
            """

            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string="You are an expert financial analyst. Provide concise, specific analysis as a single text string.",
                new_message=prompt,
                llm_config=self.llm_configs["content_analysis"],
                use_history=False
            )
            
            responses = await self.ai_utils.run_parallel_ai_completion([context])
            
            if responses and len(responses) > 0:
                response = responses[0]
                
                # If response is a dict with choices (OpenAI format)
                if isinstance(response, dict) and 'choices' in response:
                    message_content = response['choices'][0]['message']['content']
                    try:
                        # Parse JSON and extract analysis
                        content_dict = json.loads(message_content)
                        return content_dict['analysis']
                    except (json.JSONDecodeError, KeyError):
                        return message_content.strip()
                
                # If response is already a string
                if isinstance(response, str):
                    try:
                        content_dict = json.loads(response)
                        return content_dict['analysis']
                    except (json.JSONDecodeError, KeyError):
                        return response.strip()
                
                # If response is a custom object with json_object attribute
                if hasattr(response, 'json_object') and hasattr(response.json_object, 'object'):
                    return response.json_object.object.get('analysis', str(response))
                    
                return str(response)
                
            return f"No analysis available for {field_name}"
            
        except Exception as e:
            logger.error(f"Error generating custom field analysis: {str(e)}")
            return f"Error analyzing {field_name}: {str(e)}"
    async def generate_ai_summary(self, url: str, content: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Generate AI summary using schema specified in config with enhanced custom field support."""
        try:
            await self.log(f"🔄 Starting AI summary generation for: {url}")
            # Get base LLM config
            llm_config_dict = self.config.llm_configs["content_analysis"].copy()
            schema_config = llm_config_dict.pop('schema_config', {})
            system_prompt = llm_config_dict.pop('system_prompt', None)
            llm_config = LLMConfig(**llm_config_dict)

            # Get merged schema class
            schema_class = self.get_merged_schemas()
            content_text = content.get('text', '')[:self.config.content_max_length]

            # Format custom fields section
            custom_fields_section = ""
            if self.custom_schemas:
                custom_fields_section = "\nCUSTOM FIELDS ANALYSIS REQUIRED:\n"
                for schema in self.custom_schemas:
                    for field_name, field_props in schema['schema_definition']['properties'].items():
                        custom_fields_section += f"""
                        {field_name.upper()}:
                        Description: {field_props.get('description', '')}
                        Requirements:
                        - Provide specific numerical predictions where applicable
                        - Include market-based justification
                        - Reference specific data points from the content
                        - Consider both short-term and long-term implications
                        - Provide confidence levels for predictions
                        """

            # Create enhanced prompt
            formatted_prompt = f"""
            Analyze this market content and provide detailed insights including both standard and custom metrics:

            URL: {url}
            CONTENT TYPE: {content_type}
            
            CONTENT:
            {content_text}

            STANDARD ANALYSIS REQUIREMENTS:
            1. Asset identification and basic metrics
            2. Price targets with confidence levels
            3. Market sentiment analysis
            4. Key catalysts and drivers
            5. Risk assessment
            6. Trading recommendations
            7. Source credibility assessment

            {custom_fields_section}

            RESPONSE REQUIREMENTS:
            1. Base all analysis strictly on the provided content
            2. Include quantitative metrics where available
            3. Provide specific predictions with confidence levels
            4. Structure response in JSON format
            5. Include all standard and custom fields
            6. Ensure comprehensive coverage of all metrics
            """

            structured_tool = StructuredTool(
                json_schema=schema_class.model_json_schema(),
                schema_name=schema_config.get('schema_name', 'MarketResearch'),
                schema_description=schema_config.get('schema_description', 'Market analysis with custom fields'),
                instruction_string=schema_config.get('instruction_string', 'Provide structured market analysis')
            )

            context = LLMPromptContext(
                id=str(uuid.uuid4()),
                system_string=system_prompt or "You are a financial market analysis expert.",
                new_message=formatted_prompt,
                llm_config=llm_config.dict(),
                structured_output=structured_tool,
                use_schema_instruction=True,
                use_history=False
            )

            # Process with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    responses = await self.ai_utils.run_parallel_ai_completion([context])
                    
                    if responses and len(responses) > 0:
                        response = responses[0]
                        if response.json_object and hasattr(response.json_object, 'object'):
                            try:
                                # Pre-process the response to ensure required fields exist
                                response_data = response.json_object.object
                                if isinstance(response_data, dict):
                                    # Ensure required fields exist
                                    if 'analysis_type' not in response_data:
                                        response_data['analysis_type'] = 'asset'  # Default type
                                    
                                    # Initialize assets if not present
                                    if 'assets' not in response_data:
                                        response_data['assets'] = []
                                    
                                    # Process custom fields
                                    for asset in response_data.get('assets', []):
                                        if 'custom_fields' in asset:
                                            asset['custom_fields'] = {
                                                k: str(v) if isinstance(v, (dict, list)) else str(v)
                                                for k, v in asset['custom_fields'].items()
                                            }
                                
                                # Now validate with the schema
                                schema_class = self.get_merged_schemas()
                                result = schema_class(**response_data)
                                return json.loads(result.model_dump_json(exclude_none=True))
                            except Exception as validation_error:
                                logger.error(f"Validation error: {str(validation_error)}")
                                # Return a valid default structure
                                return {
                                    'analysis_type': 'asset',
                                    'assets': [{
                                        'custom_fields': {}
                                    }]
                                }

                    # Return valid default structure
                    return {
                        'analysis_type': 'asset',
                        'assets': [{
                            'custom_fields': {}
                        }]
                    }

                except Exception as e:
                    logger.error(f"Error in generate_ai_summary: {str(e)}")
                    return {
                        'analysis_type': 'asset',
                        'assets': [{
                            'custom_fields': {}
                        }]
                    }
        except Exception as e:
            logger.error(f"Error in previous method: {str(e)}")
            return {} 
    def get_schema_class(self, schema_name: str) -> Type[BaseModel]:
        """Dynamically import and return the specified schema class from research_schemas."""
        try:
            schemas_module = importlib.import_module('market_agents.research_agents.research_schemas')
            schema_class = getattr(schemas_module, schema_name)
            return schema_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading schema {schema_name}: {str(e)}")
            raise

    def get_merged_schemas(self) -> Type[BaseModel]:
        """Merge default research schemas with custom schemas"""
        try:
            # Get base schema class
            base_schema = self.get_schema_class('MarketResearch')
            
            # If no custom schemas, return base class
            if not self.custom_schemas:
                return base_schema

            # Create a new AssetAnalysis class with custom fields
            class CustomAssetAnalysis(self.get_schema_class('AssetAnalysis')):
                custom_fields: Dict[str, str] = Field(
                    default_factory=dict,
                    description="Custom analysis fields as key-value string pairs"
                )

                @validator('custom_fields')  # Use validator decorator
                def validate_custom_fields(cls, v):
                    # Ensure all values are strings
                    return {k: str(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in v.items()}

            # Create a new MarketResearch class with custom asset analysis
            class CustomMarketResearch(base_schema):
                assets: List[CustomAssetAnalysis] = Field(
                    default_factory=list,
                    description="List of asset analyses including custom fields"
                )

            logger.info(f"Successfully created merged schema with custom fields")
            return CustomMarketResearch

        except Exception as e:
            logger.error(f"Error merging schemas: {str(e)}")
            # Fallback to base schema if merge fails
            return self.get_schema_class('MarketResearch')

    def _format_custom_fields_prompt(self) -> str:
        """Format custom fields for the prompt with detailed requirements"""
        if not self.custom_schemas:
            return ""
        
        custom_fields_prompt = []
        for schema in self.custom_schemas:
            field_name = schema['name'].upper()
            description = schema.get('description', '')
            requirements = schema.get('requirements', [])
            
            # Enhanced prompt structure for better AI analysis
            field_prompt = f"""
            {field_name}:
            - Provide detailed {field_name} analysis based on available market data
            - Include quantitative metrics and qualitative insights
            - Consider historical trends and current market conditions
            - Analyze impact on trading decisions
            - Base analysis on following factors:
                * Market data and trends
                * Technical indicators
                * News sentiment
                * Trading volumes
                * Market participant behavior
            - Description: {description}
            - Additional Requirements: {', '.join(requirements) if requirements else 'None'}
            """
            custom_fields_prompt.append(field_prompt)
        
        return "\n".join(custom_fields_prompt)

    def _parse_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the analysis response including custom fields"""
        try:
            parsed = {
                'assets': [{
                    'ticker': '',
                    'rating': '',
                    'target_price': '',
                    'sentiment': '',
                    'action': '',
                    'catalysts': [],
                    'kpis': [],
                    'sources': [],
                }]
            }
            
            if 'assets' in response and response['assets']:
                asset = response['assets'][0]
                
                # Add custom fields to the asset
                if self.custom_schemas:
                    custom_fields = {}
                    for schema in self.custom_schemas:
                        field_name = schema['name'].lower()
                        if field_name in response:
                            custom_fields[field_name] = response[field_name]
                        elif field_name in asset:
                            custom_fields[field_name] = asset[field_name]
                    
                    if custom_fields:
                        asset['custom_fields'] = custom_fields
                        
                parsed['assets'][0] = asset

            return parsed

        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return {}
    def save_results(self, output_file: str):
        """Save results to file and attempt database insertion"""
        results_dict = []
        
        logger.info("\n=== ARTICLE SUMMARIES ===")
        
        for result in self.results:
            if result is None:
                continue
                
            try:
                result_data = result.model_dump(exclude_none=True)
                results_dict.append(result_data)

                # Simply print the entire summary
                logger.info(f"""
                    === ARTICLE DETAILS ===
                    URL: {result.url}
                    TITLE: {result.title}
                    EXTRACTION METHOD: {result.extraction_method}

                    SUMMARY:
                    {json.dumps(result.summary, indent=2)}
                    =============================
                    """)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
            
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Try to save to database (if applicable)
        try:
            db_params = {
                'dbname': os.getenv('DB_NAME', 'market_simulation'),
                'user': os.getenv('DB_USER', 'db_user'),
                'password': os.getenv('DB_PASSWORD', 'db_pwd@123'),
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432')
            }
            
            inserter = SimulationDataInserter(db_params)
            
            if inserter.test_connection():
                logger.info("Database connection successful")
                inserter.insert_article_summaries(results_dict)
                logger.info(f"Successfully inserted {len(results_dict)} article summaries into database")
            else:
                raise Exception("Database connection test failed")
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            logger.info(f"Results saved to file: {output_file}")


async def main():
    config_data, prompts = load_config()
    config = WebSearchConfig(**config_data)
    agent = WebSearchAgent(config, prompts)
    logger.info(f"Starting search with query: {config.query}")
    await agent.process_search_query(config.query)
    
    output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    agent.save_results(output_file)
    
    successful = sum(1 for r in agent.results if r and r.status == "success")
    failed = len(agent.results) - successful if agent.results else 0
    
    logger.info(f"""
                    Search completed:
                    - Query: {config.query}
                    - Total items processed: {len(agent.results) if agent.results else 0}
                    - Successful: {successful}
                    - Failed: {failed}
                    - Results saved to: {output_file}
                            """)

if __name__ == "__main__":
    asyncio.run(main())
