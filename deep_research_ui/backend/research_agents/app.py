from fastapi import FastAPI, HTTPException, Form, File, UploadFile  # Add Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
import sys
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import json
import openai
from io import StringIO
import io
from sse_starlette.sse import EventSourceResponse
import asyncio


load_dotenv()

# Setup path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from research_agent_enhanced import WebSearchAgent, WebSearchConfig
from utils import load_config, logger

# Initialize FastAPI with metadata

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_KEY')
)


log_queue = asyncio.Queue()

async def log_generator():
    while True:
        log_message = await log_queue.get()
        if log_message is None:  # Sentinel value to stop the generator
            break
        yield {
            "event": "log",
            "data": json.dumps({"message": log_message, "timestamp": datetime.now().isoformat()})
        }

app = FastAPI(
    title="Research API",
    description="API for web research and analysis",
    version="1.0.0"
)

# Corrected CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Research API</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 40px auto; 
                    padding: 20px;
                    line-height: 1.6;
                }
                code {
                    background: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
                pre {
                    background: #f4f4f4;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <h1>Research API Documentation</h1>
            
            <h2>Available Endpoints:</h2>
            
            <h3>1. Research Query</h3>
            <p><code>POST /api/research</code></p>
            <p>Submit a research query to analyze web content.</p>
            <p>Request body example:</p>
            <pre>
{
    "query": "Your research query here",
    "urls": ["optional_url1", "optional_url2"]  // Optional
}
            </pre>
            
            <h3>2. Health Check</h3>
            <p><code>GET /api/test</code></p>
            <p>Check if the API is running.</p>
            
            <h2>Usage Example:</h2>
            <pre>
curl -X POST "http://localhost:5000/api/research" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "Latest developments in AI technology"}'
            </pre>

            <p>For interactive API documentation, visit: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """

@app.get("/api/test")
async def test():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is running"}


class Tool(BaseModel):
    id: str
    name: str
    description: str
@app.get("/api/research/stream")
async def research_stream(query: str):
    async def event_generator():
        research_steps = [
            "Initializing search parameters and configurations...",
            "Querying search engines for relevant sources...",
            "Analyzing search results and filtering relevant content...",
            "Extracting key information from sources...",
            "Processing market data and trends...",
            "Applying custom analysis tools...",
            "Generating comprehensive market insights...",
            "Finalizing analysis and preparing response..."
        ]
        
        for step in research_steps:
            await asyncio.sleep(0.5)  # Simulate processing time
            yield {
                "event": "message",
                "data": json.dumps({"log": step})
            }
    
    return EventSourceResponse(event_generator())
@app.post("/api/custom")
async def custom_chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    tools: Optional[str] = Form(None)
):
    try:
        logger.info(f"Received custom chat request - Message: {message}")
        
        # Parse tools if provided
        enabled_tools = []
        if tools:
            try:
                enabled_tools = json.loads(tools)
                logger.info(f"Enabled tools: {enabled_tools}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse tools JSON")

        # Process file if provided
        file_content = None
        if file:
            try:
                content = await file.read()
                file_content = content.decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                raise HTTPException(status_code=400, detail="Error reading file")

        # Create the system prompt
        system_prompt = "You are an expert financial analyst and market researcher."
        if enabled_tools:
            tool_descriptions = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in enabled_tools)
            system_prompt += f"\nEnabled tools:\n{tool_descriptions}"

        try:
            # Call OpenAI API
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message if not file_content else f"{message}\n\nFile content:\n{file_content}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return {
                "content": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(),
                "file_info": {
                    "filename": file.filename,
                    "size": len(file_content) if file_content else 0
                } if file else None
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process request")

    except Exception as e:
        logger.error(f"Custom chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/api/custom")
# async def custom_chat(
#     message: Optional[str] = Form(None),
#     file: Optional[UploadFile] = File(None)
# ):
#     try:
#         logger.info(f"Received message: {message}, file: {file.filename if file else None}")
        
#         if not message and not file:
#             raise HTTPException(
#                 status_code=400, 
#                 detail="Either message or file is required for analysis"
#             )

#         if file:
#             try:
#                 content = await file.read()
#                 logger.info(f"File content type: {file.content_type}")
#                 logger.info(f"File size: {len(content)} bytes")
                
#                 data = None
#                 if file.filename.endswith('.csv'):
#                     try:
#                         df = pd.read_csv(io.StringIO(content.decode('utf-8')))
#                     except UnicodeDecodeError:
#                         df = pd.read_csv(io.StringIO(content.decode('latin-1')))
#                     data = df.head(10).to_dict('records')
#                     file_info = {
#                         "filename": file.filename,
#                         "total_records": len(df),
#                         "columns": df.columns.tolist(),
#                         "preview_records": 10
#                     }
#                 elif file.filename.endswith('.json'):
#                     json_data = json.loads(content.decode('utf-8'))
#                     if isinstance(json_data, list):
#                         data = json_data[:10]
#                         file_info = {
#                             "filename": file.filename,
#                             "total_records": len(json_data),
#                             "preview_records": 10
#                         }
#                     else:
#                         data = [json_data]
#                         file_info = {
#                             "filename": file.filename,
#                             "total_records": 1,
#                             "preview_records": 1
#                         }
#                 else:
#                     raise HTTPException(
#                         status_code=400, 
#                         detail="Currently only CSV and JSON files are supported"
#                     )

#                 system_prompt = """You are a data analysis expert. Analyze the following data and provide insights.
#                 Focus on:
#                 1. Data structure and content overview
#                 2. Key patterns or trends
#                 3. Notable observations
#                 4. Potential areas for deeper analysis
#                 Be specific and reference actual data points from the preview."""

#                 user_prompt = f"""
#                 Analyzing file: {file.filename}
#                 Total records: {file_info['total_records']}
                
#                 Preview of data:
#                 {json.dumps(data, indent=2)}
                
#                 User query: {message if message else 'Provide a general analysis of this data.'}
#                 """

#                 response = await client.chat.completions.create(
#                     model="gpt-4",
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     temperature=0.7,
#                     max_tokens=2000
#                 )

#                 return {
#                     "content": response.choices[0].message.content,
#                     "file_info": file_info,
#                     "status": "success"
#                 }

#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON parsing error: {str(e)}", exc_info=True)
#                 raise HTTPException(
#                     status_code=400, 
#                     detail="Invalid JSON file format"
#                 )
#             except pd.errors.EmptyDataError:
#                 raise HTTPException(
#                     status_code=400, 
#                     detail="The CSV file is empty"
#                 )
#             except Exception as e:
#                 logger.error(f"File parsing error: {str(e)}", exc_info=True)
#                 raise HTTPException(
#                     status_code=400, 
#                     detail=f"Error parsing file: {str(e)}"
#                 )

#         # Handle text-only queries
#         system_prompt = """You are an expert financial analyst. 
#         Provide detailed, actionable insights based on the user's query."""
        
#         response = await client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": message}
#             ],
#             temperature=0.7,
#             max_tokens=1500
#         )

#         return {
#             "content": response.choices[0].message.content,
#             "status": "success"
#         }

#     except Exception as e:
#         logger.error(f"Custom chat error: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, 
#             detail=f"An error occurred while processing your request: {str(e)}"
#         )
    
def get_tools_path():
    """Get the absolute path to the custom_tools.json file and ensure the directory exists"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(
        os.path.dirname(os.path.dirname(current_dir)),  # Go up two levels
        'storage'
    )
    # Create the storage directory if it doesn't exist
    os.makedirs(storage_dir, exist_ok=True)
    
    tools_path = os.path.join(storage_dir, 'custom_tools.json')
    
    # Create an empty tools file if it doesn't exist
    if not os.path.exists(tools_path):
        with open(tools_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
    
    return tools_path
@app.post("/api/save-tool")
async def save_tool(tool_data: dict):
    try:
        # Path to custom_tools.json
        file_path = "storage/custom_tools.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Read existing tools
        existing_tools = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_tools = json.load(f)
        
        # Merge new tool with existing tools
        existing_tools.update(tool_data)
        
        # Save updated tools back to file
        with open(file_path, 'w') as f:
            json.dump(existing_tools, f, indent=2)
            
        return {"status": "success", "message": "Tool saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class DeleteToolRequest(BaseModel):
    toolName: str

@app.delete("/api/tools/delete")
async def delete_tool(request: DeleteToolRequest):
    try:
        file_path = "storage/custom_tools.json"
        with open(file_path, 'r') as f:
            tools = json.load(f)
            
        if request.toolName in tools:
            del tools[request.toolName]
            
            with open(file_path, 'w') as f:
                json.dump(tools, f, indent=2)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/delete-tool")
async def delete_tool(request: DeleteToolRequest):
    try:
        file_path = Path("storage/custom_tools.json")
        
        if file_path.exists():
            # Read existing tools
            with open(file_path, 'r', encoding='utf-8') as f:
                tools = json.load(f)
            
            # Remove the tool if it exists
            if request.toolName in tools:
                del tools[request.toolName]
                
                # Write updated tools back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(tools, f, indent=2, ensure_ascii=False)
                
        return {"status": "success", "message": "Tool deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/api/save-tools")
async def save_tools(tools: dict):
    try:
        # Use absolute path to ensure correct file location
        base_dir = Path(__file__).resolve().parent.parent
        file_path = base_dir / "storage" / "custom_tools.json"
        
        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        print(f"Saving tools to: {file_path}")  # Debug print
        print(f"Tools data: {json.dumps(tools, indent=2)}")  # Debug print
        
        # Write tools to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tools, f, indent=2, ensure_ascii=False)
            
        return {"status": "success", "message": "Tools saved successfully"}
    except Exception as e:
        print(f"Error saving tools: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/api/tools")
async def save_tools(tools: dict):
    try:
        file_path = Path("storage/custom_tools.json")
        file_path.parent.mkdir(exist_ok=True)
        
        # Read existing tools if file exists
        existing_tools = {}
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_tools = json.load(f)
        
        # Merge new tools with existing ones
        existing_tools.update(tools)
        
        # Save all tools back to file
        with open(file_path, 'w') as f:
            json.dump(existing_tools, f, indent=2)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/tools/{tool_id}/toggle")
async def toggle_tool(tool_id: str, enabled: bool):
    try:
        tools_path = get_tools_path()
        if os.path.exists(tools_path):
            with open(tools_path, 'r', encoding='utf-8') as f:
                tools = json.load(f)
            
            if tool_id in tools:
                tools[tool_id]['enabled'] = enabled
                with open(tools_path, 'w', encoding='utf-8') as f:
                    json.dump(tools, f, indent=2, ensure_ascii=False)
                return {"status": "success"}
        
        raise HTTPException(status_code=404, detail="Tool not found")
    except Exception as e:
        logger.error(f"Error toggling tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class CustomSchema(BaseModel):
    name: str
    description: str
    schema_definition: Dict[str, Any]

class ResearchRequest(BaseModel):
    query: str
    custom_schemas: List[CustomSchema] = Field(default_factory=list)

@app.get("/api/research/logs")
async def research_logs():
    async def generate():
        try:
            while True:
                message = await log_queue.get()
                if message is None:  # Sentinel value to stop the generator
                    break
                yield {
                    "event": "message",
                    "data": json.dumps({"message": message})
                }
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(generate())

@app.post("/api/research")
async def research(request: ResearchRequest):
    try:
        if not hasattr(app, 'log_queue'):
            app.log_queue = asyncio.Queue()
        # Debug logging
        await log_queue.put("🚀 Initializing research process...")
        logger.info("Received research request")
        logger.info(f"Query: {request.query}")
        logger.info(f"Raw request data: {request.dict()}")
        logger.info(f"Custom schemas received: {[schema.dict() for schema in request.custom_schemas]}")
        logger.info(f"Processing research query: {request}")
        
        # Convert custom schemas to the format expected by WebSearchAgent
        custom_schemas = [
            {
                "name": schema.name,
                "description": schema.description,
                "schema_definition": schema.schema_definition  # Changed from 'schema' to 'schema_definition'
            }
            for schema in request.custom_schemas
        ]
        
        logger.info(f"Processed custom schemas: {custom_schemas}")
        await log_queue.put("🔍 Collecting custom tools...")
        
        # Load config first
        config_data, prompts = load_config()
        config_data["query"] = request.query
        await log_queue.put("🔍 Configuring search parameters...")
        
        config = WebSearchConfig(**config_data)
        agent = WebSearchAgent(
            config=config,
            prompts=prompts,
            custom_schemas=custom_schemas,
            log_queue=log_queue 
        )
        await log_queue.put("📊 Starting web search and analysis...")
        try:
            await agent.process_search_query(request.query)
        except Exception as search_error:
            logger.error(f"Search process error: {str(search_error)}")
            raise HTTPException(status_code=500, detail="Search process failed")
        await log_queue.put("✅ Search completed, processing results...")
        formatted_results = []
        for result in agent.results:
            if result:
                try:
                    formatted_result = {
                        "url": result.url,
                        "title": result.title,
                        "content": result.content,
                        "timestamp": result.timestamp.isoformat(),
                        "status": result.status,
                        "summary": result.summary if hasattr(result, 'summary') else {},
                        "agent_id": result.agent_id if hasattr(result, 'agent_id') else None,
                        "extraction_method": result.extraction_method if hasattr(result, 'extraction_method') else None
                    }
                    
                    # Process custom fields if they exist
                    if hasattr(result, 'summary') and isinstance(result.summary, dict):
                        if 'assets' in result.summary and result.summary['assets']:
                            for asset in result.summary['assets']:
                                if 'custom_fields' in asset:
                                    processed_fields = {}
                                    for field_name, field_value in asset['custom_fields'].items():
                                        try:
                                            if isinstance(field_value, dict) and 'choices' in field_value:
                                                # Extract analysis from OpenAI response
                                                message_content = field_value['choices'][0]['message']['content']
                                                content_dict = json.loads(message_content)
                                                processed_fields[field_name] = content_dict.get('analysis', '')
                                            else:
                                                processed_fields[field_name] = str(field_value)
                                        except Exception as e:
                                            logger.error(f"Error processing field {field_name}: {str(e)}")
                                            processed_fields[field_name] = str(field_value)
                                    
                                    asset['custom_fields'] = processed_fields
                    
                    formatted_results.append(formatted_result)
                    
                except Exception as e:
                    logger.error(f"Error formatting result: {str(e)}")
                    continue
        await log_queue.put("✨ Research process completed successfully")
        return formatted_results
        

    except Exception as e:
        await log_queue.put(f"❌ Error occurred: {str(e)}")
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_field_description(field_name: str) -> str:
    """Generate a detailed description for a custom field based on its name."""
    try:
        prompt = f"""
        Generate a detailed description for a cryptocurrency market analysis field named "{field_name}".
        The description should:
        1. Explain what kind of analysis this field should contain
        2. Specify what metrics or data points should be included
        3. Indicate how this information is relevant for market analysis
        4. Include guidance for both technical and fundamental aspects if applicable
        
        Format: Return only the description text, no additional formatting.
        """

        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a finanical market analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        description = response.choices[0].message.content.strip()
        logger.info(f"Generated description for {field_name}: {description}")
        return description

    except Exception as e:
        logger.error(f"Error generating field description: {str(e)}")
        # Fallback description if generation fails
        return f"Analysis and insights related to {field_name} in financial & trading markets"
    



def start():
    """Function to start the server when running directly"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()