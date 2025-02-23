from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
from datetime import datetime

# Setup path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from research_agent_enhanced import WebSearchAgent, WebSearchConfig
from utils import load_config, logger

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://0.0.0.0:5000", "http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str
    urls: Optional[List[str]] = None

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
        </body>
    </html>
    """

@app.get("/api/test")
async def test():
    return {"status": "ok", "message": "Backend is running"}

class ResearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    logger.info(f"Received research request: {request.query}")
    try:
        config_data, prompts = load_config()
        config_data["query"] = request.query
        if request.urls:
            config_data["urls"] = request.urls
            
        config = WebSearchConfig(**config_data)
        agent = WebSearchAgent(config, prompts)
        await agent.process_search_query(request.query)
        
        # Generate output filename
        output_file = f"outputs/web_search/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save results and get metrics
        save_result = agent.save_results(output_file)
        
        formatted_results = []
        for result in agent.results:
            if result:
                formatted_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "timestamp": result.timestamp.isoformat(),
                    "status": result.status,
                    "summary": result.summary,
                    "agent_id": result.agent_id,
                    "extraction_method": result.extraction_method
                })
        
        # Return both results and metrics in the expected format
        return {
            "results": formatted_results,
            "metrics": {
                "total_articles": len(agent.results),
                "successful_extractions": sum(1 for r in agent.results if r and r.status == "success"),
                "failed_extractions": sum(1 for r in agent.results if not r or r.status != "success"),
                "database_status": save_result["metrics"]["database_status"],
                "output_file": output_file
            }
        }
        
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/research")
# async def research(request: ResearchRequest):
#     logger.info(f"Received research request: {request.query}")
#     try:
#         config_data, prompts = load_config()
#         config_data["query"] = request.query
#         if request.urls:
#             config_data["urls"] = request.urls
            
#         config = WebSearchConfig(**config_data)
#         agent = WebSearchAgent(config, prompts)
#         await agent.process_search_query(request.query)
        
#         formatted_results = []
#         for result in agent.results:
#             if result:
#                 formatted_results.append({
#                     "url": result.url,
#                     "title": result.title,
#                     "content": result.content,
#                     "timestamp": result.timestamp.isoformat(),
#                     "status": result.status,
#                     "summary": result.summary,
#                     "agent_id": result.agent_id,
#                     "extraction_method": result.extraction_method
#                 })
        
#         return formatted_results
        
#     except Exception as e:
#         logger.error(f"Research error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)