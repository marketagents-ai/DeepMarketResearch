import json
import os
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolsManager:
    def __init__(self, storage_path: str = None):
        # Set absolute storage path
        if not storage_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                'storage',
                'custom_tools.json'
            )
        self.storage_path = storage_path or "storage/custom_tools.json"
        self.tools = self.load_tools()

    def _ensure_storage_directory(self):
        """Ensure storage directory and file exist"""
        try:
            directory = os.path.dirname(self.storage_path)
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            if not os.path.exists(self.storage_path):
                with open(self.storage_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2)
                logger.info(f"Created empty tools file at {self.storage_path}")
        except Exception as e:
            logger.error(f"Error creating storage: {str(e)}")
            raise

    def load_tools(self) -> Dict[str, Any]:
        """Load tools from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading tools: {str(e)}")
            return {}

    def save_tools(self) -> None:
        """Save tools to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.tools, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tools: {str(e)}")

    def add_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """Add or update a tool"""
        from research_schemas import validate_schema  
        try:
            if 'schema' in tool_config:
                # Fixed the missing closing parenthesis
                validate_schema(tool_config['schema'])
            # Ensure required fields
            tool_config.update({
                "name": tool_name,
                "enabled": True,
                "created_at": datetime.utcnow().isoformat(),
                "type": tool_config.get("type", "custom")
            })
            
            # Add/update tool and save immediately
            self.tools[tool_name] = tool_config
            self.save_tools()
            logger.info(f"Successfully added tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error adding tool {tool_name}: {str(e)}")
            raise

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool"""
        try:
            if tool_name in self.tools:
                del self.tools[tool_name]
                self.save_tools()
                logger.info(f"Successfully removed tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error removing tool {tool_name}: {str(e)}")
            raise

    def get_tools(self) -> Dict[str, Any]:
        """Get all tools"""
        return self.tools

    def get_tool(self, tool_name: str) -> Dict[str, Any]:
        """Get a specific tool"""
        return self.tools.get(tool_name, {})