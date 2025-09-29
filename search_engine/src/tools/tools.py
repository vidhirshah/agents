from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional
from src.rags.rag_pipeline import RagPipeline
from pydantic import BaseModel, Field

class RagToolSchema(BaseModel):
    query: Optional[str] = Field(None, description="Query to perform semantic search on")

class RagTool(BaseTool):
    name: str = "rag_pipeline_tool"
    description: str = (
        "A tool can perform semantic search on given queries in a vector database."
    )
    args_schema: type[BaseModel] = RagToolSchema
    rag_pipeline: RagPipeline = Field(default=None)
    def __init__(self, chromaDBPath: str, collection_name: str, model_name: str):
        super().__init__()
        self.rag_pipeline = RagPipeline(
            chromaDBPath=chromaDBPath,
            collection_name=collection_name,
            model_name=model_name
        )

    def _run(self,query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Handles search.
        query: optional question to search
        """
        response = {
            "ingestion_status": None,
            "search_results": []
        }
        try:

            if query:
                results = self.rag_pipeline.searchQuery(query)
                response["search_results"] = results

            return response
        except Exception as e:
            return {"error": str(e)}

class LoaderToolSchema(BaseModel):
    url: Optional[str] = Field(None, description="URL of the website to load")

class LoaderTool(BaseTool):
    name: str = "website_loader_tool"
    description: str = (
        "A tool to load and ingest content from a specified website URL into the vector database."
    )
    args_schema: type[BaseModel] = LoaderToolSchema
    rag_pipeline: RagPipeline = Field(default=None)
    def __init__(self, chromaDBPath: str, collection_name: str, model_name: str):
        super().__init__()
        self.rag_pipeline = RagPipeline(
            chromaDBPath=chromaDBPath,
            collection_name=collection_name,
            model_name=model_name
        )

    def _run(self,url: Optional[str] = None) -> Dict[str, Any]:
        """
        Handles website loading and ingestion.
        url: optional URL of the website to load
        """
        response = {
            "ingestion_status": None
        }
        try:

            if url:
                status = self.rag_pipeline.websiteLoader(url)
                response["ingestion_status"] = status

            return response
        except Exception as e:
            return {"error": str(e)}