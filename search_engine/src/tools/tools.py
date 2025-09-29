from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional
from src.rags.rag_pipeline import RagPipeline
from pydantic import BaseModel, Field

class RagToolSchema(BaseModel):
    url: Optional[str] = Field(None, description="Website URL to ingest into the vector DB")
    query: Optional[str] = Field(None, description="Query to perform semantic search on")

class RagTool(BaseTool):
    name: str = "rag_pipeline_tool"
    description: str = (
        "A tool that can ingest a website into the vector DB and perform semantic search on queries."
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

    def ingest_website(self, url: str, chunkSize: int = 500, overLap: int = 300) -> str:
        print("\n\n\nUrl: in tool ",url,"\n\n\n")
        return self.rag_pipeline.websiteLoader(url, chunkSize, overLap)

    def _run(self,url:Optional[str] = None, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Handles both ingestion and search.
        url: optional website to ingest
        query: optional question to search
        """
        response = {
            "ingestion_status": None,
            "search_results": []
        }
        try:
            if url:
                status = self.ingest_website(url)
                response["ingestion_status"] = status

            if query:
                results = self.rag_pipeline.searchQuery(query)
                response["search_results"] = results

            return response
        except Exception as e:
            return {"error": str(e)}
