import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from src.ppeline.agents import Agents 
from src.rags.rag_pipeline import RagPipeline

load_dotenv() 
os.environ["GEMINI_API_KEY"] = ""
os.environ["MISTRAL_API_KEY"] = ""

class retrivedSingleDocs(BaseModel):
    docs:str = Field(description="Sing;e document retrived from the database")

class dbOutput(BaseModel):
    retrived_docs: List[retrivedSingleDocs] = Field(description="List of retrived documents")

class RagState(BaseModel):
    query: str = ""
    retrieved_docs: List[str] = []
    answer: str = ""

class RagFlow(Flow[RagState]):
    """Flow for answering queries using Retrieval-Augmented Generation (RAG)."""
    @start()
    def get_User_Input(self) :
        """Start the flow with user input."""
        query = input("Enter your query: ")
        self.state.query = query
        return self.state

    @listen(get_User_Input)
    def retrive_Queries(self):
        print("Creating guide outline...",self.state)
        gemini_llm = LLM(model="openai/gpt-4o-mini", response_format=dbOutput)    
        input = {"query": self.state.query}
        rag_retriver = RagPipeline(
            chromaDBPath="db/chroma/",
            collection_name="my_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        data = rag_retriver.searchQuery(self.state.query)
        self.state.retrieved_docs = data[0]
        return data[1]

    @listen(retrive_Queries)
    def generate_Answer(self,retrived_data):
        agent = Agents().crew()
        ans = agent.kickoff(inputs={"query":self.state.query,"retrived_docs":retrived_data})
        return ans
