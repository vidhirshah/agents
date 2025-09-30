import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from src.ppeline.agents import Agents 
from src.rags.rag_pipeline import RagPipeline
from crewai.agent import Agent

load_dotenv() 
os.environ["GEMINI_API_KEY"] = ""
os.environ["MISTRAL_API_KEY"] = ""

class retrivedSingleDocs(BaseModel):
    docs:str = Field(description="Single document retrived from the database")

class dbOutput(BaseModel):
    retrived_docs: List[str] = Field(description="List of retrived documents/decomposed questions")

class RagState(BaseModel):
    query: str = ""
    sub_questions : List[str] = []
    retrieved_docs: List[str] = []
    answer: str = ""

class RagFlow(Flow[RagState]):
    """Flow for answering queries using Retrieval-Augmented Generation (RAG)."""
    @start()
    def get_User_Input(self) :
        """Start the flow with user input."""
        query = input("Enter your query: ")
        self.state.query = query
        return 

    @listen(get_User_Input)
    def query_Decompose(self):
        llm = LLM(model="gemini/gemini-2.0-flash", response_format=dbOutput)  
        agent = Agent(
            role="Query Decomposition Specialist",
            goal=f"Analyze complex user query {self.state.query} to determine if they should be decomposed into smaller, more specific sub-queries for better retrieval accuracy. If not return the original query",
            backstory="You are an expert at understanding natural language questions. When queries are broad, multi-faceted, or ambiguous, you break them  into focused sub-queries that can be handled more effectively by the retrieval agent. If no decomposition is needed, you simply pass the query along unchanged.",
            llm=llm,
            response_format = dbOutput,
            verbose=False
        )
        ans = agent.kickoff(self.state.query,response_format=dbOutput)
        self.state.sub_questions = ans.pydantic.retrived_docs
        print(ans,"\n\m")
        return 
    
    @listen(query_Decompose)
    def retrive_Queries(self):
        # print("Creating guide outline...",self.state)
        gemini_llm = LLM(model="gemini/gemini-2.0-flash", response_format=dbOutput)     
        input = {"query": self.state.query}
        rag_retriver = RagPipeline(
            chromaDBPath="db/chroma/",
            collection_name="my_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        for q in self.state.sub_questions:
            data = rag_retriver.searchQuery(self.state.query)
            self.state.retrieved_docs = data[0]
        return self.state.retrieved_docs

    @listen(retrive_Queries)
    def generate_Answer(self,retrived_data):
        agent = Agents().crew()
        ans = agent.kickoff(inputs={"query":self.state.query,"retrived_docs":retrived_data})
        return ans
# def selector_agent():
#     return Agent(
#         role="Relevant Points Selector",
#         goal=(
#             "Given a user query and a list of retrieved documents, "
#             "identify and return only the key points that can answer the query accurately."
#         ),
#         backstory=(
#             "You are an expert in reading text and extracting the most important information "
#             "that directly answers a specific question. Avoid irrelevant details."
#         ),
#         response_format=SelectedPointsOutput,
#         llm=None,  # replace with your LLM instance, e.g., LLM(model="gemini-2.5-flash")
#         verbose=True
#     )