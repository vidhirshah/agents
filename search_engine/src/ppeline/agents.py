from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
import yaml
from pathlib import Path
from src.tools.tools import RagTool , LoaderTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from crewai import LLM
from pydantic import BaseModel, Field
from typing import List, Dict

load_dotenv() 
os.environ["GEMINI_API_KEY"] = ""
os.environ["MISTRAL_API_KEY"] = ""

class dbOutput(BaseModel):
    retrived_docs: List[str] = Field(description="List of retrived documents/decomposed questions")

@CrewBase
class Agents:
    # Get the project root dynamically
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # agents.py → ppeline → src → project root
        AGENTS_YAML = self.PROJECT_ROOT / "config" / "agents" / "agents.yaml"
        TASKS_YAML = self.PROJECT_ROOT / "config" / "tasks" / "tasks.yaml"
        self.agents_config = self.load_yaml("/home/sysad/Desktop/research/lg/agents/search_engine/src/ppeline/config/agents.yaml")
        self.tasks_config = self.load_yaml("/home/sysad/Desktop/research/lg/agents/search_engine/src/ppeline/config/tasks.yaml")
        self.geminillm = LLM(
            model="gemini/gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
        )
        self.mistrialllm = LLM(
            model="gemini/gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )


    def load_yaml(self,file_path: Path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    @agent
    def rag_agent(self):
        config = self.agents_config['rag_agent']
        rag_tool = RagTool(
            chromaDBPath="db/chroma/",
            collection_name="my_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            # tools=[rag_tool],
            llm=self.geminillm,
            verbose=config.get("verbose", False)
        )

    @agent
    def selector_agent(self):
        config = self.agents_config['selector_agent']
        llm = self.mistrialllm
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[],
            llm=llm,
            verbose=config.get("verbose", False)
        )
    
    @agent
    def synthesizer_agent(self):
        config = self.agents_config['synthesizer_agent']
        llm = self.geminillm
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[],
            llm=llm,
            verbose=config.get("verbose", False)
        )
    
    @agent
    def doc_loader_agent(self):
        config = self.agents_config['doc_loader_agent']
        llm = self.geminillm
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            llm=llm,
            verbose=config.get("verbose", False)
        )

    @agent
    def decomposer_agent(self):
        config = self.agents_config['decomposer_agent']
        llm =LLM(
            model="gemini/gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            response_format=dbOutput
        )
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            llm=llm,
            response_format = dbOutput,
            verbose=config.get("verbose", False)
        )

    @task
    def query_decomposition_task(self) -> Task:
        config = self.tasks_config['query_decomposition_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
        )

    @task
    def document_loader_task(self) -> Task:
        config = self.tasks_config['document_loader_task']
        rag_tool = LoaderTool(
            chromaDBPath="db/chroma/",
            collection_name="my_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            tools=[rag_tool],
            agent=config["agent"],
        )

    @task
    def rag_search_task(self) -> Task:
        config = self.tasks_config['rag_search_task']
        rag_tool = RagTool(
            chromaDBPath="db/chroma/",
            collection_name="my_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            tools=[rag_tool],
            agent=config["agent"],
        )

    @task
    def point_selection_task(self) -> Task:
        config = self.tasks_config['point_selection_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
            context=[self.rag_search_task()]  # Context from the RAG search task
        )

    @task
    def synthesis_task(self) -> Task:
        config = self.tasks_config['synthesis_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
            context=[self.rag_search_task()]  # Context from previous tasks
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.synthesizer_agent()],
            tasks=[self.synthesis_task()],
            # process=Process.sequential,
            verbose=False,
        )
    
    # @crew
    # def test_crew(self) -> Crew:
    #     return Crew(
    #         agents=[self.doc_loader_agent()],
    #         tasks=[self.document_loader_task()],
    #         verbose=False,
    #     )
    
    # @crew
    # def decomp_crew(self) -> Crew:
    #     return Crew(
    #         agents=[self.decomposer_agent()],
    #         tasks=[self.query_decomposition_task()],
    #         verbose=False
    #     )