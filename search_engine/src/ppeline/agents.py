from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
import yaml
from pathlib import Path
from src.tools.tools import RagTool

@CrewBase
class Agents:
    # Get the project root dynamically
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # agents.py → ppeline → src → project root
        AGENTS_YAML = self.PROJECT_ROOT / "config" / "agents" / "agents.yaml"
        TASKS_YAML = self.PROJECT_ROOT / "config" / "tasks" / "tasks.yaml"
        self.agents_config = self.load_yaml("/home/sysad/Desktop/research/lg/agents/search_engine/src/ppeline/config/agents.yaml")
        self.tasks_config = self.load_yaml("/home/sysad/Desktop/research/lg/agents/search_engine/src/ppeline/config/tasks.yaml")
        
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
            verbose=config.get("verbose", False)
        )

    @agent
    def evaluator_agent(self):
        config = self.agents_config['evaluator_agent']
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[],
            verbose=config.get("verbose", False)
        )
    
    @agent
    def synthesizer_agent(self):
        config = self.agents_config['synthesizer_agent']
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[],
            verbose=config.get("verbose", False)
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
    def context_evaluation_task(self) -> Task:
        config = self.tasks_config['context_evaluation_task']
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
            context=[self.context_evaluation_task()]  # Context from previous tasks
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.rag_agent(),self.evaluator_agent(),self.synthesizer_agent()],
            tasks=[self.rag_search_task(), self.context_evaluation_task(), self.synthesis_task()],
            # process=Process.sequential,
            verbose=False,
        )