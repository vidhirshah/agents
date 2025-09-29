from crewai import Task
import yaml
from pathlib import Path
# from src.tools.tools import RagTool

class Tasks:
    # Get the project root dynamically
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
        TASKS_YAML = self.PROJECT_ROOT / "config" / "tasks" / "tasks.yaml"
        self.tasks_config = self.load_yaml("/home/sysad/Desktop/research/lg/agents/search_engine/src/ppeline/config/tasks.yaml")        

    def load_yaml(self,file_path: Path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def create_rag_search_task(self):
        config = self.tasks_config['rag_search_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
        )
    
    def point_selection_task(self):
        config = self.tasks_config['point_selection_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
        )
    
    def create_synthesis_task(self):
        config = self.tasks_config['synthesis_task']
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=config["agent"],
        )