from src.ppeline.agents import Agents
from src.ppeline.tasks import Tasks
import csv

def main():
    test_cases = ["Name two presses that print banknotes in India?"]
    agents = Agents()
    outputs = []
    for query in test_cases:
        crew_instance = agents.crew().kickoff(inputs={ "query":query})
        print(f"Final Output: {crew_instance}")
        outputs.append({"Question":query, "Answer":crew_instance})
        fieldnames = outputs[0].keys()     

if __name__ == "__main__":
    main()