from src.ppeline.agents import Agents
from src.ppeline.tasks import Tasks
import csv

def main():
    test_cases = ["For education loans up to what amount do banks not require collateral security?", 
                "Name two presses that print banknotes in India.",
                "What are some of the mints of Government of India for coin production?",
                "What conditions make a mutilated banknote ineligible for refund under RBI Note Refund Rules?",
                "Where can mutilated or soiled banknotes be exchanged?",
                "Do joint accounts with names in different order receive separate deposit insurance?]"]
    agents = Agents()
    outputs = []
    for query in test_cases:
        crew_instance = agents.crew().kickoff(inputs={"url": "https://rbi.org.in/Scripts/bs_viewcontent.aspx?Id=624", "query":query})
        # Extract RAG agent for printing
        # Add query if tasks expect it
        # print(f"Goal: {rag_agent.goal}")./sc  
        # print(f"Backstory: {rag_agent.backstory}")
        print(f"Final Output: {crew_instance}")
        outputs.append({"Question":query, "Answer":crew_instance})
        fieldnames = outputs[0].keys()
    with open('test.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # write column headers
        writer.writerows(outputs)  
        


if __name__ == "__main__":
    main()