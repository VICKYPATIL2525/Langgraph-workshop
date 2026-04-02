# this is the llm based workflow in this we use the node which call the llm to answer the simple query  
# Import necessary libraries
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM with Anthropic Claude Haiku
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=500,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Define the state structure
class llmstate(TypedDict):
    question: str 
    answer: str

# Node function that uses LLM to answer questions
def llm_qa(state: llmstate) -> llmstate:
    # Extract question from state
    question = state['question']
   
    # Create prompt for LLM
    prompt = f'Answer the following question{question}'
    
    # Get answer from LLM
    answer = llm.invoke(prompt).content
   
    # Update answer in state
    state['answer'] = answer

    return state

# Create the graph
graph = StateGraph(llmstate)

# Add node to graph
graph.add_node('llm_qa', llm_qa)

# Add edges to connect nodes
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# Compile the graph into executable workflow
workflow = graph.compile()

# Define initial state with question
initial_state = {'question': 'how far is the moon from earth'}

# Execute workflow with initial state
final_state = workflow.invoke(initial_state)

# Print final state
print(final_state)