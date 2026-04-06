# test1.py - Single LLM node workflow
# Shows how to plug an LLM into a LangGraph node
# Flow: START -> llm_qa -> END

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# Load ANTHROPIC_API_KEY from .env file
load_dotenv()

# Initialize the LLM - this object is used inside nodes to call Claude
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,   # low = more focused, less random
    max_tokens=500,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# State carries the question in and the answer out
class llmstate(TypedDict):
    question: str
    answer: str

# The single node - sends the question to LLM and stores the answer in state
def llm_qa(state: llmstate) -> llmstate:
    question = state['question']

    # Build the prompt string to send to the LLM
    prompt = f'Answer the following question{question}'

    # .invoke() calls the LLM and .content extracts the text from the response
    answer = llm.invoke(prompt).content

    state['answer'] = answer  # write answer back into state

    return state

# Build graph
graph = StateGraph(llmstate)

graph.add_node('llm_qa', llm_qa)

# Single straight-line flow
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()

# Pass the question in the initial state
initial_state = {'question': 'how far is the moon from earth'}

# Run - LLM is called inside llm_qa node
final_state = workflow.invoke(initial_state)

# Prints the full state: {'question': '...', 'answer': '...'}
print(final_state)
