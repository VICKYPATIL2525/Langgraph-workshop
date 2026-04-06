# test0.py - Basic LangGraph workflow with NO LLM
# Shows the simplest possible graph: one node that adds two numbers
# Flow: START -> add -> END

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# State is the data container that flows through the graph
# Every node receives this and can read/write its fields
class addition_state(TypedDict):
    num1: float    # First number for addition
    num2: float    # Second number for addition
    result: float  # Will store the final answer

# Node function - receives the current state, does work, returns updated state
def addfun(state: addition_state) -> addition_state:
    num1 = state['num1']
    num2 = state['num2']
    result = num1 + num2

    state['result'] = result  # write the result back into state

    return state  # return the full updated state

# Create the graph and tell it which state structure to use
graph = StateGraph(addition_state)

# Register the node - 'add' is just the name we give it inside the graph
graph.add_node('add', addfun)

# Connect the nodes: START -> add -> END (linear flow)
graph.add_edge(START, 'add')
graph.add_edge('add', END)

# Compile turns the graph definition into a runnable workflow
workflow = graph.compile()

# Set the starting values for the state
initial_state = {'num1': 5, 'num2': 5}

# Run the workflow - it executes all nodes in order and returns final state
final_state = workflow.invoke(initial_state)

print(final_state['result'])  # output: 10
