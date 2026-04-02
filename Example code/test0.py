# this code is to demonstrate the coding flow in langgraph with simple addition of two number functionality
#start and end are the dummy nodes that we use to define the start and end node
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict

# TypedDict defines the structure of our state dictionary
# Think of it like a blueprint for what data our workflow will carry
class addition_state(TypedDict):
    num1: float    # First number for addition
    num2: float    # Second number for addition
    result: float  # Will store the final answer


#write the node/function
def addfun(state: addition_state)->addition_state:
    num1 = state['num1']
    num2 = state['num2']
    result = num1 + num2
    
    state['result'] = result#here we are updating the result value of the state
    
    return state #here we are returning the whole state 


#define graph
# whenever we have to make a graph that we make using the state graph
# while making the graph we pass the state as input in it
graph = StateGraph(addition_state)

# add nodes to the graph
#here the first parameter is the name that represents the addfun fucntion the names can be differet
graph.add_node('add', addfun)

# add edges to the graph
graph.add_edge(START,'add')
graph.add_edge('add',END)

# compile the graph  
workflow = graph.compile()

# defining the initial values of the state
initial_state = {'num1':5,'num2':5}

# execute the workflow
final_state = workflow.invoke(initial_state)

print(final_state['result'])

