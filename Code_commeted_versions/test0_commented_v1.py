"""
LangGraph Basic Example: Simple Addition Workflow
Perfect for beginners to understand LangGraph fundamentals!
"""

# ========================================================
# IMPORT STATEMENTS EXPLAINED
# ========================================================
# StateGraph: The main class to create our workflow graph
# START: A special marker that indicates where the graph begins
# END: A special marker that indicates where the graph ends
from langgraph.graph import StateGraph, START, END 

# TypedDict: Helps us define the structure of our state (like a blueprint)
from typing import TypedDict


# ========================================================
# STEP 1: DEFINE THE STATE STRUCTURE (Like a Data Container)
# ========================================================
# Think of this as defining what information our workflow will carry
# It's like creating a template/form with specific fields
class addition_state(TypedDict):
    """
    This defines the 'state' - the data that flows through our graph.
    
    Fields:
    - num1: First number for addition (must be float)
    - num2: Second number for addition (must be float) 
    - result: Where we'll store the answer (must be float)
    
    Why TypedDict? It tells Python (and LangGraph) exactly what data to expect.
    This helps catch errors early!
    """
    num1 : float
    num2 : float
    result: float  # This will hold our final answer


# ========================================================
# STEP 2: CREATE NODE FUNCTIONS (The Workers in Our Factory)
# ========================================================
# Nodes are like workers on an assembly line. Each does one specific job.
def addfun(state: addition_state) -> addition_state:
    """
    This is our NODE function - it performs the actual addition.
    
    Think of it as a machine that takes in the state, does work,
    and updates the state with the result.
    
    Workflow:
    1. Receives the current state (contains num1 and num2)
    2. Extracts the numbers from the state
    3. Performs the addition
    4. Updates the state with the result
    5. Returns the updated state
    
    Parameter:
    - state: The current state containing num1 and num2
    
    Returns:
    - The updated state with the 'result' field filled
    """
    
    # STEP 2a: Extract data from state
    # Think of this as opening a package to get what's inside
    num1 = state['num1']  # Get first number from the state
    num2 = state['num2']  # Get second number from the state
    
    # STEP 2b: Perform the actual work (addition in this case)
    result = num1 + num2  # Add the two numbers together
    
    # STEP 2c: Update the state with our result
    # This is like putting the finished product back in the package
    state['result'] = result  # Store the answer in the state
    
    # STEP 2d: Return the updated state
    # The package now contains both the original numbers AND the result
    return state  # Send the updated package forward


# ========================================================
# STEP 3: BUILD THE GRAPH (Create Our Assembly Line)
# ========================================================
# Now we create the actual workflow/graph that connects everything

# Create a new graph that will handle our addition_state type of data
# Think of this as building a new empty factory
graph = StateGraph(addition_state)

# ========================================================
# STEP 4: ADD NODES TO THE GRAPH (Place Workers in Factory)
# ========================================================
# We add our 'addfun' function as a node in the graph
# The first parameter 'add' is just a name we give to this node
# You can name it anything - it's like giving a worker a name tag
graph.add_node('add', addfun)  # "add" is the node name, addfun is the function

# ========================================================
# STEP 5: CONNECT THE NODES (Create the Assembly Line Path)
# ========================================================
# Now we tell our graph how data should flow between nodes

# Connect START to our 'add' node
# This means: "When we start, send data to the 'add' node first"
# START → 'add'
graph.add_edge(START, 'add')

# Connect 'add' node to END
# This means: "After the 'add' node finishes, we're done"
# 'add' → END
graph.add_edge('add', END)

# Our complete flow looks like: START → [add node] → END
# This is like: [Start conveyor] → [Addition machine] → [Finished goods area]

# ========================================================
# STEP 6: COMPILE THE GRAPH (Start Up the Factory)
# ========================================================
# This finalizes our graph and makes it ready to run
# Think of this as turning on the power to our factory
workflow = graph.compile()  # Now our workflow is ready to use!

# ========================================================
# STEP 7: PREPARE INPUT DATA (What We Want to Add)
# ========================================================
# Create our initial state - the starting package/data
# This is like preparing the raw materials for our factory
initial_state = {
    'num1': 5,      # First number to add
    'num2': 5       # Second number to add
    # Note: We don't need to include 'result' here - it will be created
}

# ========================================================
# STEP 8: RUN THE WORKFLOW (Start the Assembly Line)
# ========================================================
# Feed our initial state into the workflow and let it run
# This is like putting raw materials on the conveyor belt
final_state = workflow.invoke(initial_state)

# The workflow will:
# 1. Take initial_state
# 2. Send it to START (entry point)
# 3. START sends it to 'add' node
# 4. 'add' node performs addition, updates state
# 5. 'add' node sends updated state to END
# 6. We get the final_state back

# ========================================================
# STEP 9: GET THE RESULT (See What We Made)
# ========================================================
# Extract just the result from our final state
# This is like taking the finished product off the conveyor belt
print(final_state['result'])  # Should print: 10.0

# ========================================================
# VISUAL REPRESENTATION OF OUR GRAPH:
# ========================================================
"""
GRAPH VISUALIZATION:

        ┌─────────────┐
        │    START    │  (Entry point - where we begin)
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │     add     │  (Our addition node/worker)
        │   (addfun)  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │     END     │  (Exit point - where we finish)
        └─────────────┘

DATA FLOW:
1. initial_state = {num1: 5, num2: 5} enters at START
2. Flows to 'add' node
3. 'add' node calculates: 5 + 5 = 10
4. 'add' node updates state: {num1: 5, num2: 5, result: 10}
5. Updated state flows to END
6. We get final_state = {num1: 5, num2: 5, result: 10}
"""

# ========================================================
# BONUS: TEST WITH DIFFERENT NUMBERS
# ========================================================
print("\n" + "="*50)
print("TESTING WITH DIFFERENT NUMBERS:")
print("="*50)

# Test case 1
test1_state = {'num1': 10, 'num2': 20}
result1 = workflow.invoke(test1_state)
print(f"10 + 20 = {result1['result']}")  # Should print 30

# Test case 2  
test2_state = {'num1': 3.5, 'num2': 2.5}
result2 = workflow.invoke(test2_state)
print(f"3.5 + 2.5 = {result2['result']}")  # Should print 6.0

# Test case 3
test3_state = {'num1': 100, 'num2': -50}
result3 = workflow.invoke(test3_state)
print(f"100 + (-50) = {result3['result']}")  # Should print 50

print("="*50)
print("All tests complete! ✅")