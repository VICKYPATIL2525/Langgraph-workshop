"""
HUMAN-IN-THE-LOOP ITERATIVE WORKFLOW DEMONSTRATION
==================================================
This code demonstrates two key LangGraph concepts:
1. Human-in-the-loop: Human approves/rejects LLM output
2. Iterative workflow: Loop back based on human feedback

Workflow Flow:
START → generate → approval → 
          ↑           ↓ (if approved → END)
          └───────────┘ (if not approved → loop back)

The LLM generates descriptions, human reviews them, and we iterate until approval.
"""

# ====================================================
# IMPORT SECTION
# ====================================================
# Import necessary libraries for LangGraph workflow
from langgraph.graph import StateGraph, START, END  # Core LangGraph components
from langchain_openai import AzureChatOpenAI  # Azure OpenAI LLM interface
from typing import TypedDict  # For defining structured state types
import os  # For accessing environment variables

# Load environment variables from .env file
# This allows us to store Azure credentials securely
from dotenv import load_dotenv
load_dotenv()


# ====================================================
# AZURE OPENAI SETUP
# ====================================================
# Initialize the AzureChatOpenAI LLM with specific configuration
# This is the AI model that will generate our product descriptions
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",  # Your Azure deployment name
    temperature=0.1,  # Low temperature = less random, more focused responses
    max_tokens=1000,  # Maximum length of generated responses
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # From environment variables
    api_version=os.getenv("AZURE_OPENAI_VERSION"),  # From environment variables
    api_key=os.getenv("OPENAI_API_KEY"),  # From environment variables
)


# ====================================================
# STATE DEFINITION
# ====================================================
# Define the structure of data that flows through the workflow
# This is like creating a template/form that all nodes will use
class ProductState(TypedDict):
    """
    This defines ALL data that moves through our workflow graph.
    Each node will READ from and WRITE to specific fields of this state.
    
    Fields:
    - product_name: The product we're describing (input, never changes)
    - description: The current LLM-generated description (changes each iteration)
    - approved: Whether human approved the description (True/False)
    - attempts: How many times we've generated descriptions
    - feedback: Human's latest feedback for improvement (if any)
    """
    product_name: str      # Product to describe (e.g., "Smart Watch")
    description: str       # Current generated description
    approved: bool         # Approval status: True if human approved
    attempts: int          # Counter of how many attempts made
    feedback: str          # Latest human feedback for improvements


# ====================================================
# NODE 1: DESCRIPTION GENERATOR
# ====================================================
def generate_description(state: ProductState):
    """
    This is Node 1: Generates product descriptions using Azure OpenAI LLM.
    
    How it works:
    1. Checks if there's human feedback from previous rejection
    2. Builds appropriate prompt (with or without feedback)
    3. Calls Azure OpenAI LLM
    4. Returns ONLY the fields it updates (LangGraph merges them)
    
    Important: Returns DICT with only changed fields, not entire state.
    This is a key LangGraph pattern for efficiency.
    """
    
    # Check if we have feedback from human (means previous attempt was rejected)
    if state['feedback']:
        # Include previous feedback to improve the description
        prompt = f"Generate a product description for '{state['product_name']}'. Previous feedback: {state['feedback']}"
    else:
        # First attempt - no feedback yet
        prompt = f"Generate a short, compelling product description for '{state['product_name']}'"
    
    # Call Azure OpenAI LLM with our prompt
    result = llm.invoke(prompt).content
    
    # Return ONLY the fields we're updating (LangGraph merges this with current state)
    return {
        'description': result,  # Store the newly generated description
        'attempts': state.get('attempts', 0) + 1,  # Increment attempt counter
        'feedback': ''  # Clear feedback for next iteration (fresh start)
    }


# ====================================================
# NODE 2: HUMAN APPROVAL NODE
# ====================================================
def get_approval(state: ProductState):
    """
    This is Node 2: Gets human approval/rejection for the generated description.
    
    How it works:
    1. Shows current attempt number and generated description
    2. Asks human for approval (Yes/No) via console input
    3. If "No", asks for improvement feedback
    4. Returns appropriate state updates
    
    Note: For demo purposes, auto-approves after 3 attempts.
    """
    
    # Display current attempt information
    print(f"\n=== ATTEMPT {state['attempts']} ===")
    print(f"Product: {state['product_name']}")
    print(f"Description: {state['description']}")
    
    # Auto-approval after 3 attempts (demo safety feature)
    if state['attempts'] >= 3:  # Auto-approve after 3 attempts
        print("(Auto-approved after 3 attempts)")
        return {'approved': True}  # Return only approved field update
    
    # Get human input via console
    response = input("\nApprove this description? (y/n): ").lower()
    
    if response == 'y':
        # Human approved - we're done!
        print("✅ Approved!")
        return {'approved': True}  # Only update approved field
    else:
        # Human rejected - ask for feedback to improve
        feedback = input("What should be changed? (e.g., 'make it shorter', 'more technical'): ")
        
        # Return both that it's NOT approved AND the feedback for next iteration
        return {
            'approved': False,
            'feedback': feedback
        }


# ====================================================
# BUILD THE GRAPH STRUCTURE
# ====================================================
"""
We're building a graph with 2 nodes and conditional routing:

START → generate → approval → 
          ↑           ↓ (if approved → END)
          └───────────┘ (if not approved → generate)

This creates an iterative loop until human approval.
"""

# Create a new graph that handles ProductState type data
graph = StateGraph(ProductState)

# ====================================================
# ADD NODES TO GRAPH
# ====================================================
# Each node is a function that processes the state
graph.add_node("generate", generate_description)  # Node 1: Generate descriptions
graph.add_node("approval", get_approval)          # Node 2: Get human approval

# ====================================================
# ADD EDGES (CONNECTIONS BETWEEN NODES)
# ====================================================
# Edges define the flow of execution between nodes

# Start at the "generate" node
graph.add_edge(START, "generate")

# Always go from "generate" to "approval" (after generating, get approval)
graph.add_edge("generate", "approval")


# ====================================================
# CONDITIONAL ROUTING FUNCTION
# ====================================================
def decide_next(state: ProductState):
    """
    This function decides what happens after approval node.
    
    Logic:
    - If human approved (approved=True) → Go to END (finish)
    - If human rejected (approved=False) → Go back to "generate" (try again)
    
    Returns: Either END or "generate" (next node name)
    """
    if state['approved']:
        return END  # Finished - go to end
    else:
        return "generate"  # Not approved - try again


# ====================================================
# ADD CONDITIONAL EDGES
# ====================================================
# Conditional edges allow different paths based on state values
graph.add_conditional_edges(
    "approval",      # From this node (approval)
    decide_next,     # Use this function to decide where to go
    {
        END: END,            # If decide_next returns END, go to END
        "generate": "generate"  # If decide_next returns "generate", go to generate
    }
)


# ====================================================
# COMPILE THE GRAPH
# ====================================================
# Compile converts our graph definition into an executable workflow
workflow = graph.compile()


# ====================================================
# TEST THE WORKFLOW
# ====================================================
print("🚀 HUMAN-IN-THE-LOOP PRODUCT DESCRIPTION GENERATOR")
print("=" * 50)

# Define initial state with a test product
initial_state = {
    'product_name': 'Smart Watch',  # Our test product
    'description': '',              # Empty description (will be generated)
    'approved': False,              # Not approved yet
    'attempts': 0,                  # No attempts made yet
    'feedback': ''                  # No feedback yet
}

# ====================================================
# EXECUTE THE WORKFLOW
# ====================================================
# Run the compiled workflow with our initial state
# The workflow will:
# 1. Start at START → generate (Node 1)
# 2. generate → approval (Node 2)
# 3. Based on approval result, either END or loop back to generate
final_state = workflow.invoke(initial_state)

# ====================================================
# DISPLAY FINAL RESULTS
# ====================================================
print("\n" + "=" * 50)
print("✅ FINAL RESULT:")
print("=" * 50)
print(f"Product: {final_state['product_name']}")
print(f"Final Description: {final_state['description']}")
print(f"Total Attempts: {final_state['attempts']}")
print("=" * 50)


# ====================================================
# WORKFLOW VISUALIZATION
# ====================================================
"""
GRAPH STRUCTURE VISUALIZED:

        ┌─────────────┐
        │    START    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   generate  │ ← Creates description using LLM
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   approval  │ ← Human approves/rejects
        └──────┬──────┘
               │
         ┌─────┴─────┐
         │           │
         ▼           ▼
    approved?     not approved
         │           │
         ▼           ▼
        END       generate (loop back)
"""


# ====================================================
# KEY LEARNING POINTS
# ====================================================
"""
WHAT THIS CODE DEMONSTRATES:

1. HUMAN-IN-THE-LOOP: 
   - AI generates output → Human reviews → AI improves → Loop continues
   - Human has final say (approval/rejection)

2. ITERATIVE WORKFLOW:
   - Can loop back to previous nodes
   - Continues until condition met (human approval)

3. STATE MANAGEMENT:
   - State carries all data between nodes
   - Each node updates only what it changes
   - LangGraph automatically merges updates

4. CONDITIONAL ROUTING:
   - Different paths based on state values
   - Dynamic workflow execution

TRY IT WITH:
- Different products
- Different feedback types
- Different approval thresholds
"""