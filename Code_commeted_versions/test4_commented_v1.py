# this code is to demonstrate the conditional edges in langgraph using the customer feedback
"""
CUSTOMER FEEDBACK ANALYSIS WORKFLOW
====================================
This demonstrates a conditional workflow using LangGraph and Azure OpenAI.

Workflow Flow:
START → check_feedback → 
                ├─(positive)→ thank_you → END
                └─(negative)→ apology → END

Each node returns ONLY the state fields it changes.
This is the correct way to update state in LangGraph.
"""

# ====================================================
# IMPORT STATEMENTS
# ====================================================
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from typing import TypedDict
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ====================================================
# AZURE OPENAI SETUP
# ====================================================
"""
Initialize AzureChatOpenAI with your deployment settings:
- deployment_name: Your Azure deployment name for GPT-4.1-mini
- temperature: 0.1 = low randomness, consistent responses
- max_tokens: Maximum response length
- azure_endpoint, api_version, api_key: From your .env file

The .env file should contain:
OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_VERSION=2024-12-01-preview
"""
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.1,
    max_tokens=1000,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ====================================================
# STATE DEFINITION
# ====================================================
"""
The State is a TypedDict that defines ALL data flowing through the graph.
Each node will READ from this state and RETURN ONLY the fields it changes.

Fields:
- feedback: Original customer feedback (input only, never changed)
- sentiment: Result from Node 1 (positive/negative)
- response: Generated response from Node 2 or Node 3

IMPORTANT: Each node returns a dictionary with ONLY the keys it updates.
LangGraph automatically merges these changes into the full state.
"""
class State(TypedDict):
    feedback: str      # Customer's original feedback (read-only)
    sentiment: str     # Analysis result: "positive" or "negative"
    response: str      # Generated response message

# ====================================================
# NODE 1: SENTIMENT ANALYSIS
# ====================================================
def check_feedback(state: State):
    """
    Node 1: Analyzes customer feedback using LLM.
    
    Flow:
    1. Takes current state with 'feedback' field
    2. Creates simple prompt asking for sentiment
    3. Calls Azure OpenAI LLM
    4. Returns ONLY the 'sentiment' field
    
    Returns: {'sentiment': 'positive'} or {'sentiment': 'negative'}
    
    Note: We use .strip().lower() to clean the LLM response
    """
    # Create simple prompt
    prompt = f"Is this feedback positive or negative? Answer only 'positive' or 'negative': {state['feedback']}"
    
    # Call Azure OpenAI
    result = llm.invoke(prompt).content
    
    # Return ONLY the changed field (sentiment)
    # .strip() removes whitespace, .lower() makes it lowercase
    return {'sentiment': result.strip().lower()}

# ====================================================
# NODE 2: POSITIVE RESPONSE GENERATOR
# ====================================================
def thank_you(state: State):
    """
    Node 2: Generates thank you message for POSITIVE feedback.
    
    Flow:
    1. Only runs if sentiment == "positive"
    2. Creates prompt for thank you message
    3. Calls Azure OpenAI LLM
    4. Returns ONLY the 'response' field
    
    Returns: {'response': 'Thank you message...'}
    """
    # Create simple prompt
    prompt = f"Say thank you for positive feedback: {state['feedback']}"
    
    # Call Azure OpenAI
    result = llm.invoke(prompt)
    
    # Return ONLY the changed field (response)
    return {'response': result.content}

# ====================================================
# NODE 3: NEGATIVE RESPONSE GENERATOR
# ====================================================
def apology(state: State):
    """
    Node 3: Generates apology message for NEGATIVE feedback.
    
    Flow:
    1. Only runs if sentiment == "negative"
    2. Creates prompt for apology message
    3. Calls Azure OpenAI LLM
    4. Returns ONLY the 'response' field
    
    Returns: {'response': 'Apology message...'}
    """
    # Create simple prompt
    prompt = f"Apologize and say we'll contact support about: {state['feedback']}"
    
    # Call Azure OpenAI
    result = llm.invoke(prompt)
    
    # Return ONLY the changed field (response)
    return {'response': result.content}

# ====================================================
# BUILD THE GRAPH
# ====================================================
"""
Graph Structure:
START → check → 
              ├─(positive)→ thank → END
              └─(negative)→ sorry → END

Key Components:
1. Nodes: Functions that do work
2. Edges: Define flow between nodes
3. Conditional Edges: Route based on state values
"""
graph = StateGraph(State)

# Add nodes (workers in the assembly line)
graph.add_node("check", check_feedback)  # Analyzes sentiment
graph.add_node("thank", thank_you)       # Positive response
graph.add_node("sorry", apology)         # Negative response

# Start at the 'check' node
graph.add_edge(START, "check")

# ====================================================
# CONDITIONAL ROUTING FUNCTION
# ====================================================
def decide_next(state: State):
    """
    This function decides which path to take after sentiment analysis.
    
    Logic:
    - If sentiment == "positive" → go to "thank" node
    - If sentiment == "negative" → go to "sorry" node
    
    Returns: The name of the next node to execute
    """
    if state['sentiment'] == "positive":
        return "thank"
    else:
        return "sorry"

# Add conditional edges
graph.add_conditional_edges(
    "check",          # From this node
    decide_next,      # Use this function to decide
    {
        "thank": "thank",  # If returns "thank", go to thank node
        "sorry": "sorry"   # If returns "sorry", go to sorry node
    }
)

# Connect final nodes to END
graph.add_edge("thank", END)
graph.add_edge("sorry", END)

# Compile the graph into a runnable workflow
workflow = graph.compile()

# ====================================================
# TEST THE WORKFLOW
# ====================================================
print("=" * 60)
print("CUSTOMER FEEDBACK ANALYSIS WORKFLOW")
print("=" * 60)

# Test Case 1: Positive Feedback
print("\n📝 Test 1: Positive Feedback")
good_feedback = "I love this product! Very good."

# Initial state for test 1
initial_state_1 = {
    "feedback": good_feedback,
    "sentiment": "",  # Will be filled by Node 1
    "response": ""    # Will be filled by Node 2 or 3
}

# Run the workflow
result1 = workflow.invoke(initial_state_1)

print(f"   Input: '{good_feedback}'")
print(f"   Sentiment: {result1['sentiment']}")
print(f"   Response: {result1['response']}")
print(f"   Path: check → thank → END")

# Test Case 2: Negative Feedback
print("\n📝 Test 2: Negative Feedback")
bad_feedback = "This product is terrible. Worst ever."

# Initial state for test 2
initial_state_2 = {
    "feedback": bad_feedback,
    "sentiment": "",  # Will be filled by Node 1
    "response": ""    # Will be filled by Node 2 or 3
}

# Run the workflow
result2 = workflow.invoke(initial_state_2)

print(f"   Input: '{bad_feedback}'")
print(f"   Sentiment: {result2['sentiment']}")
print(f"   Response: {result2['response']}")
print(f"   Path: check → sorry → END")

print("\n" + "=" * 60)
print("WORKFLOW COMPLETE ✅")
print("=" * 60)

# ====================================================
# ADDITIONAL TEST CASES
# ====================================================
print("\n" + "=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

test_cases = [
    "Excellent service, very helpful!",
    "Terrible experience, never buying again.",
    "The quality is perfect, best purchase ever.",
    "Very disappointed with the customer service."
]

for i, feedback in enumerate(test_cases, 3):
    print(f"\n📝 Test {i}: '{feedback}'")
    
    result = workflow.invoke({
        "feedback": feedback,
        "sentiment": "",
        "response": ""
    })
    
    # Determine which path was taken
    if result['sentiment'] == "positive":
        path = "check → thank → END"
    else:
        path = "check → sorry → END"
    
    print(f"   Sentiment: {result['sentiment']}")
    print(f"   Response: {result['response'][:50]}...")
    print(f"   Path: {path}")

print("\n" + "=" * 60)
print("KEY CONCEPT: Each node returns ONLY what it changes")
print("=" * 60)

print("""
NODE RETURN VALUES:
------------------
check_feedback: Returns {'sentiment': 'positive/negative'}
thank_you:      Returns {'response': 'Thank you message...'}
apology:        Returns {'response': 'Apology message...'}

LangGraph automatically merges these partial updates into the full state.

This is more efficient than returning the entire state and helps prevent
accidentally overwriting fields that other nodes might update.
""")