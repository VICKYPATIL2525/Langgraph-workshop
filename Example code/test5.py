# Human-in-the-loop iterative workflow for product descriptions
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from typing import TypedDict
import os

from dotenv import load_dotenv
load_dotenv()

# Setup Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.1,
    max_tokens=1000,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define state
class ProductState(TypedDict):
    product_name: str      # Input product name
    description: str       # Generated description
    approved: bool         # Human approval status
    attempts: int          # Number of attempts
    feedback: str          # Human feedback for revision

# Node 1: Generate product description
def generate_description(state: ProductState):
    # If we have feedback, use it to improve
    if state['feedback']:
        prompt = f"Generate a product description for '{state['product_name']}'. Previous feedback: {state['feedback']}"
    else:
        prompt = f"Generate a short, compelling product description for '{state['product_name']}'"
    
    # Get LLM response
    result = llm.invoke(prompt).content
    
    # Update state
    return {
        'description': result,
        'attempts': state.get('attempts', 0) + 1,
        'feedback': ''  # Clear feedback for next iteration
    }

# Node 2: Get human approval (simulated)
def get_approval(state: ProductState):
    print(f"\n=== ATTEMPT {state['attempts']} ===")
    print(f"Product: {state['product_name']}")
    print(f"Description: {state['description']}")
    
    # Simulate human approval
    if state['attempts'] >= 3:  # Auto-approve after 3 attempts
        print("(Auto-approved after 3 attempts)")
        return {'approved': True}
    
    response = input("\nApprove this description? (y/n): ").lower()
    
    if response == 'y':
        print("✅ Approved!")
        return {'approved': True}
    else:
        feedback = input("What should be changed? (e.g., 'make it shorter', 'more technical'): ")
        return {
            'approved': False,
            'feedback': feedback
        }

# Build graph
graph = StateGraph(ProductState)

# Add nodes
graph.add_node("generate", generate_description)
graph.add_node("approval", get_approval)

# Add edges
graph.add_edge(START, "generate")
graph.add_edge("generate", "approval")

# Conditional routing
def decide_next(state: ProductState):
    if state['approved']:
        return END  # Go to END if approved
    else:
        return "generate"  # Go back to generate if not approved

graph.add_conditional_edges(
    "approval",
    decide_next,
    {
        END: END,  # If END is returned, go to END
        "generate": "generate"  # If "generate" is returned, go to generate
    }
)

# Compile
workflow = graph.compile()

# Test it
print("🚀 HUMAN-IN-THE-LOOP PRODUCT DESCRIPTION GENERATOR")
print("=" * 50)

# Test product
initial_state = {
    'product_name': 'Smart Watch',
    'description': '',
    'approved': False,
    'attempts': 0,
    'feedback': ''
}

# Run workflow
final_state = workflow.invoke(initial_state)

print("\n" + "=" * 50)
print("✅ FINAL RESULT:")
print(f"Product: {final_state['product_name']}")
print(f"Final Description: {final_state['description']}")
print(f"Total Attempts: {final_state['attempts']}")