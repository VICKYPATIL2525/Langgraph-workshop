# test5.py - Human-in-the-loop iterative workflow
# The graph loops back until a human approves the output
# Node 1 generates a description, Node 2 asks the human to approve it
# If rejected, human provides feedback and node 1 runs again with that feedback
# Auto-approves after 3 attempts to prevent infinite loops
#
# Flow:
#   START -> generate -> approval -> (approved) -> END
#                 ^          |
#                 |          v (not approved)
#                 +----------+   (loop back with feedback)

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict
import os

from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=1000,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# State tracks the product, the current description, approval status,
# attempt count, and any feedback from the human
class ProductState(TypedDict):
    product_name: str      # Input product name (never changes)
    description: str       # Current generated description (changes each loop)
    approved: bool         # True when human approves, loop stops
    attempts: int          # How many times we've generated a description
    feedback: str          # Human's feedback for the next attempt (cleared after use)

# Node 1: generates or re-generates the description
# On first run: no feedback, generates fresh
# On re-run: uses human feedback to improve the previous description
def generate_description(state: ProductState):
    if state['feedback']:
        # Human rejected last time - incorporate their feedback
        prompt = f"Generate a product description for '{state['product_name']}'. Previous feedback: {state['feedback']}"
    else:
        # First attempt - no feedback yet
        prompt = f"Generate a short, compelling product description for '{state['product_name']}'"

    result = llm.invoke(prompt).content

    return {
        'description': result,
        'attempts': state.get('attempts', 0) + 1,  # increment attempt counter
        'feedback': ''    # clear feedback so it doesn't carry over to next attempt
    }

# Node 2: shows the description to the human and asks for approval
# Returns {'approved': True} or {'approved': False, 'feedback': '...'}
def get_approval(state: ProductState):
    print(f"\n=== ATTEMPT {state['attempts']} ===")
    print(f"Product: {state['product_name']}")
    print(f"Description: {state['description']}")

    # Safety exit - auto-approve after 3 attempts so the loop doesn't run forever
    if state['attempts'] >= 3:
        print("(Auto-approved after 3 attempts)")
        return {'approved': True}

    # Ask human via console input
    response = input("\nApprove this description? (y/n): ").lower()

    if response == 'y':
        print("Approved!")
        return {'approved': True}
    else:
        # Human wants changes - collect their feedback for the next generate call
        feedback = input("What should be changed? (e.g., 'make it shorter', 'more technical'): ")
        return {
            'approved': False,
            'feedback': feedback
        }

# Build graph
graph = StateGraph(ProductState)

graph.add_node("generate", generate_description)
graph.add_node("approval", get_approval)

# Always start by generating
graph.add_edge(START, "generate")
# After generating, always go to approval
graph.add_edge("generate", "approval")

# Routing function: after approval node, check if approved
# Returns END to stop, or "generate" to loop back
def decide_next(state: ProductState):
    if state['approved']:
        return END        # done - exit the graph
    else:
        return "generate" # not approved - go back and try again

# Conditional edge on the approval node
graph.add_conditional_edges(
    "approval",
    decide_next,
    {
        END: END,
        "generate": "generate"
    }
)

workflow = graph.compile()

print("HUMAN-IN-THE-LOOP PRODUCT DESCRIPTION GENERATOR")
print("=" * 50)

# Initial state - description/feedback start empty, attempts start at 0
initial_state = {
    'product_name': 'Smart Watch',
    'description': '',
    'approved': False,
    'attempts': 0,
    'feedback': ''
}

# Run - will loop until approved or 3 attempts reached
final_state = workflow.invoke(initial_state)

print("\n" + "=" * 50)
print("FINAL RESULT:")
print(f"Product: {final_state['product_name']}")
print(f"Final Description: {final_state['description']}")
print(f"Total Attempts: {final_state['attempts']}")
