# test4.py - Conditional routing workflow
# The graph takes different paths based on LLM output (sentiment)
# If feedback is positive -> thank_you node
# If feedback is negative -> apology node
#
# Flow:
#   START -> check -> (positive) -> thanknode -> END
#                  -> (negative) -> sorrynode -> END

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

# State carries the feedback in and the sentiment + response out
class State(TypedDict):
    feedback: str      # Customer feedback (input, never modified)
    sentiment: str     # "positive" or "negative" - set by check_feedback node
    response: str      # Final reply - set by either thank_you or apology node

# Node 1: asks LLM to classify the sentiment - runs first always
def check_feedback(state: State):
    prompt = f"Is this feedback positive or negative? Answer only 'positive' or 'negative': {state['feedback']}"

    result = llm.invoke(prompt).content

    # Return ONLY the field this node updates
    return {'sentiment': result}

# Node 2: runs ONLY if sentiment == "positive"
def thank_you(state: State):
    prompt = f"Say thank you for positive feedback: {state['feedback']}"

    result = llm.invoke(prompt).content

    return {'response': result}

# Node 3: runs ONLY if sentiment == "negative"
def apology(state: State):
    prompt = f"Apologize and say we'll contact support about: {state['feedback']}"

    result = llm.invoke(prompt).content

    return {'response': result}

# Build graph
graph = StateGraph(State)

graph.add_node("check", check_feedback)
graph.add_node("thanknode", thank_you)
graph.add_node("sorrynode", apology)

# Always start at check node
graph.add_edge(START, "check")

# Routing function - reads the sentiment from state and returns the next node name
# This function is called after "check" node finishes
def decide_next(state: State):
    if state['sentiment'] == "positive":
        return "thank"   # maps to "thanknode" below
    else:
        return "sorry"   # maps to "sorrynode" below

# Conditional edge: after "check", call decide_next() to pick which node runs next
# The dict maps the return value of decide_next to actual node names
graph.add_conditional_edges(
    "check",          # from this node
    decide_next,      # call this function to decide
    {
        "thank": "thanknode",   # if decide_next returns "thank", go to thanknode
        "sorry": "sorrynode"    # if decide_next returns "sorry", go to sorrynode
    }
)

# Both response nodes lead to END
graph.add_edge("thanknode", END)
graph.add_edge("sorrynode", END)

workflow = graph.compile()

# Test with positive feedback
good_feedback = "I love this product! Very good."
bad_feedback = "This product is terrible. Worst ever."

print("Testing GOOD feedback:")
result1 = workflow.invoke({"feedback": good_feedback})
print(f"Sentiment: {result1['sentiment']}")
print(f"Response: {result1['response']}")

print("\nTesting BAD feedback:")
result2 = workflow.invoke({"feedback": bad_feedback})
print(f"Sentiment: {result2['sentiment']}")
print(f"Response: {result2['response']}")
