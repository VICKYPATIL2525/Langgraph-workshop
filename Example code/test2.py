# test2.py - Two LLM nodes chained in sequence
# Node 1 generates a blog outline, Node 2 uses that outline to write the full blog
# Output of node 1 becomes the input of node 2 via shared state
# Flow: START -> outline_generator -> content_generator -> END

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

# One LLM instance shared across both nodes
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=500,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# State grows as it moves through the graph:
# After node 1: {title, outline}
# After node 2: {title, outline, content}
class BlogState(TypedDict):
    title: str      # Blog title from user
    outline: str    # Generated outline from LLM (node 1 writes this)
    content: str    # Final blog content from LLM (node 2 writes this)

# Node 1: takes the title, asks LLM to create an outline
def generate_outline(state: BlogState) -> BlogState:
    title = state['title']

    prompt = f"Create a detailed outline for a blog about: {title}"

    # LLM returns the outline as a string
    outline = llm.invoke(prompt).content

    state['outline'] = outline  # store outline so node 2 can use it

    return state

# Node 2: takes both title and outline from state, asks LLM to write the full blog
def generate_content(state: BlogState) -> BlogState:
    title = state['title']
    outline = state['outline']   # this was written by node 1

    prompt = f"Write a detailed blog post with this title: {title}\n\nUse this outline:\n{outline}"

    content = llm.invoke(prompt).content

    state['content'] = content

    return state

# Build graph with two sequential nodes
graph = StateGraph(BlogState)

graph.add_node('outline_generator', generate_outline)
graph.add_node('content_generator', generate_content)

# Chain: START -> node1 -> node2 -> END
graph.add_edge(START, 'outline_generator')
graph.add_edge('outline_generator', 'content_generator')
graph.add_edge('content_generator', END)

workflow = graph.compile()

# Only title is needed - outline and content will be filled by the nodes
initial_state = {'title': 'The Future of Artificial Intelligence'}
final_state = workflow.invoke(initial_state)

print("Title:", final_state['title'])
print("\nOutline:\n", final_state['outline'])
print("\nContent:\n", final_state['content'])
