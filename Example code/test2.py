# In this workflow we are generating the blog using 2 llm's, the 2nd llm generate the blog content based on the 1st llm's outline  
# Import necessary libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=500,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)



# Define state structure
class BlogState(TypedDict):
    title: str      # Blog title from user
    outline: str    # Generated outline from LLM
    content: str    # Final blog content from LLM

# Node 1: Generate outline from title
def generate_outline(state: BlogState) -> BlogState:
    # Extract title from state
    title = state['title']
    
    # Create prompt for outline generation
    prompt = f"Create a detailed outline for a blog about: {title}"
    
    # Get outline from LLM
    outline = llm.invoke(prompt).content
    
    # Update state with outline
    state['outline'] = outline
    
    return state

# Node 2: Generate blog content from outline
def generate_content(state: BlogState) -> BlogState:
    # Extract title and outline from state
    title = state['title']
    outline = state['outline']
    
    # Create prompt for blog generation
    prompt = f"Write a detailed blog post with this title: {title}\n\nUse this outline:\n{outline}"
    
    # Get blog content from LLM
    content = llm.invoke(prompt).content
    
    # Update state with content
    state['content'] = content
    
    return state

# Create graph
graph = StateGraph(BlogState)

# Add nodes to graph
graph.add_node('outline_generator', generate_outline)
graph.add_node('content_generator', generate_content)

# Add edges to connect nodes
graph.add_edge(START, 'outline_generator')
graph.add_edge('outline_generator', 'content_generator')
graph.add_edge('content_generator', END)

# Compile graph
workflow = graph.compile()

# Example usage
initial_state = {'title': 'The Future of Artificial Intelligence'}
final_state = workflow.invoke(initial_state)

# Print results
print("Title:", final_state['title'])
print("\nOutline:\n", final_state['outline'])
print("\nContent:\n", final_state['content'])