"""
PARALLEL LLM ANALYSIS WORKFLOW WE ARE EXECUTING MULTIPLE NODES AT A TIME TO REVIEW THE ESSAY
Each node returns ONLY the state fields it updates
"""
# demo essay for testing just copy and paste it in to the initial state.
# essay 1: Critical thinking teaches us to analyze information objectively rather than accepting it at face value.
# essay 2: Social media connects people globally but often reduces the depth of our personal interactions.
# essay 3: Renewable energy offers sustainable solutions for both environmental protection and economic growth.
# essay 4: Liberal arts education develops essential human skills that complement technical expertise in our digital age.
# essay 5: Personal resilience comes from adapting to challenges while maintaining core values and purpose.

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=100,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# State definition
class ParallelState(TypedDict):
    essay: str
    grammar_score: int
    sentiment_score: int
    clarity_score: int
    final_result: str

# ============================================
# 3 PARALLEL ANALYSIS NODES (Correct: Return only updates)
# ============================================

def grammar_node(state: ParallelState) -> Dict:
    """Node 1: Analyze grammar - Returns ONLY grammar_score update"""
    print("🤖 [Grammar Node] Analyzing grammar...")
    
    prompt = f"""Analyze the grammar of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider:
- Spelling errors
- Punctuation
- Sentence structure

Return ONLY a number between 0-100:"""
    
    response = llm.invoke(prompt).content.strip()
    score = int(''.join(filter(str.isdigit, response))[:3])
    score = min(score, 100)
    
    print(f"    ✅ Grammar score: {score}/100")
    
    # Return ONLY what we're updating
    return {'grammar_score': score}

def sentiment_node(state: ParallelState) -> Dict:
    """Node 2: Analyze sentiment - Returns ONLY sentiment_score update"""
    print("🤖 [Sentiment Node] Analyzing sentiment...")
    
    prompt = f"""Analyze the sentiment of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider:
- Overall emotional tone
- Positivity/negativity
- Engagement level

Return ONLY a number between 0-100:"""
    
    response = llm.invoke(prompt).content.strip()
    score = int(''.join(filter(str.isdigit, response))[:3])
    score = min(score, 100)
    
    print(f"    ✅ Sentiment score: {score}/100")
    
    # Return ONLY what we're updating
    return {'sentiment_score': score}

def clarity_node(state: ParallelState) -> Dict:
    """Node 3: Analyze clarity - Returns ONLY clarity_score update"""
    print("🤖 [Clarity Node] Analyzing clarity...")
    
    prompt = f"""Analyze the clarity of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider:
- Ease of understanding
- Clear expression
- Logical flow

Return ONLY a number between 0-100:"""
    
    response = llm.invoke(prompt).content.strip()
    score = int(''.join(filter(str.isdigit, response))[:3])
    score = min(score, 100)
    
    print(f"    ✅ Clarity score: {score}/100")
    
    # Return ONLY what we're updating
    return {'clarity_score': score}

# ============================================
# FINALIZER NODE
# ============================================

def finalizer_node(state: ParallelState) -> Dict:
    """Node 4: Combine all scores - Returns ONLY final_result update"""
    print("\n📊 [Finalizer Node] Combining all scores...")
    
    # Calculate average
    avg_score = (state['grammar_score'] + state['sentiment_score'] + state['clarity_score']) / 3
    
    prompt = f"""Generate a comprehensive analysis report based on these scores:

Essay: "{state['essay']}"

Scores:
- Grammar: {state['grammar_score']}/100
- Sentiment: {state['sentiment_score']}/100  
- Clarity: {state['clarity_score']}/100

Average Score: {avg_score:.1f}/100

Create a detailed report with:
1. Overall assessment
2. Strengths
3. Areas for improvement
4. Specific recommendations

Format the report clearly:"""
    
    final_result = llm.invoke(prompt).content
    
    # Return ONLY what we're updating
    return {'final_result': final_result}

# ============================================
# BUILD PARALLEL WORKFLOW
# ============================================

print("🏗️  BUILDING PARALLEL LLM ANALYZER - CORRECT VERSION")
print("=" * 60)
print("Key: Each node returns ONLY the state fields it updates")
print("=" * 60)

# Create graph
graph = StateGraph(ParallelState)

# Add nodes
graph.add_node("grammar", grammar_node)
graph.add_node("sentiment", sentiment_node)
graph.add_node("clarity", clarity_node)
graph.add_node("finalizer", finalizer_node)

print("\n✅ Added nodes that return partial state updates:")

# TRUE PARALLEL CONNECTIONS
print("\n🔗 Making TRUE parallel connections:")
print("   START → [Grammar, Sentiment, Clarity] (parallel)")
print("   All nodes → Finalizer")

graph.add_edge(START, "grammar")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "clarity")

graph.add_edge("grammar", "finalizer")
graph.add_edge("sentiment", "finalizer")
graph.add_edge("clarity", "finalizer")

graph.add_edge("finalizer", END)

print("\n✅ Graph ready with proper parallel execution!")

# Compile
workflow = graph.compile()

# ============================================
# DEMONSTRATION
# ============================================

print("\n" + "=" * 60)
print("🚀 PARALLEL LLM ANALYSIS - PROPER STATE MANAGEMENT")
print("=" * 60)

# Sample essay
essay = "Artificial intelligence is revolutionizing education by providing personalized learning experiences for students and valuable tools for teachers."
print(f"\n📝 Essay to analyze:\n'{essay}'")
print("-" * 60)

# Initial state
initial_state = {
    'essay': essay,
    'grammar_score': 0,
    'sentiment_score': 0,
    'clarity_score': 0,
    'final_result': ''
}

print("\n🚀 Running parallel analysis...")
print("\n💡 Notice: Each node runs in parallel and returns ONLY what it updates")
print("   LangGraph automatically merges all updates into final state")
print("-" * 60)

# Run workflow
result = workflow.invoke(initial_state)

# Show results
print("\n" + "=" * 60)
print("📊 FINAL STATE AFTER PARALLEL EXECUTION:")
print("=" * 60)

print("\n🎯 Individual Scores (updated by parallel nodes):")
print(f"  • Grammar score: {result['grammar_score']}/100")
print(f"  • Sentiment score: {result['sentiment_score']}/100")
print(f"  • Clarity score: {result['clarity_score']}/100")

avg = (result['grammar_score'] + result['sentiment_score'] + result['clarity_score']) / 3
print(f"\n📈 Average Score: {avg:.1f}/100")

print("\n📄 Final Report (generated by finalizer node):")
print("-" * 40)
print(result['final_result'])
print("=" * 60)