# test3.py - Parallel LLM nodes workflow
# Three nodes (grammar, sentiment, clarity) all start from START at the same time
# Each node scores one aspect of the essay and updates ONLY its own field in state
# Finalizer node runs after all three finish and combines the scores into a report
#
# Flow:
#          START
#        /   |   \
#   grammar  sentiment  clarity   <- run in parallel
#        \   |   /
#        finalizer
#            |
#           END
#
# Sample essays to try (paste into initial_state['essay']):
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

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=100,    # small limit since we only need a number back
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# State holds the essay input and all three scores + final report
class ParallelState(TypedDict):
    essay: str             # input essay to analyze
    grammar_score: int     # filled by grammar_node
    sentiment_score: int   # filled by sentiment_node
    clarity_score: int     # filled by clarity_node
    final_result: str      # filled by finalizer_node

# ============================================
# PARALLEL NODES - each returns ONLY its own field
# Returning partial state is correct in parallel execution
# LangGraph merges all partial updates automatically
# ============================================

def grammar_node(state: ParallelState) -> Dict:
    print("[Grammar Node] Analyzing grammar...")

    prompt = f"""Analyze the grammar of this essay and give a score out of 100:

Essay: "{state['essay']}"

Consider:
- Spelling errors
- Punctuation
- Sentence structure

Return ONLY a number between 0-100:"""

    response = llm.invoke(prompt).content.strip()

    # LLM might return "85" or "85/100" - extract just the digits
    score = int(''.join(filter(str.isdigit, response))[:3])
    score = min(score, 100)  # cap at 100 in case of parsing edge case

    print(f"    [OK] Grammar score: {score}/100")

    return {'grammar_score': score}  # return ONLY the field this node owns

def sentiment_node(state: ParallelState) -> Dict:
    print("[Sentiment Node] Analyzing sentiment...")

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

    print(f"    [OK] Sentiment score: {score}/100")

    return {'sentiment_score': score}  # return ONLY the field this node owns

def clarity_node(state: ParallelState) -> Dict:
    print("[Clarity Node] Analyzing clarity...")

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

    print(f"    [OK] Clarity score: {score}/100")

    return {'clarity_score': score}  # return ONLY the field this node owns

# ============================================
# FINALIZER NODE - runs after all 3 parallel nodes complete
# At this point state has all three scores filled in
# ============================================

def finalizer_node(state: ParallelState) -> Dict:
    print("\n[Finalizer Node] Combining all scores...")

    # All three scores are now available in state
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

    return {'final_result': final_result}  # return ONLY the field this node owns

# ============================================
# BUILD GRAPH
# ============================================

print("BUILDING PARALLEL LLM ANALYZER - CORRECT VERSION")
print("=" * 60)
print("Key: Each node returns ONLY the state fields it updates")
print("=" * 60)

graph = StateGraph(ParallelState)

# Register all four nodes
graph.add_node("grammar", grammar_node)
graph.add_node("sentiment", sentiment_node)
graph.add_node("clarity", clarity_node)
graph.add_node("finalizer", finalizer_node)

print("\n[OK] Added nodes that return partial state updates:")

# All three analysis nodes connect FROM START - this makes them run in parallel
print("\nMaking TRUE parallel connections:")
print("   START -> [Grammar, Sentiment, Clarity] (parallel)")
print("   All nodes -> Finalizer")

graph.add_edge(START, "grammar")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "clarity")

# All three must finish before finalizer runs (LangGraph waits automatically)
graph.add_edge("grammar", "finalizer")
graph.add_edge("sentiment", "finalizer")
graph.add_edge("clarity", "finalizer")

graph.add_edge("finalizer", END)

print("\n[OK] Graph ready with proper parallel execution!")

workflow = graph.compile()

# ============================================
# RUN
# ============================================

print("\n" + "=" * 60)
print("PARALLEL LLM ANALYSIS - PROPER STATE MANAGEMENT")
print("=" * 60)

essay = "Artificial intelligence is revolutionizing education by providing personalized learning experiences for students and valuable tools for teachers."
print(f"\nEssay to analyze:\n'{essay}'")
print("-" * 60)

# Initialize all score fields to 0 - the parallel nodes will fill them
initial_state = {
    'essay': essay,
    'grammar_score': 0,
    'sentiment_score': 0,
    'clarity_score': 0,
    'final_result': ''
}

print("\nRunning parallel analysis...")
print("\nNote: Each node runs in parallel and returns ONLY what it updates")
print("   LangGraph automatically merges all updates into final state")
print("-" * 60)

result = workflow.invoke(initial_state)

# Display results
print("\n" + "=" * 60)
print("FINAL STATE AFTER PARALLEL EXECUTION:")
print("=" * 60)

print("\nIndividual Scores (updated by parallel nodes):")
print(f"  - Grammar score: {result['grammar_score']}/100")
print(f"  - Sentiment score: {result['sentiment_score']}/100")
print(f"  - Clarity score: {result['clarity_score']}/100")

avg = (result['grammar_score'] + result['sentiment_score'] + result['clarity_score']) / 3
print(f"\nAverage Score: {avg:.1f}/100")

print("\nFinal Report (generated by finalizer node):")
print("-" * 40)
print(result['final_result'])
print("=" * 60)
