"""
PARALLEL LLM ANALYSIS WORKFLOW WITH DETAILED EXPLANATION
================================================================
This example demonstrates a true parallel workflow where:
1. Three analysis nodes run in parallel from the START node
2. Each node uses LLM to analyze a different aspect of an essay
3. Each node returns ONLY the state field it updates (not entire state)
4. All results are combined in a finalizer node
5. LangGraph automatically merges all partial state updates

KEY CONCEPT: In LangGraph, when multiple nodes run in parallel, 
each should return a dictionary containing ONLY the state fields 
that it modifies. LangGraph will automatically merge all these
partial updates into the complete state.

WORKFLOW VISUALIZATION:
        [START]
       /   |   \
      /    |    \
     ↓     ↓     ↓
[Grammar] [Sentiment] [Clarity]   ← All run in parallel!
     \     |     /
      \    |    /
       \   |   /
        \  |  /
         ↓ ↓ ↓
      [Finalizer]
           ↓
          [END]
"""

# =================================================================
# SECTION 1: IMPORT STATEMENTS AND SETUP
# =================================================================

# Import necessary libraries
from langgraph.graph import StateGraph, START, END  # Core LangGraph components
from typing import TypedDict, Dict                  # For type hints and state definition
from langchain_anthropic import ChatAnthropic        # Anthropic Claude LLM integration
from dotenv import load_dotenv                      # For loading environment variables
import os                                           # For accessing environment variables

# Load environment variables from .env file
# This keeps API keys and configuration secret
load_dotenv()

# =================================================================
# SECTION 2: LLM INITIALIZATION
# =================================================================

"""
Initialize the Language Model (LLM) that will be used for analysis.

Why ChatAnthropic?
- Integrates with Anthropic Claude models
- Provides a consistent interface for Claude models
- Handles authentication and API calls automatically

Parameters Explained:
- model: Specific Claude model variant (claude-haiku-4-5-20251001 is fast and efficient)
- temperature: Controls randomness (0.1 = very deterministic)
- max_tokens: Maximum response length (100 tokens ≈ 75 words)
- api_key: Authentication key for Anthropic API

This LLM object will be used by all analysis nodes.
"""
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=100,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# =================================================================
# SECTION 3: STATE DEFINITION USING TypedDict
# =================================================================

"""
Define the structure of our workflow state using TypedDict.

Think of the state as a "shared container" that flows through the workflow.
Each node can read from and write to specific parts of this container.

State Fields:
1. essay: The text to analyze (input, never modified)
2. grammar_score: Score from grammar analysis (0-100)
3. sentiment_score: Score from sentiment analysis (0-100)
4. clarity_score: Score from clarity analysis (0-100)
5. final_result: Combined report from finalizer

Why use TypedDict?
- Provides type hints for better code completion
- Documents what data is expected
- Helps catch errors early
- Makes code self-documenting
"""
class ParallelState(TypedDict):
    """State container for parallel essay analysis workflow"""
    essay: str            # Original essay text (input)
    grammar_score: int    # Grammar analysis score (0-100)
    sentiment_score: int  # Sentiment analysis score (0-100)
    clarity_score: int    # Clarity analysis score (0-100)
    final_result: str     # Final combined report

# =================================================================
# SECTION 4: PARALLEL ANALYSIS NODE FUNCTIONS
# =================================================================

"""
IMPORTANT: Each analysis node function follows this pattern:
1. Receives the current state
2. Uses LLM to analyze one specific aspect
3. Returns ONLY the state field it updates (as a dictionary)

This is CRITICAL for parallel execution because:
- Multiple nodes might run simultaneously
- Each should only modify its own part of the state
- LangGraph merges all partial updates automatically

Returning only what you change prevents conflicts and makes the
workflow more robust and maintainable.
"""

def grammar_node(state: ParallelState) -> Dict:
    """
    NODE 1: Grammar Analysis Specialist
    
    This node analyzes the grammar of the essay and returns a score.
    It ONLY modifies the 'grammar_score' field in the state.
    
    Workflow:
    1. Extract essay from state
    2. Create prompt asking LLM to analyze grammar
    3. Send prompt to LLM and get response
    4. Extract numerical score from response
    5. Return {'grammar_score': score} (ONLY this update)
    
    The LLM prompt is carefully crafted to get a number between 0-100.
    """
    print("🤖 [Grammar Analysis Node] Starting grammar analysis...")
    
    # Create focused prompt for grammar analysis
    prompt = f"""Analyze the grammar of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider these grammar aspects:
- Spelling errors and typos
- Proper punctuation usage
- Sentence structure and syntax
- Subject-verb agreement
- Tense consistency

Important: Return ONLY a number between 0-100, no other text.
Example responses: "85" or "92" or "78"
    
Your score (0-100):"""
    
    # Call LLM with the prompt
    response = llm.invoke(prompt).content.strip()
    
    # Extract numerical score from LLM response
    # Filter only digits from response and take first 3 characters
    score = int(''.join(filter(str.isdigit, response))[:3])
    
    # Ensure score is within 0-100 range
    score = min(score, 100)
    
    # Print confirmation
    print(f"    ✅ Grammar Analysis Complete: {score}/100")
    
    """
    CRITICAL: Return ONLY the state field we're updating.
    
    Why? Because in parallel execution:
    - This node runs at the same time as other analysis nodes
    - Each node should only modify its own part of the state
    - LangGraph will merge {'grammar_score': score} with updates from other nodes
    """
    return {'grammar_score': score}


def sentiment_node(state: ParallelState) -> Dict:
    """
    NODE 2: Sentiment Analysis Specialist
    
    This node analyzes the emotional tone and sentiment of the essay.
    It ONLY modifies the 'sentiment_score' field in the state.
    
    Workflow:
    1. Extract essay from state
    2. Create prompt asking LLM to analyze sentiment
    3. Send prompt to LLM and get response
    4. Extract numerical score from response
    5. Return {'sentiment_score': score} (ONLY this update)
    """
    print("🤖 [Sentiment Analysis Node] Starting sentiment analysis...")
    
    # Create focused prompt for sentiment analysis
    prompt = f"""Analyze the sentiment and emotional tone of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider these sentiment aspects:
- Overall emotional tone (positive/negative/neutral)
- Level of enthusiasm or concern
- Persuasive power and engagement
- Emotional impact on reader
- Consistency of tone throughout

Important: Return ONLY a number between 0-100, no other text.
Example responses: "85" or "92" or "78"
    
Your score (0-100):"""
    
    # Call LLM with the prompt
    response = llm.invoke(prompt).content.strip()
    
    # Extract numerical score from LLM response
    score = int(''.join(filter(str.isdigit, response))[:3])
    
    # Ensure score is within 0-100 range
    score = min(score, 100)
    
    # Print confirmation
    print(f"✅ Sentiment Analysis Complete: {score}/100")
    
    """
    CRITICAL: Return ONLY the state field we're updating.
    
    This node doesn't know or care about grammar_score or clarity_score.
    It only focuses on its job: analyzing sentiment.
    """
    return {'sentiment_score': score}


def clarity_node(state: ParallelState) -> Dict:
    """
    NODE 3: Clarity Analysis Specialist
    
    This node analyzes how clear and understandable the essay is.
    It ONLY modifies the 'clarity_score' field in the state.
    
    Workflow:
    1. Extract essay from state
    2. Create prompt asking LLM to analyze clarity
    3. Send prompt to LLM and get response
    4. Extract numerical score from response
    5. Return {'clarity_score': score} (ONLY this update)
    """
    print("🤖 [Clarity Analysis Node] Starting clarity analysis...")
    
    # Create focused prompt for clarity analysis
    prompt = f"""Analyze the clarity and understandability of this essay and give a score out of 100:
    
Essay: "{state['essay']}"

Consider these clarity aspects:
- How easy it is to understand the main points
- Logical flow and organization of ideas
- Clear expression and avoidance of ambiguity
- Effectiveness of explanations
- Overall readability and comprehension

Important: Return ONLY a number between 0-100, no other text.
Example responses: "85" or "92" or "78"
    
Your score (0-100):"""
    
    # Call LLM with the prompt
    response = llm.invoke(prompt).content.strip()
    
    # Extract numerical score from LLM response
    score = int(''.join(filter(str.isdigit, response))[:3])
    
    # Ensure score is within 0-100 range
    score = min(score, 100)
    
    # Print confirmation
    print(f"    ✅ Clarity Analysis Complete: {score}/100")
    
    """
    CRITICAL: Return ONLY the state field we're updating.
    
    This pattern ensures parallel nodes don't interfere with each other.
    Each node is like a specialist doing one job very well.
    """
    return {'clarity_score': score}

# =================================================================
# SECTION 5: FINALIZER NODE FUNCTION
# =================================================================

def finalizer_node(state: ParallelState) -> Dict:
    """
    NODE 4: Results Combiner (Finalizer)
    
    This node runs AFTER all parallel analysis nodes complete.
    It receives the complete state with all three scores.
    
    Workflow:
    1. Calculate average score from all three analyses
    2. Create comprehensive prompt asking LLM to generate a report
    3. Send prompt to LLM and get final report
    4. Return {'final_result': report} (ONLY this update)
    
    This node demonstrates how to use results from multiple
    parallel nodes to create a comprehensive output.
    """
    print("\n📊 [Finalizer Node] Starting final report generation...")
    print("   (This runs after all parallel analyses complete)")
    
    # Calculate average score from all three analyses
    grammar = state['grammar_score']
    sentiment = state['sentiment_score']
    clarity = state['clarity_score']
    avg_score = (grammar + sentiment + clarity) / 3
    
    print(f"   📈 Received scores: Grammar={grammar}, Sentiment={sentiment}, Clarity={clarity}")
    print(f"   🧮 Average score: {avg_score:.1f}/100")
    
    # Create comprehensive prompt for final report
    prompt = f"""You are an expert writing coach. Create a detailed analysis report.

ESSAY TO ANALYZE:
"{state['essay']}"

ANALYSIS RESULTS:
1. GRAMMAR: {grammar}/100
   - Focus: Spelling, punctuation, sentence structure
   
2. SENTIMENT: {sentiment}/100
   - Focus: Emotional tone, engagement, persuasive power
   
3. CLARITY: {clarity}/100
   - Focus: Understandability, logical flow, clear expression

OVERALL AVERAGE: {avg_score:.1f}/100

INSTRUCTIONS:
Create a comprehensive report with these sections:

1. EXECUTIVE SUMMARY
   - Brief overview of the essay
   - Overall assessment based on all scores

2. DETAILED ANALYSIS
   - Grammar: Strengths and areas for improvement
   - Sentiment: Emotional effectiveness
   - Clarity: Readability and organization

3. KEY STRENGTHS
   - What the writer does well (based on high scores)

4. AREAS FOR IMPROVEMENT
   - Specific, actionable suggestions (based on lower scores)

5. RECOMMENDATIONS
   - 3-5 specific actions to improve writing

Tone: Professional, constructive, encouraging
Length: Approximately 200-300 words

YOUR REPORT:"""
    
    # Call LLM to generate comprehensive report
    print("   🤖 Generating comprehensive report with LLM...")
    final_result = llm.invoke(prompt).content
    
    print("   ✅ Final report generated successfully!")
    
    """
    CRITICAL: Return ONLY the state field we're updating.
    
    Even though this node reads multiple fields, it only writes
    to the 'final_result' field. This maintains the pattern of
    each node being responsible for specific state updates.
    """
    return {'final_result': final_result}

# =================================================================
# SECTION 6: BUILDING THE GRAPH (Parallel Architecture)
# =================================================================

"""
This is where we define the PARALLEL workflow structure.

Key Concept: True Parallel Execution
- All three analysis nodes start from START simultaneously
- They can run in parallel (conceptually, if resources allow)
- All must complete before finalizer node runs
- LangGraph manages the execution order and state merging

Visual Representation:
        [START]
       /   |   \
      /    |    \
     ↓     ↓     ↓
  Grammar Sentiment Clarity   ← ALL START AT SAME TIME
     \     |     /
      \    |    /
       \   |   /
        ↓  ↓  ↓
      [Finalizer]
           ↓
          [END]
"""

print("\n" + "=" * 80)
print("🏗️  BUILDING PARALLEL WORKFLOW ARCHITECTURE")
print("=" * 80)
print("\nStep 1: Creating StateGraph with our ParallelState structure")
# Initialize the graph with our state type
graph = StateGraph(ParallelState)

print("\nStep 2: Adding nodes to the graph")
print("   Each node is a specialist function added with a unique name")
graph.add_node("grammar", grammar_node)
print("   ✅ Added: Grammar Analysis Node")
graph.add_node("sentiment", sentiment_node)
print("   ✅ Added: Sentiment Analysis Node")
graph.add_node("clarity", clarity_node)
print("   ✅ Added: Clarity Analysis Node")
graph.add_node("finalizer", finalizer_node)
print("   ✅ Added: Finalizer Node (combines all results)")

print("\nStep 3: Creating TRUE PARALLEL connections")
print("   This is the key to parallel execution:")

# CRITICAL: All three analysis nodes start from START
# This means they can run in parallel
print("\n   🔗 Connecting START → [Grammar, Sentiment, Clarity]")
print("   (All three nodes start simultaneously from START)")
graph.add_edge(START, "grammar")
graph.add_edge(START, "sentiment")
graph.add_edge(START, "clarity")

# All analysis nodes must complete before finalizer runs
print("\n   🔗 Connecting [Grammar, Sentiment, Clarity] → Finalizer")
print("   (Finalizer waits for all three analyses to complete)")
graph.add_edge("grammar", "finalizer")
graph.add_edge("sentiment", "finalizer")
graph.add_edge("clarity", "finalizer")

# Finalizer completes the workflow
print("\n   🔗 Connecting Finalizer → END")
graph.add_edge("finalizer", END)

print("\n✅ PARALLEL GRAPH STRUCTURE COMPLETE!")
print("\n" + "=" * 80)
print("GRAPH ARCHITECTURE SUMMARY:")
print("=" * 80)
print("""
VISUAL WORKFLOW DIAGRAM:

        ┌─────────────┐
        │    START    │
        └─────┬───────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐
│Grammar ││Sentiment││ Clarity │
│Analysis││Analysis ││ Analysis│
└───┬────┘└───┬────┘└───┬────┘
    │         │         │
    └─────────┼─────────┘
              │
              ▼
        ┌─────────────┐
        │  Finalizer  │
        │   (Report   │
        │   Generator)│
        └──────┬──────┘
               │
               ▼
              END

KEY CHARACTERISTICS:
1. TRUE PARALLEL: Grammar, Sentiment, and Clarity nodes all start from START
2. SYNCHRONIZATION: All three must finish before Finalizer runs
3. STATE MERGING: LangGraph automatically merges updates from all nodes
4. SPECIALIZATION: Each node updates only its assigned state field
""")
print("=" * 80)

# Compile the graph into an executable workflow
print("\n🔧 COMPILING WORKFLOW...")
workflow = graph.compile()
print("✅ WORKFLOW COMPILED AND READY FOR EXECUTION!")

# =================================================================
# SECTION 7: DEMONSTRATION AND EXECUTION
# =================================================================

print("\n" + "=" * 80)
print("🚀 DEMONSTRATION: PARALLEL ESSAY ANALYSIS")
print("=" * 80)

# Define a sample essay for analysis
sample_essay = """Artificial intelligence is revolutionizing education by providing personalized learning experiences for students and valuable tools for teachers. AI-powered systems can adapt to individual learning styles, identify knowledge gaps, and provide targeted exercises. This technology also helps educators by automating administrative tasks, generating lesson plans, and offering insights into student performance. While AI presents exciting opportunities, it's important to use it ethically and ensure it complements rather than replaces human instruction. The future of education lies in the synergy between human teachers and intelligent systems."""

print(f"\n📝 SAMPLE ESSAY FOR ANALYSIS:")
print("-" * 80)
print(f'"{sample_essay[:150]}..."')
print(f"\nEssay Statistics:")
print(f"  • Characters: {len(sample_essay)}")
print(f"  • Words: {len(sample_essay.split())}")
print(f"  • Sentences: {sample_essay.count('.') + sample_essay.count('!') + sample_essay.count('?')}")
print("-" * 80)

# Create initial state with the essay
print("\n📦 PREPARING INITIAL STATE:")
print("   (This is the data container that flows through the workflow)")
initial_state = {
    'essay': sample_essay,
    'grammar_score': 0,      # Will be updated by grammar_node
    'sentiment_score': 0,    # Will be updated by sentiment_node
    'clarity_score': 0,      # Will be updated by clarity_node
    'final_result': ''       # Will be updated by finalizer_node
}

print(f"\nInitial State Contents:")
for key, value in initial_state.items():
    if key == 'essay':
        print(f"  • {key}: '{value[:50]}...'")
    else:
        print(f"  • {key}: {value}")

print("\n" + "=" * 80)
print("⚡ EXECUTING PARALLEL WORKFLOW")
print("=" * 80)
print("\nIMPORTANT: Watch the execution order in the output below.")
print("All three analysis nodes should start around the same time.")
print("Finalizer node waits for all three to complete.")

print("\n🔍 EXECUTION TRACE:")
print("-" * 80)

# Execute the workflow with our initial state
print("\nStarting workflow.invoke(initial_state)...")
print("=" * 80)

# The actual execution happens here
final_state = workflow.invoke(initial_state)

print("=" * 80)
print("✅ WORKFLOW EXECUTION COMPLETE!")
print("=" * 80)

# =================================================================
# SECTION 8: RESULTS PRESENTATION
# =================================================================

print("\n" + "=" * 80)
print("📊 ANALYSIS RESULTS")
print("=" * 80)

print("\n🎯 INDIVIDUAL ANALYSIS SCORES:")
print("-" * 40)

# Display each score with emoji indicators
scores = [
    ("Grammar", final_state['grammar_score']),
    ("Sentiment", final_state['sentiment_score']),
    ("Clarity", final_state['clarity_score'])
]

for name, score in scores:
    # Add visual indicators based on score
    if score >= 90:
        indicator = "🌟 EXCELLENT"
    elif score >= 80:
        indicator = "✅ VERY GOOD"
    elif score >= 70:
        indicator = "⚠️  GOOD"
    elif score >= 60:
        indicator = "📝 FAIR"
    else:
        indicator = "🔧 NEEDS WORK"
    
    print(f"  {name:12} {score:3}/100 {indicator}")

# Calculate and display average
average_score = sum(s[1] for s in scores) / len(scores)
print("-" * 40)
print(f"  {'AVERAGE':12} {average_score:3.1f}/100", end=" ")

# Add overall assessment
if average_score >= 90:
    print("🏆 OUTSTANDING")
elif average_score >= 80:
    print("🎯 EXCELLENT")
elif average_score >= 70:
    print("👍 GOOD")
elif average_score >= 60:
    print("📋 SATISFACTORY")
else:
    print("💡 NEEDS IMPROVEMENT")

print("\n" + "=" * 80)
print("📄 COMPREHENSIVE ANALYSIS REPORT")
print("=" * 80)

# Display the final report generated by the LLM
print("\n" + final_state['final_result'])
print("=" * 80)

# =================================================================
# SECTION 9: WORKFLOW ANALYSIS AND INSIGHTS
# =================================================================

print("\n" + "=" * 80)
print("💡 WORKSHOP LEARNING POINTS")
print("=" * 80)

print("""
KEY CONCEPTS DEMONSTRATED:

1. TRUE PARALLEL EXECUTION
   - Multiple nodes starting from START simultaneously
   - Each node runs independently
   - LangGraph manages execution and state merging

2. PROPER STATE MANAGEMENT
   - Each node returns ONLY what it updates
   - No conflicts between parallel nodes
   - Clear separation of responsibilities

3. LLM INTEGRATION PATTERNS
   - Structured prompts for consistent outputs
   - Score extraction from LLM responses
   - Chaining multiple LLM calls effectively

4. WORKFLOW ORCHESTRATION
   - Parallel nodes → Synchronization point → Final node
   - Clear data flow through state
   - Modular, maintainable architecture

5. REAL-WORLD APPLICATION
   - Automated essay scoring and feedback
   - Parallel analysis for efficiency
   - Comprehensive reporting from multiple inputs

EXTENSION IDEAS FOR WORKSHOP:
1. Add more analysis criteria (vocabulary, structure, originality)
2. Implement weighted scoring (grammar 40%, sentiment 30%, clarity 30%)
3. Add confidence scores for each analysis
4. Create different report formats (executive, detailed, student-friendly)
5. Implement batch processing for multiple essays
6. Add visualization of scores (charts, graphs)
7. Integrate with learning management systems
""")

print("=" * 80)
print("🎉 PARALLEL LLM ANALYSIS WORKFLOW DEMONSTRATION COMPLETE!")
print("=" * 80)