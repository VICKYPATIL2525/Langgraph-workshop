import os
import json
import sqlite3
from typing import TypedDict, Literal, Any

from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

# ============================================================
# ENV & LLM
# ============================================================
load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=200,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# ============================================================
# DATABASES
# ============================================================
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["langgraph_db"]
students_col = mongo_db["students"]

SQLITE_DB = "institute.db"

def sqlite_connect():
    return sqlite3.connect(SQLITE_DB)

# ============================================================
# GRAPH STATE
# ============================================================
class GraphState(TypedDict):
    question: str
    intent: dict
    db_result: Any
    final_answer: str

# ============================================================
# PLANNER NODE (INTENT ONLY)
# ============================================================
def planner_node(state: GraphState):
    prompt = f"""
You MUST respond with ONLY valid JSON.
NO text. NO explanation. NO markdown.

MongoDB students document example:
{{
  "student_id": 101,
  "name": "Rohit Kumar",
  "hostel_room": "A-101",
  "fees_status": "pending"
}}

SQLite tables:
- exam_results(student_id, subject_id, marks)
- subjects(subject_id, subject_name)
- timetable(course_id, day, time, subject)

Allowed intents ONLY:

Hostel room:
{{ "source": "mongo", "action": "get_hostel", "student_name": "FULL NAME OR PARTIAL NAME" }}

Pending fees:
{{ "source": "mongo", "action": "pending_fees" }}

Top scorers:
{{ "source": "sqlite", "action": "top_scorers", "limit": 3 }}

Timetable:
{{ "source": "sqlite", "action": "timetable" }}

User question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    # 🔍 DEBUG: raw LLM output
    print("\n🧠 RAW LLM PLANNER OUTPUT:")
    print(response.content)

    try:
        state["intent"] = json.loads(response.content)
    except Exception as e:
        state["intent"] = {
            "error": "INVALID_JSON_FROM_LLM",
            "raw_output": response.content
        }

    # 🔍 DEBUG: parsed intent
    print("\n📌 PARSED INTENT:")
    print(state["intent"])

    return state


# ============================================================
# EXECUTOR NODE (SAFE QUERIES)
# ============================================================
def executor_node(state: GraphState):
    intent = state["intent"]

    print("\n⚙️ EXECUTING INTENT:")
    print(intent)

    # =========================
    # MongoDB execution
    # =========================
    if intent.get("source") == "mongo":

        if intent.get("action") == "get_hostel":
            name = intent.get("student_name", "")

            state["db_result"] = students_col.find_one(
                {"name": {"$regex": name, "$options": "i"}},
                {"_id": 0, "name": 1, "hostel_room": 1}
            )

        elif intent.get("action") == "pending_fees":
            state["db_result"] = list(
                students_col.find(
                    {"fees_status": "pending"},
                    {"_id": 0, "student_id": 1, "name": 1}
                )
            )

        else:
            state["db_result"] = "UNSUPPORTED_MONGO_ACTION"

    # =========================
    # SQLite execution
    # =========================
    elif intent.get("source") == "sqlite":
        conn = sqlite_connect()
        cursor = conn.cursor()

        if intent.get("action") == "top_scorers":
            limit = intent.get("limit", 3)
            cursor.execute("""
            SELECT exam_results.student_id, subjects.subject_name, exam_results.marks
            FROM exam_results
            JOIN subjects ON exam_results.subject_id = subjects.subject_id
            ORDER BY exam_results.marks DESC
            LIMIT ?
            """, (limit,))

            state["db_result"] = cursor.fetchall()

        elif intent.get("action") == "timetable":
            cursor.execute("""
            SELECT day, time, subject
            FROM timetable
            WHERE course_id = 2
            """)
            state["db_result"] = cursor.fetchall()

        else:
            state["db_result"] = "UNSUPPORTED_SQLITE_ACTION"

        conn.close()

    else:
        state["db_result"] = "INVALID_SOURCE"

    # 🔍 DEBUG: DB result
    print("\n📦 DB RESULT:")
    print(state["db_result"])

    # 🚨 Fail loud if no data
    if not state["db_result"]:
        state["db_result"] = "NO_DATA_FOUND"

    return state


# ============================================================
# FINALIZER NODE
# ============================================================
def finalizer_node(state: GraphState):
    prompt = f"""
User Question:
{state["question"]}

Retrieved Data:
{state["db_result"]}

Give a clear, factual answer.
If data is empty, say so politely.
"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state

# ============================================================
# BUILD GRAPH
# ============================================================
builder = StateGraph(GraphState)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "finalizer")
builder.add_edge("finalizer", END)

graph = builder.compile()

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("🔹 LangGraph Multi-DB (LLM Intent Driven)")
    print("Type 'exit' to quit")

    while True:
        q = input("\nAsk: ").strip()

        # 🔒 FIX 1: prevent empty input
        if not q:
            print("⚠️ Please enter a valid question.")
            continue

        # 🔒 FIX 2: clean exit
        if q.lower() == "exit":
            print("👋 Exiting...")
            break

        result = graph.invoke({"question": q})

        print("\n✅ Answer:")
        print(result["final_answer"])
