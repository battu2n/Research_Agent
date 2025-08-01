import os
import re
import requests
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ------------------
# 📦 State Schema
# ------------------
class ResearchState(TypedDict, total=False):
    topic: str
    subquestions: List[str]
    findings: List[str]
    summary: str
    report: str
    citations: List[str]
    pdf_text: str
    confidence_scores: List[float]

# ------------------
# 🧠 Question Type Detector
# ------------------
def detect_question_type(topic: str) -> str:
    prompt = (
        "Classify the following topic into one category:\n"
        "- summary\n- comparison\n- pros_cons\n- timeline\n\n"
        f"Topic: {topic}\nRespond with only one category."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

# ------------------
# 🧠 Planner Node
# ------------------
def planner_node(state: Dict) -> Dict:
    topic = state["topic"]
    prompt = (
        f"You are a research agent with access to Tavily's real-time web search. "
        f"Generate 3-5 sub-questions to get the **latest information** on '{topic}'. "
        f"Focus on recent events and trends. Return as a bullet list starting with '-'."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    questions = response.choices[0].message.content.split("\n")
    subquestions = [q.strip("- ").strip() for q in questions if q.strip()]
    return {"topic": topic, "subquestions": subquestions[:5]}

# ------------------
# 🌐 Tavily Web Search
# ------------------
def tavily_search(query: str) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "include_answers": True,
        "include_raw_content": False,
        "include_images": False,
        "max_results": 3,
        "search_depth": "advanced",
        "timeframe": "month"
    }
    response = requests.post(url, headers=headers, json=body, timeout=15)
    data = response.json()
    results = []
    for item in data.get("results", []):
        results.append({
            "question": query,
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")
        })
    return results

# ------------------
# 🔍 Gatherer Node
# ------------------
def gatherer_node(state: Dict) -> Dict:
    findings = []
    citations = []

    def fetch(q):
        return q, tavily_search(q)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch, state["subquestions"])

    for question, search_results in results:
        for item in search_results:
            findings.append(f"Q: {question}\nA: {item['content']}")
            citations.append(item["url"])

    return {
        "topic": state["topic"],
        "subquestions": state["subquestions"],
        "findings": findings,
        "citations": citations,
        "pdf_text": state.get("pdf_text", "")[:6000]
    }

# ------------------
# 📊 Synthesizer Node
# ------------------
def synthesizer_node(state: Dict) -> Dict:
    topic = state["topic"]
    partial_summaries = []

    for finding in state["findings"]:
        try:
            prompt = (
                "Summarize this finding in 1-3 sentences and rate confidence (0–100).\n\n"
                "Respond in this exact format:\n"
                "<summary text> [XX%]\n\n"
                "Do NOT add labels or extra text.\n\n"
                f"Finding:\n{finding}"
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()

            # ✅ Ensure % sign is present
            if "%" not in content:
                content = content.rstrip("0123456789") + " [50%]"

            partial_summaries.append(content)

        except Exception as e:
            partial_summaries.append(f"[Error summarizing: {e}] [50%]")

    combined = "\n\n".join(partial_summaries)[:6000]

    q_type = detect_question_type(topic)

    if q_type == "comparison":
        final_prompt = f"Compare and present as a table:\n\n{combined}"
    elif q_type == "pros_cons":
        final_prompt = f"Create pros and cons list:\n\n{combined}"
    elif q_type == "timeline":
        final_prompt = f"Create a chronological timeline:\n\n{combined}"
    else:
        final_prompt = f"Create a structured research report:\n\n{combined}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": final_prompt}]
    )
    full_summary = response.choices[0].message.content.strip()

    return {
        "summary": full_summary,
        "report": full_summary,
        "citations": state.get("citations", [])
    }

# ------------------
# 🚀 Orchestrator with Streaming
# ------------------
def run_research_agent(topic: str, pdf_text: str = "") -> Dict:
    st.info("🧠 Planning research sub-questions...")
    planner_output = planner_node({"topic": topic})
    st.success(f"✅ Planning complete. Generated {len(planner_output['subquestions'])} sub-questions.")

    st.info("🌐 Gathering information via Tavily search...")
    gather_output = gatherer_node(planner_output)
    st.success(f"✅ Information gathering complete. Collected {len(gather_output['findings'])} findings.")

    st.info("📝 Summarizing findings and generating report...")
    synth_output = synthesizer_node(gather_output)
    st.success("✅ Summary and report generation complete.")

    return {
        "topic": topic,
        "subquestions": planner_output["subquestions"],
        "findings": gather_output["findings"],
        "citations": synth_output["citations"],
        "summary": synth_output["summary"],
        "report": synth_output["report"],
    }
