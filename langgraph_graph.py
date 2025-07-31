import os
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

# State Schema
class ResearchState(TypedDict, total=False):
    topic: str
    subquestions: List[str]
    findings: List[str]
    summary: str
    report: str
    citations: List[str]
    pdf_text: str
    attempted_replanning: bool
    confidence_scores: List[float]

# Planner Node
def planner_node(state: Dict) -> Dict:
    topic = state["topic"]
    prompt = f" You are an intelligent research agent who have access to tavily search engine.Get the latest info about the research topic '{topic}' into 3-5 detailed sub-questions for research ang get accurate answers for them. Return the sub-questions in a list format, each starting with a hyphen."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    questions = response.choices[0].message.content.split("\n")
    subquestions = [q.strip("- ").strip() for q in questions if q.strip()]
    return {"topic": topic, "subquestions": subquestions}

# Tavily Web Search using API
def tavily_search(query: str) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "include_answers": True,
        "include_raw_content": False,
        "include_images": False,
        "max_results": 2
    }
    response = requests.post(url, headers=headers, json=body)
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

# Gatherer Node with Replanning
def gatherer_node(state: Dict) -> Dict:
    findings = []
    citations = []
    empty_results = 0

    def fetch(q):
        return q, tavily_search(q)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch, state["subquestions"])

    for question, search_results in results:
        if not search_results:
            empty_results += 1
            continue
        for item in search_results:
            findings.append(f"Q: {question}\nA: {item['content']}")
            citations.append(item["url"])

    if empty_results >= len(state["subquestions"]) // 2 and not state.get("attempted_replanning", False):
        return {
            "replan_needed": True,
            "topic": state["topic"],
            "attempted_replanning": True
        }

    return {
        "topic": state["topic"],
        "subquestions": state["subquestions"],
        "findings": findings,
        "citations": citations,
        "pdf_text": state.get("pdf_text", "")[:6000],
        "attempted_replanning": state.get("attempted_replanning", False)
    }

# Synthesizer Node (summarize + score)
def synthesizer_node(state: Dict) -> Dict:
    topic = state["topic"]
    partial_summaries = []
    confidence_scores = []

    pdf_context = state.get("pdf_text", "")
    if pdf_context:
        partial_summaries.append(f"📎 Context from PDF:\n{pdf_context[:3000]}")

    for finding in state["findings"]:
        try:
            prompt = (
                f"Summarize and rate confidence (0-100) in factual accuracy. "
                f"Return as:\nSummary: <text>\nConfidence: <number>\n\nInput:\n{finding}"
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            summary = content.split("Summary:")[1].split("Confidence:")[0].strip()
            score = float(content.split("Confidence:")[1].strip())

            partial_summaries.append(summary)
            confidence_scores.append(score)
        except Exception as e:
            partial_summaries.append(f"[Error summarizing: {e}]")
            confidence_scores.append(50.0)

    combined = "\n\n".join(partial_summaries)[:6000]
    final_prompt = f"Based on the following summaries and PDF context, provide a structured research report on '{topic}':\n\n{combined}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": final_prompt}]
    )
    full_summary = response.choices[0].message.content.strip()

    return {
        "summary": full_summary,
        "report": full_summary,
        "citations": state.get("citations", []),
        "confidence_scores": confidence_scores
    }

# Orchestrator with Progress Feedback
def run_research_agent(topic: str, pdf_text: str = "") -> Dict:
    builder = StateGraph(ResearchState)
    builder.add_node("Planner", planner_node)
    builder.add_node("Gatherer", gatherer_node)
    builder.add_node("Synthesizer", synthesizer_node)

    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Gatherer")

    def gather_decision(state: Dict) -> str:
        return "Planner" if state.get("replan_needed") else "Synthesizer"

    builder.add_conditional_edges("Gatherer", gather_decision)
    builder.add_edge("Synthesizer", END)

    graph = builder.compile()
    state = {"topic": topic, "pdf_text": pdf_text[:6000]}

    # Step-by-step execution with UI updates
    st.info("Planning research sub-questions...")
    planner_output = planner_node(state)
    st.success("✅ Planning complete.")
    # st.write("**Sub-questions:**", planner_output["subquestions"])

    st.info("Gathering information via Tavily search...")
    gather_output = gatherer_node(planner_output)
    if gather_output.get("replan_needed"):
        st.warning("⚠️ Not enough info. Replanning sub-questions...")
        planner_output = planner_node(gather_output)
        gather_output = gatherer_node(planner_output)
    st.success("✅ Info gathering complete.")
    st.write(f"**{len(gather_output['findings'])} findings collected.**")

    st.info("Summarizing findings and scoring confidence...")
    synth_output = synthesizer_node(gather_output)
    st.success("✅ Summary and report generation complete.")

    return {
        "topic": topic,
        "subquestions": planner_output["subquestions"],
        "findings": gather_output["findings"],
        "citations": synth_output["citations"],
        "summary": synth_output["summary"],
        "report": synth_output["report"],
        "confidence_scores": synth_output["confidence_scores"]
    }
