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


#Question Type Detector

def detect_question_type(topic: str) -> str:
    prompt = f"""
    You are a research assistant. Classify the following research topic into ONE category:
    - 'summary' (general research overview)
    - 'comparison' (comparing 2+ items, technologies, companies)
    - 'pros_cons' (analyze advantages and disadvantages)
    - 'timeline' (chronological events)

    Respond with ONLY the category name.

    Topic: {topic}
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()


# Planner Node
# 
def planner_node(state: Dict) -> Dict:
    topic = state["topic"]
    prompt = (
        f"You are an intelligent research agent with access to Tavily's real-time search engine. "
        f"Generate 3-5 detailed, non-overlapping sub-questions that will help comprehensively research '{topic}'. "
        f"Ensure they cover multiple perspectives and lead to accurate answers. "
        f"Return the sub-questions as a bullet list, each starting with a hyphen (-)."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    questions = response.choices[0].message.content.split("\n")
    subquestions = [q.strip("- ").strip() for q in questions if q.strip()]
    return {"topic": topic, "subquestions": subquestions[:5]}


# Tavily Web Search

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


# Gatherer Node

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


# Synthesizer Node 

def synthesizer_node(state: Dict) -> Dict:
    topic = state["topic"]
    partial_summaries = []

    pdf_context = state.get("pdf_text", "")
    if pdf_context:
        partial_summaries.append(f" Context from PDF:\n{pdf_context[:3000]}")

    for finding in state["findings"]:
        try:
            prompt = (
                f"You are a research assistant. Summarize the following finding briefly and provide a confidence score "
                f"(0-100) based on factual accuracy and reliability of the information.\n\n"
                f"Return in the format:\n<summary text> [XX%]\n\n"
                f"Finding:\n{finding}"
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            partial_summaries.append(content)
        except Exception as e:
            partial_summaries.append(f"[Error summarizing finding: {e}] [50%]")

    combined = "\n\n".join(partial_summaries)[:6000]

    # Dynamic final prompt based on question type
    q_type = detect_question_type(topic)

    if q_type == "comparison":
        final_prompt = (
            f"Compare the following findings on '{topic}' and present them in a well-structured table format with clear columns.\n\n{combined}"
        )
    elif q_type == "pros_cons":
        final_prompt = (
            f"Analyze the following findings on '{topic}' and present them as a Pros and Cons list with bullet points.\n\n{combined}"
        )
    elif q_type == "timeline":
        final_prompt = (
            f"Create a chronological timeline for '{topic}' with events, dates (if available), and explanations.\n\n{combined}"
        )
    else:
        final_prompt = (
            f"Provide a detailed and well-structured research report on '{topic}'. Use headings, subheadings, and bullet points if needed.\n\n{combined}"
        )

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


# Orchestrator

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

    st.info("Planning research sub-questions...")
    planner_output = planner_node(state)
    st.success("✅ Planning complete.")

    st.info("Gathering information via Tavily search...")
    gather_output = gatherer_node(planner_output)
    if gather_output.get("replan_needed"):
        st.warning("⚠️ Not enough info. Replanning...")
        planner_output = planner_node(gather_output)
        gather_output = gatherer_node(planner_output)
    st.success("✅ Info gathering complete.")
    st.write(f"**{len(gather_output['findings'])} findings collected.**")

    st.info("Synthesizing results and generating final report...")
    synth_output = synthesizer_node(gather_output)
    st.success("✅ Summary and report generation complete.")

    return {
        "topic": topic,
        "subquestions": planner_output["subquestions"],
        "findings": gather_output["findings"],
        "citations": synth_output["citations"],
        "summary": synth_output["summary"],
        "report": synth_output["report"]
    }
