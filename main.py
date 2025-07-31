import streamlit as st
import fitz
from langgraph_graph import run_research_agent

st.set_page_config(page_title="Research Agent", layout="wide")
st.title("Generative AI Research Assistant")

# Session state to persist last report
if "last_report" not in st.session_state:
    st.session_state.last_report = None

#  Input: Topic + Optional PDF
topic = st.text_input("Enter your research topic or question", placeholder="e.g., Impact of AI on healthcare")
pdf = st.file_uploader(" Optionally upload a PDF for additional context", type=["pdf"])

#  Extract PDF Text
pdf_text = ""
if pdf is not None:
    with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

#  Run Research Agent
if st.button("Start Research") and topic:
    try:
        with st.spinner("🔄 Running research ..."):
            report = run_research_agent(topic, pdf_text=pdf_text)
            st.session_state.last_report = report
        st.success("✅ Research completed!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

#  Display Results
if st.session_state.last_report:
    report = st.session_state.last_report

    st.info("Research process completed: Planner → Gatherer → Synthesizer → Report Generated")

    st.subheader(" Executive Summary")
    st.markdown(report.get("summary", "No summary generated."))

    st.subheader(" Detailed Report")
    st.markdown(report.get("report", "No report available."))

    #  Citations
    st.subheader(" Citations")
    citations = report.get("citations", [])
    if citations:
        if "show_all_citations" not in st.session_state:
            st.session_state.show_all_citations = False

        show_all = st.checkbox("Show all citations", value=st.session_state.show_all_citations)
        st.session_state.show_all_citations = show_all

        display_citations = citations if show_all else citations[:5]
        for i, src in enumerate(display_citations, 1):
            st.markdown(f"**{i}.** [{src}]({src})")

        if not show_all and len(citations) > 5:
            st.caption(f"Showing 5 of {len(citations)} citations")
    else:
        st.info("No citations found.")

    #  Download Report
    st.download_button(
        " Download Full Report",
        data=report.get("report", ""),
        file_name="research_report.txt"
    )

else:
    st.info("Enter a topic to start the research.")
