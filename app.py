"""InsightForge: two-file capstone demo for AI-powered business intelligence."""
import os
import requests
import pandas as pd
import streamlit as st
import altair as alt
from data_engine import load_dataset, dataframe_profile, make_retrieval_chunks, retrieve_context, quick_answer

st.set_page_config(page_title="InsightForge BI Assistant", page_icon="📊", layout="wide")
st.title("📊 InsightForge")
st.caption("AI-Powered Business Intelligence Assistant | Data analysis • RAG • Insights • Visualizations")

DEFAULT_PATH = os.getenv("INSIGHTFORGE_DATA_PATH", "data")
with st.sidebar:
    st.header("Data source")
    source_path = st.text_input("Folder or ZIP path", value=DEFAULT_PATH,
        help=r"Example: C:\Users\nader\Downloads\Datasets_New.zip")
    st.caption("Supported: CSV, Excel, JSON, PDF, TXT, Markdown")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    model_name = st.text_input("Ollama model", value="qwen3:4b")
    use_llm = st.checkbox("Use local Ollama LLM", value=True)
    if st.button("Load / refresh data", type="primary", use_container_width=True):
        st.session_state.pop("loaded_source", None)
        st.session_state.pop("tables", None)
        st.session_state.pop("documents", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("load_messages", None)
        st.rerun()

if st.session_state.get("loaded_source") != source_path:
    tables, documents, messages = load_dataset(source_path)
    st.session_state["loaded_source"] = source_path
    st.session_state["tables"] = tables
    st.session_state["documents"] = documents
    st.session_state["chunks"] = make_retrieval_chunks(tables, documents)
    st.session_state["load_messages"] = messages
    st.session_state["chat"] = []

tables = st.session_state.get("tables", [])
documents = st.session_state.get("documents", [])
chunks = st.session_state.get("chunks", [])
for message in st.session_state.get("load_messages", []):
    st.warning(message)

if not tables and not documents:
    st.info("Enter the path to your dataset folder or ZIP in the sidebar. The path must exist on the computer running Streamlit.")
    st.code(r"streamlit run app.py")
    st.stop()

total_rows = sum(len(item["data"]) for item in tables)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Datasets", len(tables))
c2.metric("Total rows", f"{total_rows:,}")
c3.metric("Reference documents", len(documents))
c4.metric("Retrieved chunks", len(chunks))

tabs = st.tabs(["Dashboard", "Explore data", "AI Chat / RAG", "About"])
with tabs[0]:
    st.subheader("Business overview")
    if tables:
        selected = st.selectbox("Choose dataset", range(len(tables)), format_func=lambda i: tables[i]["name"])
        df = tables[selected]["data"]
        profile = dataframe_profile(df)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Dataset preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with right:
            st.markdown("**Numeric summary**")
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                st.dataframe(numeric.describe().T.round(3), use_container_width=True)
                chart_col = st.selectbox("Column to visualize", list(numeric.columns))
                plot_df = df[[chart_col]].dropna().reset_index(names="Row")
                plot_df = plot_df.head(3000)
                chart = alt.Chart(plot_df).mark_line(point=False).encode(
                    x=alt.X("Row:Q", title="Row index"),
                    y=alt.Y(f"{chart_col}:Q", title=str(chart_col)),
                    tooltip=["Row:Q", alt.Tooltip(f"{chart_col}:Q")]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No numeric columns detected in this dataset.")
        st.markdown("**Dataset profile**")
        st.json(profile)
    else:
        st.write("No tabular dataset was found. You can still ask questions about loaded documents in AI Chat / RAG.")

with tabs[1]:
    st.subheader("Explore data")
    if tables:
        table_index = st.selectbox("Dataset", range(len(tables)), format_func=lambda i: tables[i]["name"], key="explore_dataset")
        df = tables[table_index]["data"]
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        columns = st.multiselect("Columns to display", list(df.columns), default=list(df.columns[:min(8, len(df.columns))]))
        st.dataframe(df[columns].head(500), use_container_width=True)
        if len(df.select_dtypes(include="number").columns) >= 2:
            numeric_cols = list(df.select_dtypes(include="number").columns)
            x_col = st.selectbox("X-axis", numeric_cols, key="xaxis")
            y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="yaxis")
            st.altair_chart(alt.Chart(df[[x_col, y_col]].dropna().head(3000)).mark_circle(size=55).encode(
                x=x_col, y=y_col, tooltip=[x_col, y_col]
            ).properties(height=350), use_container_width=True)
    else:
        st.info("No table found. Add a CSV, Excel, or JSON file to your source.")

def ask_ollama(question, context):
    prompt = f"""You are InsightForge, a careful business intelligence analyst.
Answer the user's question using only the supplied retrieved context.
Do not invent figures. If the data does not contain enough evidence, state what is missing.
Give a concise answer with: Finding, Evidence, Recommendation (when justified).
Retrieved context:
{context}

User question: {question}"""
    response = requests.post(
        ollama_url.rstrip("/") + "/api/chat",
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()

with tabs[2]:
    st.subheader("Ask your business data")
    st.caption("Questions are matched to relevant data/document context before the answer is generated.")
    for msg in st.session_state.get("chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Example: Summarize the main trends and identify useful KPIs")
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        retrieved = retrieve_context(question, chunks, top_k=4)
        context = "\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in retrieved)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and preparing analysis..."):
                answer = None
                if use_llm:
                    try:
                        answer = ask_ollama(question, context)
                        if not answer:
                            answer = None
                    except Exception as exc:
                        st.warning(f"Ollama was not available; using the local analytics fallback. Details: {exc}")
                if answer is None:
                    answer = quick_answer(question, tables, documents, retrieved)
                st.markdown(answer)
                if retrieved:
                    with st.expander("Sources retrieved by RAG"):
                        for item in retrieved:
                            st.markdown(f"**{item['source']}**")
                            st.write(item["text"][:1200])
        st.session_state["chat"].append({"role": "assistant", "content": answer})

with tabs[3]:
    st.subheader("Project scope")
    st.markdown("""
- **Data preparation:** loads pre-prepared datasets rather than attempting extensive data cleaning.
- **Knowledge base:** creates searchable context from dataset profiles, sample rows, and PDF/text documents.
- **RAG:** retrieves relevant context before answering.
- **LLM:** optionally uses a local Ollama model; a statistics-based fallback remains available.
- **Memory:** keeps the conversation visible during the current Streamlit session.
- **Visualization:** provides previews, numeric summaries, and charts.
- **Evaluation note:** before submission, test with known questions and record whether answers match the source data. This compact two-file version does not implement QAEvalChain automatically.
""")
    st.markdown("**Important:** business analyses depend on the actual dataset columns. No sales, region, product, or demographic fields are assumed.")

st.sidebar.divider()
st.sidebar.caption("InsightForge • Capstone prototype")
