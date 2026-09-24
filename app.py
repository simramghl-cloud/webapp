# Required packages (add to your existing venv):
# pip install streamlit

import os
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool


st.set_page_config(page_title="My All-in-One Agent", page_icon="🤖")
st.title("🤖 My All-in-One Agent")
st.caption("Ask about your documents (PDF, Word, Excel, web page) or anything else — it will search Wikipedia if needed.")


# ─────────────────────────────────────────────────────────────────────────────
# Everything below runs ONCE and is cached, so re-typing a question doesn't
# reload the documents/embeddings every single time (that would be very slow).
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading your documents and setting up the agent...")
def build_agent():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "docs")

    def load_excel(file_path):
        df = pd.read_excel(file_path)
        docs = []
        for _, row in df.iterrows():
            text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(Document(page_content=text, metadata={"source": file_path}))
        return docs

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
    }

    all_docs = []
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            file_path = os.path.join(folder_path, filename)
            if ext == ".xlsx":
                all_docs.extend(load_excel(file_path))
                continue
            loader_cls = LOADER_MAP.get(ext)
            if loader_cls is None:
                continue
            all_docs.extend(loader_cls(file_path).load())

    WEB_URL = "https://www.who.int/news-room/fact-sheets/detail/adolescent-mental-health"
    try:
        all_docs.extend(WebBaseLoader(WEB_URL).load())
    except Exception:
        pass

    retriever = None
    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def search_my_documents(query: str) -> str:
        """Search the user's own uploaded documents (PDF, Word, Excel, and the
        loaded web page) for information relevant to the query. Use this FIRST
        for anything that might be in the user's files."""
        if retriever is None:
            return "No documents are loaded."
        results = retriever.invoke(query)
        if not results:
            return "No relevant information found in the documents."
        return "\n\n".join(doc.page_content for doc in results)

    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search_my_documents, wikipedia_tool]
    tools_by_name = {t.name: t for t in tools}

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key="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",
    )
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools_by_name


llm_with_tools, tools_by_name = build_agent()


def ask(query: str) -> str:
    messages = [HumanMessage(content=query)]
    for _ in range(5):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            return ai_msg.content
        for tool_call in ai_msg.tool_calls:
            chosen_tool = tools_by_name.get(tool_call["name"])
            tool_result = chosen_tool.invoke(tool_call["args"]) if chosen_tool else "Unknown tool"
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
    return "Sorry, I couldn't settle on a final answer in time."


# ─────────────────────────────────────────────────────────────────────────────
# Chat UI — keeps the conversation on screen using Streamlit's session_state
# ─────────────────────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(user_input)
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))
