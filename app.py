# -*- coding: utf-8 -*-
""" Agentic RAG Pipeline

Converted from standard (naive) RAG to Agentic RAG using LangGraph.

Additional requirements (install before running):
    pip install langgraph langchain-community tavily-python pydantic

Agentic RAG Graph Flow:
    User Question
        ↓
    [Router]  → vectorstore | web_search
        ↓
    [Retrieve] → FAISS vector store
        ↓
    [Grade Documents] → LLM-as-judge filters irrelevant chunks
        ├── all irrelevant → [Rewrite Query] → re-retrieve (max 2 retries)
        └── relevant found → [Generate]
        ↓
    [Generate] → LLM produces grounded answer
        ↓
    [Grade Generation]
        ├── hallucinated → [Regenerate] → re-generate
        ├── not useful   → [Rewrite Query] → re-retrieve
        └── useful       → END (return answer to user)
"""

import os
from typing import List, Tuple, TypedDict, Literal

import streamlit as st
from io import BytesIO

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

import pypdf


# --- Constants ---
SOURCE_DOCUMENT_PATH = "Cracking The Machine Learning Interview.pdf"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_DIR = os.getenv("LOCAL_EMB_DIR", "./models/all-MiniLM-L6-v2")
FIXED_TEMPERATURE = 0.1
MAX_RETRIES = 2  # Max agent retries before forcing a final answer


# ---------------- boot ----------------
load_dotenv()
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Agentic RAG Chatbot 🤖")
st.subheader(f"Grounded on: **{SOURCE_DOCUMENT_PATH}**")


# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("Keys & Models")

    gemini_key = st.text_input(
        "GOOGLE_API_KEY (or GEMINI_API_KEY)",
        type="password",
        value=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "",
    )
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

    hf_token = st.text_input(
        "Hugging Face token (optional—for first download)",
        type="password",
        value=os.getenv("HUGGINGFACE_HUB_TOKEN") or "",
    )
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    tavily_key = st.text_input(
        "Tavily API Key (web search fallback)",
        type="password",
        value=os.getenv("TAVILY_API_KEY") or "",
        help="Get a free key at https://tavily.com — enables web search when documents lack context.",
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    gemini_model = st.selectbox(
        "Gemini Model (LLM Choice)",
        ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
    )
    st.caption(f"Temperature: *{FIXED_TEMPERATURE}*")

    st.divider()
    st.subheader("Embeddings")
    st.caption("First run downloads the model to disk; later runs load from disk.")
    local_dir = st.text_input("Local embedding folder", value=DEFAULT_LOCAL_DIR)
    laptop_mode = st.toggle("Laptop-friendly mode", value=True)

    st.divider()
    st.subheader("Retrieval")
    k = st.slider("Top-K chunks", 1, 10, 4)
    mmr = st.toggle("Use MMR (diverse results)", value=True)

    st.divider()
    st.subheader("Chunking")
    chunk_mode = st.radio("Mode", ["Auto (recommended)", "Manual"], index=0)
    if chunk_mode == "Manual":
        chunk_size = st.number_input("Chunk size (chars)", 128, 4000, 800, step=64)
        chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 1000, 120, step=20)
    else:
        chunk_size = None
        chunk_overlap = None

    st.divider()
    st.subheader("🤖 Agentic Settings")
    show_agent_steps = st.toggle("Show agent reasoning steps", value=True,
                                  help="Display each step the agent takes (retrieve, grade, rewrite, generate).")
    st.caption(f"Max retries: *{MAX_RETRIES}*")

    st.divider()
    if st.button("Clear FAISS index", type="secondary"):
        st.session_state.pop("vs", None)
        st.session_state.pop("retriever", None)
        st.success("Index cleared.")


# ---------------- session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ---------------- helpers ----------------
def have_local_model(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    needed = ["config.json", "tokenizer.json"]
    present = all(os.path.exists(os.path.join(folder, f)) for f in needed)
    has_weights = any(
        os.path.exists(os.path.join(folder, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    )
    return present and has_weights


@st.cache_resource(show_spinner=False)
def ensure_local_model(model_id: str, folder: str, token: str | None) -> str:
    if have_local_model(folder):
        return folder
    os.makedirs(folder, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=folder,
        local_dir_use_symlinks=False,
        token=token or os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    return folder


def suggest_chunk_params(laptop: bool) -> Tuple[int, int]:
    size, overlap = (900, 120)
    if laptop:
        size = 700
        overlap = 100
    return size, overlap


def build_splitter() -> RecursiveCharacterTextSplitter:
    if chunk_mode == "Manual":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    size, overlap = suggest_chunk_params(laptop_mode)
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def pdf_to_langchain_docs(pdf_path: str) -> List[Document]:
    docs = []
    try:
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
    except FileNotFoundError:
        st.error(f"FATAL: Source document not found at: {pdf_path}")
        return []
    pdf_reader = pypdf.PdfReader(BytesIO(pdf_content))
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": i + 1},
            ))
    return docs


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        formatted.append(f"[{source.split('/')[-1]}, page {page}]\n{d.page_content}")
    return "\n\n".join(formatted)


def ensure_retriever():
    if st.session_state.retriever is not None:
        return st.session_state.retriever
    if st.session_state.vs is None:
        return None
    if mmr:
        fetch_k = max(k * (2 if laptop_mode else 4), 10)
        retr = st.session_state.vs.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k}
        )
    else:
        retr = st.session_state.vs.as_retriever(search_kwargs={"k": k})
    st.session_state.retriever = retr
    return retr


@st.cache_resource(show_spinner=False)
def load_embeddings_from_folder(folder: str, normalize: bool = True) -> HuggingFaceEmbeddings:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return HuggingFaceEmbeddings(
        model_name=folder,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize},
    )


# ============================================================
# AGENTIC RAG — LangGraph Graph Definition
# ============================================================

# ── Pydantic models for structured LLM outputs ──────────────
class RouteQuery(BaseModel):
    """Route user question to vectorstore or web search."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description=(
            "Route to 'vectorstore' for ML, data science, or interview questions covered in the document. "
            "Route to 'web_search' for current events or topics clearly outside the document."
        )
    )


class GradeBatchDocuments(BaseModel):
    """Indices of relevant documents from the retrieved set — graded in one batched LLM call."""
    relevant_indices: List[int] = Field(
        description=(
            "Zero-based indices of documents that ARE relevant to the question. "
            "Be lenient — partial relevance counts. Return an empty list only if ALL documents are completely off-topic."
        )
    )


class GradeGeneration(BaseModel):
    """Combined hallucination check + usefulness check — done in one LLM call."""
    grounded: Literal["yes", "no"] = Field(
        description="'yes' if every claim in the answer is supported by the provided facts, 'no' if it hallucinates."
    )
    useful: Literal["yes", "no"] = Field(
        description="'yes' if the answer resolves the question, 'no' if it is incomplete or off-topic."
    )


# ── Graph State ──────────────────────────────────────────────
class GraphState(TypedDict):
    """State object that flows through every node of the LangGraph."""
    question: str
    generation: str
    web_search_needed: bool
    documents: List[Document]
    retries: int


# ── Graph Builder ────────────────────────────────────────────
def build_agentic_graph(llm: ChatGoogleGenerativeAI, retriever):
    """
    Build and compile the Agentic RAG LangGraph.

    Nodes
    -----
    retrieve        : Query the FAISS vector store.
    grade_documents : LLM-as-judge filters irrelevant chunks.
    rewrite_query   : LLM rewrites the question for better retrieval.
    web_search      : Tavily web search used as fallback.
    generate        : LLM generates the final grounded answer.
    regenerate      : Increments retry counter before re-generating.

    Conditional Edges
    -----------------
    route_question      : vectorstore | web_search
    decide_after_grading: rewrite | generate
    grade_generation    : useful | not_useful | not_grounded
    """

    # ── Structured-output chains (LLM-as-judge) ───────────────
    router_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert router. Route the question to 'vectorstore' if it is about "
             "machine learning, statistics, data science, interviews, algorithms, or any topic "
             "likely covered in an ML interview preparation book. "
             "Route to 'web_search' only for clearly current events or topics outside such a book."),
            ("human", "{question}"),
        ])
        | llm.with_structured_output(RouteQuery)
    )

    # Grades ALL chunks in ONE call — avoids one LLM call per chunk
    batch_doc_grader_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a relevance grader. Given a list of numbered document chunks and a question, "
             "return the zero-based indices of the chunks that are relevant. "
             "Be lenient — include a chunk if it has any partial relevance. "
             "Return an empty list only if every chunk is completely off-topic."),
            ("human",
             "Question: {question}\n\nDocuments:\n{documents}\n\n"
             "Return the indices of relevant documents."),
        ])
        | llm.with_structured_output(GradeBatchDocuments)
    )

    # Combines hallucination check + answer quality check into ONE call
    generation_grader_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a generation grader. Given retrieved facts, an answer, and the original question:\n"
             "1. Check if the answer is grounded in the facts (no hallucinations).\n"
             "2. Check if the answer is useful and resolves the question.\n"
             "Grade both 'grounded' and 'useful' as 'yes' or 'no'."),
            ("human",
             "Facts:\n{documents}\n\nQuestion: {question}\n\nAnswer: {generation}"),
        ])
        | llm.with_structured_output(GradeGeneration)
    )

    rewriter_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a question rewriter. Improve the input question to better optimize "
             "semantic vectorstore retrieval. Focus on the core intent. Return ONLY the rewritten question."),
            ("human", "Original question: {question}\n\nImproved question:"),
        ])
        | llm
        | StrOutputParser()
    )

    generation_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful, accurate, document-grounded AI assistant. "
             "Answer using ONLY the provided context chunks. "
             "Include citations (e.g., [filename.pdf, page X]) after each fact you use. "
             "If the answer is not in the context, respond: "
             "'I cannot find the answer in the document.'"),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"),
        ])
        | llm
        | StrOutputParser()
    )

    # Web search — only active when TAVILY_API_KEY is set
    web_search_tool = (
        TavilySearch(max_results=3) if os.getenv("TAVILY_API_KEY") else None
    )

    # ── Node functions ────────────────────────────────────────
    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def grade_documents(state: GraphState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"documents": [], "web_search_needed": True}

        # Format all chunks into a numbered list for one batched LLM call
        numbered = "\n\n".join(
            f"[{i}] {d.page_content[:600]}" for i, d in enumerate(docs)
        )
        result = batch_doc_grader_chain.invoke({
            "question": state["question"],
            "documents": numbered,
        })

        # Keep only the chunks the LLM flagged as relevant
        valid_indices = {i for i in result.relevant_indices if 0 <= i < len(docs)}
        filtered = [docs[i] for i in sorted(valid_indices)]
        web_search_needed = len(filtered) < len(docs) or not filtered

        if not filtered:
            web_search_needed = True
        return {"documents": filtered, "web_search_needed": web_search_needed}

    def rewrite_query(state: GraphState) -> dict:
        better_q = rewriter_chain.invoke({"question": state["question"]})
        return {
            "question": better_q,
            "retries": state.get("retries", 0) + 1,
            "web_search_needed": False,
        }

    def web_search(state: GraphState) -> dict:
        if web_search_tool is None:
            fallback = Document(
                page_content=(
                    "Web search is unavailable. Add a TAVILY_API_KEY in the sidebar to enable it."
                ),
                metadata={"source": "system", "page": "n/a"},
            )
            return {"documents": [fallback]}

        # TavilySearch (langchain_tavily) accepts a plain string query
        raw = web_search_tool.invoke(state["question"])

        # langchain_tavily returns a formatted string; older TavilySearchResults returned list of dicts
        if isinstance(raw, str):
            web_docs = [
                Document(
                    page_content=raw,
                    metadata={"source": "Tavily Web Search", "page": "web"},
                )
            ]
        elif isinstance(raw, list):
            web_docs = [
                Document(
                    page_content=r.get("content", str(r)) if isinstance(r, dict) else str(r),
                    metadata={
                        "source": r.get("url", "web") if isinstance(r, dict) else "web",
                        "page": "web",
                    },
                )
                for r in raw
            ]
        else:
            web_docs = [
                Document(
                    page_content=str(raw),
                    metadata={"source": "Tavily Web Search", "page": "web"},
                )
            ]

        return {"documents": state.get("documents", []) + web_docs}

    def generate(state: GraphState) -> dict:
        context = (
            format_docs(state["documents"]) if state["documents"] else "No context available."
        )
        generation = generation_chain.invoke({
            "question": state["question"],
            "context": context,
        })
        return {"generation": generation}

    def regenerate(state: GraphState) -> dict:
        """Increment retry counter so grade_generation can enforce the retry limit."""
        return {"retries": state.get("retries", 0) + 1}

    # ── Conditional edge functions ────────────────────────────
    def route_question(state: GraphState) -> str:
        result = router_chain.invoke({"question": state["question"]})
        return result.datasource  # "vectorstore" | "web_search"

    def decide_after_grading(state: GraphState) -> str:
        if state.get("web_search_needed") and state.get("retries", 0) < MAX_RETRIES:
            return "rewrite"
        return "generate"

    def grade_generation(state: GraphState) -> str:
        # Enforce retry ceiling to prevent infinite loops
        if state.get("retries", 0) >= MAX_RETRIES:
            return "useful"

        # Single combined call: hallucination check + answer quality check
        grade = generation_grader_chain.invoke({
            "documents": format_docs(state["documents"]),
            "question": state["question"],
            "generation": state["generation"],
        })

        if grade.grounded == "no":
            return "not_grounded"
        return "useful" if grade.useful == "yes" else "not_useful"

    # ── Build graph ───────────────────────────────────────────
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("regenerate", regenerate)

    # Entry point: route question to vectorstore or web_search
    workflow.add_conditional_edges(
        START,
        route_question,
        {"vectorstore": "retrieve", "web_search": "web_search"},
    )

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )

    # After rewriting, go back and retrieve again
    workflow.add_edge("rewrite_query", "retrieve")

    # Web search results go directly to generation
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "not_useful": "rewrite_query",   # answer didn't address the question
            "not_grounded": "regenerate",    # hallucination detected — regenerate
        },
    )

    # After incrementing retries, try generating again
    workflow.add_edge("regenerate", "generate")

    return workflow.compile()


# ---------------- ingest ----------------
st.subheader("RAG Pipeline Status")

if not os.path.exists(SOURCE_DOCUMENT_PATH):
    st.error(
        f"The required source document **{SOURCE_DOCUMENT_PATH}** must be placed "
        "in the same directory as this script."
    )
    st.stop()

c1, c2 = st.columns([1, 1])

with c1:
    if st.button("Build / Rebuild Index", type="primary"):
        try:
            with st.spinner("Preparing embedding model (first time may download)…"):
                folder = ensure_local_model(HF_MODEL_ID, local_dir, hf_token or None)
        except Exception as e:
            st.error(
                f"Could not download the embedding model. "
                f"Check your connection or HUGGINGFACE_HUB_TOKEN. Error: {e}"
            )
            st.stop()

        embeddings = load_embeddings_from_folder(folder)
        splitter = build_splitter()
        raw_docs = pdf_to_langchain_docs(SOURCE_DOCUMENT_PATH)

        if not raw_docs:
            st.error("Could not extract any text from the PDF. Check the file.")
            st.stop()

        chunks = splitter.split_documents(raw_docs)

        with st.spinner("Embedding & indexing…"):
            st.session_state.vs = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = None

        st.success(
            f"Indexed {len(chunks)} chunks from {len(raw_docs)} pages of {SOURCE_DOCUMENT_PATH}."
        )

with c2:
    if have_local_model(local_dir):
        st.info(f"Local embedding ready ✅  ({HF_MODEL_ID})")
    else:
        st.warning("Local embedding not present yet. It will be downloaded on first index build.")


# ---------------- LLM & Agentic Graph ----------------
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    st.warning("Add your *GOOGLE_API_KEY* (or *GEMINI_API_KEY*) in the sidebar to query Gemini.")
else:
    try:
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=FIXED_TEMPERATURE)
    except Exception:
        st.toast("Selected model not available, falling back to gemini-1.5-flash.", icon="⚠️")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=FIXED_TEMPERATURE)

    retriever = ensure_retriever()
    agentic_graph = build_agentic_graph(llm, retriever) if retriever else None

    # ---------------- chat UI ----------------
    st.subheader("Ask the Agentic ML Interview Chatbot")

    # Replay chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_q = st.chat_input("Ask about ML concepts, algorithms, interview tips…")

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)
        st.session_state.messages.append(("user", user_q))

        if agentic_graph is None:
            with st.chat_message("assistant"):
                st.warning("Please **Build / Rebuild Index** first to start the chatbot.")
        else:
            with st.chat_message("assistant"):

                NODE_LABELS = {
                    "retrieve":        "🔍 Retrieving relevant chunks from vector store…",
                    "grade_documents": "📋 Grading document relevance (LLM-as-judge)…",
                    "rewrite_query":   "✏️  Query not good enough — rewriting for better retrieval…",
                    "web_search":      "🌐 Searching the web for additional context…",
                    "generate":        "💭 Generating grounded answer…",
                    "regenerate":      "🔄 Hallucination detected — re-generating…",
                }

                initial_state: GraphState = {
                    "question": user_q,
                    "generation": "",
                    "web_search_needed": False,
                    "documents": [],
                    "retries": 0,
                }

                answer = ""

                if show_agent_steps:
                    with st.status("🤖 Agent reasoning…", expanded=True) as status:
                        for update in agentic_graph.stream(
                            initial_state, stream_mode="updates"
                        ):
                            node_name = list(update.keys())[0]
                            node_data = update[node_name]
                            st.write(NODE_LABELS.get(node_name, f"⚙️ {node_name}…"))
                            if node_data.get("generation"):
                                answer = node_data["generation"]
                        status.update(
                            label="Agent complete ✅", state="complete", expanded=False
                        )
                else:
                    for update in agentic_graph.stream(
                        initial_state, stream_mode="updates"
                    ):
                        node_data = list(update.values())[0]
                        if node_data.get("generation"):
                            answer = node_data["generation"]

                if answer:
                    st.markdown(answer)
                else:
                    st.warning(
                        "The agent could not produce an answer. "
                        "Try rebuilding the index or rephrasing your question."
                    )

            st.session_state.messages.append(("assistant", answer))
