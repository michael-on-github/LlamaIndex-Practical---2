import json
import os
import re
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

import chromadb

load_dotenv()

from typing import List

from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.tools import FunctionTool
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from schemas import Candidate, CandidateSummary
from utils import DATA_DIR, STORAGE_DIR, embed_model, llm

GROUPED_CHUNKS_PATH = f"{STORAGE_DIR}/grouped_chunks.json"
CANDIDATES_ROSTER_PATH = f"{STORAGE_DIR}/candidates_roster.json"
nodes = []
grouped_chunks = {}
candidates_roster = {}
candidates = []

Settings.embed_model = embed_model
Settings.llm = llm


def create_nodes():
    progress_text.empty()
    progress_text.text("Creating nodes...")
    documents = []
    pdf_loader = PyMuPDFReader()
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            docs = pdf_loader.load_data(file_path)
            documents.extend(docs)
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=75, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    progress_text.text("Nodes created")
    return nodes


def ensure_nodes():
    if len(nodes):
        return nodes
    else:
        return create_nodes()


def create_index(nodes: List[BaseNode]):
    progress_text.empty()
    progress_text.text("Creating index...")
    texts = [node.text for node in nodes]
    ids = [f"text_{i}" for i in range(len(texts))]
    embeddings = [embed_model.get_text_embedding(text) for text in texts]
    metadatas = [node.metadata for node in nodes]

    db = chromadb.PersistentClient(path=f"{STORAGE_DIR}")
    chroma_collection = db.get_or_create_collection("resumes")
    if chroma_collection.count() == 0:
        chroma_collection.add(
            documents=texts, ids=ids, embeddings=embeddings, metadatas=metadatas
        )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=f"{STORAGE_DIR}")
    progress_text.text("Index created")
    return index


@st.cache_resource
def load_index():
    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection("resumes")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def ensure_index():
    if os.path.exists(STORAGE_DIR):
        index = load_index()
    else:
        os.makedirs(STORAGE_DIR)
        nodes = ensure_nodes()
        index = create_index(nodes)
    return index


def ensure_grouped_chunks():
    ensure_index()
    grouped_chunks = {}
    if os.path.exists(GROUPED_CHUNKS_PATH):
        with open(GROUPED_CHUNKS_PATH, "r") as json_file:
            grouped_chunks = json.load(json_file)
    else:
        grouped_chunks = defaultdict(list)
        nodes = ensure_nodes()
        for node in nodes:
            file_path = node.metadata.get("file_path", "unknown")
            grouped_chunks[file_path].append(node.text)
        with open(GROUPED_CHUNKS_PATH, "w") as json_file:
            json.dump(grouped_chunks, json_file, indent=4)
    return grouped_chunks


def ensure_candidates_roster():
    ensure_index()
    grouped_chunks = ensure_grouped_chunks()
    candidates_roster = {}
    if os.path.exists(CANDIDATES_ROSTER_PATH):
        with open(CANDIDATES_ROSTER_PATH, "r") as json_file:
            candidates_roster = json.load(json_file)
    else:
        candidates_length = len(grouped_chunks.keys())
        for i, (file_path, chunks) in enumerate(grouped_chunks.items()):
            progress_text.text(
                f"Extracting job title, profession, etc for {candidates_length} candidates..."
            )
            full_text = " ".join(chunks)
            prompt = f"""
                Extract the relevant information from the resume below.

                Resume:
                {full_text}
            """
            input_msg = ChatMessage.from_str(prompt)
            sllm = llm.as_structured_llm(output_cls=Candidate)
            output = sllm.chat([input_msg])
            output_json = output.raw.model_dump_json()
            candidates_roster[file_path] = json.loads(output_json)

            progress_text.text(
                f"Job title, profession, etc for {i + 1} of {candidates_length} candidates extracted."
                + (" Extracting more..." if (i + 1) < candidates_length else "")
            )
        with open(CANDIDATES_ROSTER_PATH, "w") as json_file:
            json.dump(candidates_roster, json_file, indent=4)

    return candidates_roster


def load_candidates():
    progress_text.empty()

    if os.path.exists(STORAGE_DIR):
        progress_text.text("Loading existing candidates...")
        grouped_chunks = ensure_grouped_chunks()
        progress_text.text("Candidates loaded")
    else:
        progress_text.text("Creating a new set of candidates' information...")
        grouped_chunks = ensure_grouped_chunks()
        progress_text.text("Candidates created")

    candidates_length = len(grouped_chunks.keys())
    progress_text.text(f"Generating summaries for {candidates_length} candidates...")

    for i, (file_path, chunks) in enumerate(grouped_chunks.items()):
        full_text = " ".join(chunks)
        prompt = f"""
        From the candidate's resume below, generate a concise summary of their strongest skills and professional highlights.

        Respond with:
        - "summary": A brief overview in no more than 3 sentences.
        - "details": A more specific breakdown of notable achievements, technologies used, and roles held.

        Resume:
        {full_text}
        """
        input_msg = ChatMessage.from_str(prompt)
        sllm = llm.as_structured_llm(output_cls=CandidateSummary)
        output = sllm.chat([input_msg])
        output_obj = output.raw
        candidates.append(
            {
                "name": file_path,
                "summary": output_obj.summary,
                "details": output_obj.details,
            }
        )
        progress_text.text(
            f"Summary for {i + 1} of {candidates_length} candidates generated."
            + (" Generating more..." if (i + 1) < candidates_length else "")
        )
    progress_text.text(f"Summaries for {candidates_length} candidates are generated")
    return candidates


### Streamlit

st.set_page_config(
    page_title="Candidates Central",
    page_icon="❂",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Candidates Central")
progress_text = st.empty()

if st.button("Load Candidates"):
    with st.spinner("Loading candidates..."):
        st.session_state.candidates = load_candidates()
    st.success("Candidates loaded!")

if "candidates" in st.session_state:
    for candidate in st.session_state.candidates:
        with st.expander(candidate["name"]):
            st.write(f"**Summary:** {candidate['summary']}")
            st.write(f"**Details:** {candidate['details']}")

# ---


def retrieve(input: str) -> str:
    """
    Retrieves information from VectorStoreIndex.
    Stop after retrieving.
    """
    query = input
    index = ensure_index()

    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    response = query_engine.query(query)
    print("\n-----\nRaw response:", response)
    # return response.response
    return f"Observation: {response.response}\nFinal Answer: {response.response}"


retrieval_tool = FunctionTool.from_defaults(
    fn=retrieve,
    name="retrieve",
    description="Retrieves information about candidates from VectorStoreIndex. Stop after retrieving.",
)


class ChatResponse:
    def __init__(self, response_text):
        self.response = response_text


def respond(query: str) -> ChatResponse:
    """
    Responds to information from query.
    Stop after responding.
    """
    response = llm.complete(f"Respond to the following query, max 3 sentences: {query}")
    return ChatResponse(response.text)


response_tool = FunctionTool.from_defaults(
    fn=respond,
    name="respond",
    description="Responds to general queries. Stop after responding.",
)

import asyncio

from llama_index.core.agent.workflow import (AgentOutput, AgentStream,
                                             ReActAgent, ToolCallResult)
from llama_index.core.workflow import Context
from llama_index.tools.database.base import DatabaseToolSpec

db_tools = DatabaseToolSpec(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="FakeExamplePassword",  # Database Password
    dbname="postgres",  # Database Name
)

all_tools = [
    FunctionTool.from_defaults(
        fn=db_tools.to_tool_list()[0].fn,
        name="load_data",
        description="Loads data from a table. Accepts a single string argument: the table name. "
        "The database is PostgreSQL.",
    ),
    retrieval_tool,
    response_tool,
]

agent = ReActAgent(
    tools=all_tools,
    llm=llm,
    max_execution_time=20,
    early_stopping_method="force",
)

ctx = Context(agent)

from llama_index.core.workflow.errors import WorkflowRuntimeError


async def get_response(query):
    final_answer = None
    try:
        handler = agent.run(query, ctx=ctx, max_iterations=5)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)
            elif isinstance(ev, ToolCallResult):
                print(f"ToolCallResult: {ev}")
                for block in ev.tool_output.blocks:
                    if hasattr(block, "text"):
                        matches = re.findall(r"Final Answer:\s*(.*)", block.text)
                        if matches:
                            final_answer = matches[-1].strip()
        return await handler  # Try returning structured output
    except WorkflowRuntimeError:
        print("Agent stopped due to max iterations.")
        return final_answer  # Fallback to extracted answer


query = st.text_input("Enter your query:")
if query:
    with st.spinner("Preparing an answer..."):
        response = asyncio.run(get_response(query))
        print(type(response))
        print("\n--- Done ---\n")
        progress_text.empty()

        if isinstance(response, AgentOutput):
            for block in response.response.blocks:
                st.write(block.text)
        elif isinstance(response, ChatResponse):
            st.write(response.text)
        elif isinstance(response, str):
            st.write(response)
        else:
            st.write("⚠️ Unexpected response type:")
            st.write(str(response))
