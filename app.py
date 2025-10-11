import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. Include numbered citations [1], [2], etc., linking back to the source documents.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. Include a "References" section at the end listing the source documents with their citation numbers.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks."""
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search."""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents."""
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def re_rank_cross_encoders(documents: list[str], ids: list[str], prompt: str) -> tuple[str, list[int], list[str]]:
    """Re-ranks documents and attaches citation numbers."""
    if not documents:
        return "", [], []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    relevant_text = ""
    relevant_text_ids = []
    relevant_doc_ids = []

    for citation_number, rank in enumerate(ranks, 1):
        idx = rank["corpus_id"]
        relevant_text += f"{documents[idx]} [{citation_number}]\n\n"
        relevant_text_ids.append(idx)
        relevant_doc_ids.append(ids[idx])

    return relevant_text, relevant_text_ids, relevant_doc_ids


def call_llm(context: str, prompt: str):
    """Calls the LLM to generate a response using the provided context."""
    response = ollama.chat(
        model="koesn/llama3-8b-instruct:latest",  # updated model
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


if __name__ == "__main__":
    # Streamlit Sidebar: Upload PDF
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer with Citations")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button("⚡️ Process")
        if uploaded_file and process:
            normalized_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalized_file_name)

    # Main Area: Ask Questions
    st.header("🗣️ RAG Question Answer with Citations")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context_docs = results.get("documents")[0]
        context_ids = results.get("ids")[0]  # assuming IDs are returned
        relevant_text, relevant_text_indices, relevant_doc_ids = re_rank_cross_encoders(
            context_docs, context_ids, prompt
        )

        # Stream LLM Response
        response_stream = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response_stream)

        # Citation Transparency Section
        with st.expander("See retrieved documents"):
            for i, doc_id in enumerate(relevant_doc_ids, 1):
                st.markdown(f"**[{i}] Source:** {doc_id}")

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_indices)
            st.write(relevant_text)
