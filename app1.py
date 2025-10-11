import os
import streamlit as st
import chromadb
import ollama
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from extractors.text_extractor import extract_text_and_tables, chunk_documents
from extractors.image_extractor import extract_images_with_context
from extractors.audio_transcriber import transcribe_audio
from langchain_core.documents import Document

st.set_page_config(page_title="📚 Multimodal RAG System", layout="wide")
def remove_pdf_data(file_name):
    """Remove all documents and images related to a PDF."""
    collection = get_collection()

    # 1. Find all document IDs related to this PDF
    results = collection.query(
        query_texts=[""],
        n_results=10000  # assuming max 10k docs per PDF
    )
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    ids_to_delete = []
    for i, meta in enumerate(metadatas):
        if meta.get("source_file") == file_name:
            ids_to_delete.append(ids[i])
            # Delete image file if it's an image doc
            if meta.get("type") == "image":
                img_path = meta.get("image_name")
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)

    # 2. Delete docs from ChromaDB
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    # 3. Delete PDF file itself
    pdf_path = os.path.join("uploads", file_name)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

# -------------------- VECTOR STORE --------------------
def get_collection():
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )
    client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return client.get_or_create_collection(
        name="multi_rag",
        embedding_function=ollama_ef
    )

# -------------------- ADD DATA --------------------
def add_docs(docs, file_name):
    if not docs:
        return
    collection = get_collection()
    docs_content = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [f"{file_name}_{i}" for i in range(len(docs))]
    collection.upsert(documents=docs_content, metadatas=metadatas, ids=ids)

# -------------------- QUERY --------------------
def query_docs(query, n=10):
    coll = get_collection()
    return coll.query(query_texts=[query], n_results=n)

# -------------------- CROSS-ENCODER RE-RANK --------------------
def rerank(prompt, docs, ids):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = model.rank(prompt, docs, top_k=3)
    final_text = ""
    citations = []
    for i, r in enumerate(ranks, start=1):
        idx = r["corpus_id"]
        final_text += f"{docs[idx]} [{i}]\n\n"
        citations.append((i, ids[idx]))
    return final_text, citations

# -------------------- LLM CALL --------------------
def call_llm(context, question):
    response = ollama.chat(
        model="koesn/llama3-8b-instruct:latest",
        stream=True,
        messages=[
            {"role": "system", "content": "Answer clearly using only the context provided. Include numbered citations."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("📂 Upload Data")
    uploaded_pdf = st.file_uploader("Upload PDF/DOC", type=["pdf", "docx"])
    uploaded_audio = st.file_uploader("🎤 Upload Audio", type=["wav", "mp3"])
    process = st.button("Process File")

    # -------------------- REMOVE PDF --------------------
    remove_pdf = st.text_input("Enter PDF filename to delete (with extension)")
    delete_btn = st.button("Delete PDF & Images")

    if delete_btn and remove_pdf:
        def remove_pdf_data(file_name):
            collection = get_collection()

            # 1️⃣ Remove from ChromaDB
            collection.delete(where={"source_file": file_name})

            # 2️⃣ Remove PDF file
            pdf_path = os.path.join("uploads", file_name)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

            # 3️⃣ Remove related images
            for f in os.listdir("uploads"):
                if f.startswith(file_name.replace(".pdf", "")) and f.endswith(".png"):
                    os.remove(os.path.join("uploads", f))


        remove_pdf_data(remove_pdf)
        st.success(f"{remove_pdf} and its images have been deleted ✅")

    # -------------------- PROCESS PDF / AUDIO --------------------
    if process and uploaded_pdf:
        file_path = os.path.join("uploads", uploaded_pdf.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.read())

        # Text
        text_docs = extract_text_and_tables(file_path)
        text_chunks = chunk_documents(text_docs)
        add_docs(text_chunks, uploaded_pdf.name)

        # Images with context
        images = extract_images_with_context(file_path)
        add_docs(images, uploaded_pdf.name)  # extract_images already returns Document objects

        st.success(f"Processed {len(text_chunks)} text chunks & {len(images)} images ✅")

    if process and uploaded_audio:
        # Audio transcription
        text = transcribe_audio(uploaded_audio)
        audio_doc = Document(page_content=text, metadata={
            "type": "audio",
            "audio_path": os.path.join("uploads", uploaded_audio.name)
        })
        add_docs([audio_doc], uploaded_audio.name)
        st.success("Audio transcribed & indexed ✅")

# -------------------- MAIN CHAT UI --------------------
st.title("💬 Multimodal RAG Assistant")
query = st.text_area("Ask a question across all data formats...")
if st.button("Ask"):
    result = query_docs(query)
    docs = result["documents"][0]
    ids = result["ids"][0]
    context, citations = rerank(query, docs, ids)

    # -------------------- LLM RESPONSE --------------------
    st.subheader("Answer:")
    st.write_stream(call_llm(context, query))

    # -------------------- CITATIONS --------------------
    st.subheader("📜 Citations")
    collection = get_collection()
    for num, cid in citations:
        result = collection.get(ids=[cid])
        paragraph_text = result["documents"][0]  # actual paragraph text
        meta = result["metadatas"][0]
        pdf_name = meta.get("source_file", "Unknown")
        page_no = meta.get("page_no", "-")

        st.markdown(f"""
            <div style="border:1px solid #aaa; border-radius:6px; padding:8px; margin-bottom:8px;">
                <div style="font-weight:bold; background-color:#e0e0e0; padding:4px 6px; border-radius:4px;">
                    {pdf_name} - Page {page_no}
                </div>
                <div style="margin-top:4px;">{paragraph_text}</div>
            </div>
        """, unsafe_allow_html=True)

    # -------------------- IMAGES --------------------
    st.subheader("🖼️ Images")
    image_docs = [
        collection.get(ids=[cid])["metadatas"][0]
        for _, cid in citations
        if collection.get(ids=[cid])["metadatas"][0].get("type") == "image"
    ]
    cols = st.columns(3)
    for idx, img_meta in enumerate(image_docs):
        col = cols[idx % 3]
        img_path = img_meta.get("image_name")
        if img_path and os.path.exists(img_path):
            col.image(
                img_path,
                width=300,
                caption=f"{img_meta.get('source_file','Unknown')} - Page {img_meta.get('page_no','-')}",
                use_container_width=False
            )
