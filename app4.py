import os
import streamlit as st
import tempfile
import chromadb
import ollama
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from extractors.text_extractor import extract_text_and_tables, chunk_documents
from extractors.audio_transcriber import transcribe_audio
from langchain_core.documents import Document
from streamlit_mic_recorder import mic_recorder
import fitz  # PyMuPDF
import json
from PIL import Image
import pytesseract

st.set_page_config(page_title="📚 Multimodal RAG System", layout="wide")

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

# -------------------- REMOVE PDF / AUDIO DATA --------------------
def remove_pdf_data(file_name):
    collection = get_collection()
    results = collection.query(query_texts=[""], n_results=10000)
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    ids_to_delete = []
    for i, meta in enumerate(metadatas):
        if meta.get("source_file") == file_name:
            ids_to_delete.append(ids[i])

            if meta.get("type") == "image":
                img_path = meta.get("image_name")
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)

            if meta.get("type") == "audio":
                audio_path = meta.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    file_path = os.path.join("uploads", file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    base_name = os.path.splitext(file_name)[0]
    json_path = os.path.join("uploads", f"{base_name}_images.json")
    if os.path.exists(json_path):
        os.remove(json_path)

# -------------------- REMOVE EVERYTHING --------------------
def remove_all_data():
    collection = get_collection()
    results = collection.query(query_texts=[""], n_results=10000)
    ids = results["ids"][0]
    if ids:
        collection.delete(ids=ids)

    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            f_path = os.path.join(uploads_dir, f)
            if os.path.isfile(f_path):
                os.remove(f_path)

# -------------------- QUERY --------------------
def query_docs(query, n=10):
    coll = get_collection()
    return coll.query(query_texts=[query], n_results=n)

# -------------------- CROSS-ENCODER RE-RANK --------------------
def rerank(prompt, docs, ids):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = model.rank(prompt, docs, top_k=len(docs))
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
        if not chunk.get("done"):
            yield chunk["message"]["content"]

# -------------------- IMAGE EXTRACTION FUNCTION --------------------
def extract_images_with_context(pdf_path: str):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_context_data = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_no, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")
        text_blocks = sorted(text_blocks, key=lambda b: b[1])

        for img_no, img in enumerate(images, start=1):
            xref = img[0]
            smask = img[1]
            bbox = img[1:5]
            img_bottom = bbox[3]

            try:
                main_pix = fitz.Pixmap(doc, xref)
                if smask != 0:
                    mask_pix = fitz.Pixmap(doc, smask)
                    if mask_pix.alpha or mask_pix.colorspace.n == 1:
                        main_pix = fitz.Pixmap(main_pix, mask_pix)
                    mask_pix = None
                else:
                    if main_pix.colorspace is None or main_pix.n < 3:
                        continue

                img_filename = f"{pdf_name}_page{page_no}_img{img_no}.png"
                img_path = os.path.join(uploads_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(main_pix.tobytes("png"))

                below_paras = [t for t in text_blocks if t[1] > img_bottom]
                context_text = ""
                if below_paras:
                    closest_para = min(below_paras, key=lambda t: t[1] - img_bottom)
                    context_text = closest_para[4].strip()

                image_context_data.append({
                    "page_no": page_no,
                    "image_path": img_path,
                    "bbox": bbox,
                    "context": context_text if context_text else "No nearby text found."
                })
                main_pix = None

            except Exception as e:
                print(f"Error processing page {page_no}, image {img_no}: {e}")

    json_path = os.path.join(uploads_dir, f"{pdf_name}_images.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_context_data, f, indent=4, ensure_ascii=False)

    return json_path, image_context_data

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("📂 Upload Data")
    uploaded_pdf = st.file_uploader("Upload PDF/DOC", type=["pdf", "docx"])
    uploaded_audio = st.file_uploader("🎤 Upload Audio", type=["wav", "mp3"])
    process = st.button("Process File")
    remove_pdf_btn = st.button("Remove Uploaded PDF")
    remove_all_btn = st.button("Remove Everything")

    if process and uploaded_pdf:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_pdf.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.read())

        text_docs = extract_text_and_tables(file_path)
        text_chunks = chunk_documents(text_docs)
        add_docs(text_chunks, uploaded_pdf.name)

        json_path, image_context_data = extract_images_with_context(file_path)
        image_docs = [
            Document(
                page_content=img["context"],
                metadata={
                    "type": "image",
                    "image_name": img["image_path"],
                    "page_no": img["page_no"],
                    "source_file": uploaded_pdf.name
                }
            )
            for img in image_context_data
        ]
        add_docs(image_docs, uploaded_pdf.name)
        st.success(f"Processed {len(text_chunks)} text chunks & {len(image_docs)} images ✅")
        st.info(f"Image metadata JSON saved at: {json_path}")

    if process and uploaded_audio:
        os.makedirs("uploads", exist_ok=True)
        audio_path = os.path.join("uploads", uploaded_audio.name)
        uploaded_audio.seek(0)
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.read())

        text = transcribe_audio(audio_path)
        audio_doc = Document(
            page_content=text,
            metadata={
                "type": "audio",
                "source_file": uploaded_audio.name,
                "audio_path": audio_path
            }
        )
        add_docs([audio_doc], uploaded_audio.name)
        st.success("Audio transcribed & indexed ✅")

    if remove_pdf_btn and uploaded_pdf:
        remove_pdf_data(uploaded_pdf.name)
        st.warning(f"Deleted {uploaded_pdf.name} and all related files ✅")

    if remove_all_btn:
        remove_all_data()
        st.warning("Deleted all uploads and ChromaDB entries ✅")

# -------------------- MAIN CHAT UI --------------------
import tempfile
from PIL import Image

extra_docs = []  # Initialize to avoid NameError
st.title("💬 Multimodal RAG Assistant")

# ---- Input Section (Text or Live Audio) ----
st.subheader("Query Input")

# Text query
query = st.text_area("Ask a question across all data formats...")

# Live Audio input
st.markdown("🎙️ **Or record a voice query:**")
audio_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=False,
    use_container_width=True,
    key="live_audio"
)

# ---- Clear Audio Button ----
if st.button("Clear Audio", key="clear_audio"):
    audio_data = None
    query = ""
    st.success("Audio input cleared!")

# Convert recorded audio to text if present
if audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data["bytes"])
        temp_audio_path = temp_audio.name

    try:
        query = transcribe_audio(temp_audio_path)
        st.success(f"🎧 Transcribed Audio Query: {query}")
    except Exception as e:
        st.error(f"Audio transcription failed: {e}")
    finally:
        os.unlink(temp_audio_path)

# ---- Unified Ask Button ----
if st.button("Ask", key="ask_query"):
    if not query.strip():
        st.warning("Please provide a text or voice query before asking.")
    else:
        # -------------------- RETRIEVE DOCUMENTS --------------------
        result = query_docs(query)
        docs = result["documents"][0]
        ids = result["ids"][0]

        # Include extra_docs if any
        if extra_docs:
            docs.extend([d.page_content for d in extra_docs])
            ids.extend([f"query_extra_{i}" for i in range(len(extra_docs))])

        # -------------------- RERANK --------------------
        context, citations = rerank(query, docs, ids)

        # -------------------- LLM RESPONSE --------------------
        st.subheader("Answer:")
        st.write_stream(call_llm(context, query))

        # -------------------- CITATIONS (TEXT, AUDIO, IMAGES) --------------------
        st.subheader("📜 Citations")
        collection = get_collection()
        for num, cid in citations:
            if str(cid).startswith("query_extra"):
                idx = int(cid.split("_")[-1])
                meta = extra_docs[idx].metadata
                paragraph_text = extra_docs[idx].page_content
            else:
                result = collection.get(ids=[cid])
                paragraph_text = result["documents"][0]
                meta = result["metadatas"][0]

            pdf_name = meta.get("source_file", "Unknown")
            page_no = meta.get("page_no", "-")
            doc_type = meta.get("type", "text")

            if doc_type == "audio":
                st.markdown(f"""
                    <div style="border:1px solid #aaa; border-radius:6px; padding:8px; margin-bottom:8px;">
                        <div style="font-weight:bold; background-color:#e0e0e0; padding:4px 6px; border-radius:4px;">
                            {pdf_name} - Audio
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if meta.get("audio_path"):
                    st.audio(meta.get("audio_path"), format="audio/mp3")

            elif doc_type == "image":
                img_path = meta.get("image_name")
                if img_path and os.path.exists(img_path):
                    st.image(
                        img_path,
                        width=300,
                        caption=f"{pdf_name} - Page {page_no}",
                        use_container_width=False
                    )
            else:  # regular text/table
                st.markdown(f"""
                    <div style="border:1px solid #aaa; border-radius:6px; padding:8px; margin-bottom:8px;">
                        <div style="font-weight:bold; background-color:#e0e0e0; padding:4px 6px; border-radius:4px;">
                            {pdf_name} - Page {page_no}
                        </div>
                        <div style="margin-top:4px;">{paragraph_text}</div>
                    </div>
                """, unsafe_allow_html=True)
