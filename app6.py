# app_multisession_rag.py
import os
import uuid
import json
import shutil
import tempfile
import traceback

import streamlit as st
import chromadb
import ollama
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_core.documents import Document

# Your local extractor helpers (must exist)
from extractors.text_extractor import extract_text_and_tables, chunk_documents
from extractors.audio_transcriber import transcribe_audio
from streamlit_mic_recorder import mic_recorder

# ---------- Configuration ----------
BASE_SESSIONS_DIR = "./sessions"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_LLAMA_MODEL = "koesn/llama3-8b-instruct:latest"
CHROMA_COLLECTION_NAME_PREFIX = "multi_rag"  # collection per session will be f"{prefix}_{session_id}"
os.makedirs(BASE_SESSIONS_DIR, exist_ok=True)

st.set_page_config(page_title="📚 Multisession Multimodal RAG", layout="wide")

# ---------- Utility: session file handling ----------
def session_dir(session_id: str) -> str:
    return os.path.join(BASE_SESSIONS_DIR, session_id)

def session_uploads_dir(session_id: str) -> str:
    return os.path.join(session_dir(session_id), "uploads")

def session_chroma_path(session_id: str) -> str:
    return os.path.join(session_dir(session_id), "chroma")

def session_history_path(session_id: str) -> str:
    return os.path.join(session_dir(session_id), "history.json")

def list_sessions() -> list[dict]:
    """Return list of {id, title, created_at} for folders under sessions dir."""
    sessions = []
    for name in os.listdir(BASE_SESSIONS_DIR):
        path = os.path.join(BASE_SESSIONS_DIR, name)
        if os.path.isdir(path):
            created = os.path.getctime(path)
            title = load_session_title(name) or name
            sessions.append({"id": name, "title": title, "created": created})
    sessions = sorted(sessions, key=lambda s: s["created"], reverse=True)
    return sessions

def create_session(title: str | None = None) -> str:
    sid = uuid.uuid4().hex[:8]
    path = session_dir(sid)
    os.makedirs(path, exist_ok=True)
    os.makedirs(session_uploads_dir(sid), exist_ok=True)
    os.makedirs(session_chroma_path(sid), exist_ok=True)
    save_history(sid, [])
    if title:
        save_session_title(sid, title)
    return sid

def delete_session(session_id: str):
    path = session_dir(session_id)
    if os.path.exists(path):
        shutil.rmtree(path)

def save_history(session_id: str, history: list):
    p = session_history_path(session_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(session_id: str) -> list:
    p = session_history_path(session_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_session_title(session_id: str, title: str):
    p = os.path.join(session_dir(session_id), "meta.json")
    meta = {"title": title}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_session_title(session_id: str) -> str | None:
    p = os.path.join(session_dir(session_id), "meta.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
                return meta.get("title")
        except Exception:
            return None
    return None

# ---------- Chroma collection per session ----------
def get_collection(session_id: str):
    chroma_path = session_chroma_path(session_id)
    os.makedirs(chroma_path, exist_ok=True)
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=DEFAULT_EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=chroma_path)
    coll_name = f"{CHROMA_COLLECTION_NAME_PREFIX}_{session_id}"
    return client.get_or_create_collection(name=coll_name, embedding_function=ollama_ef)

# ---------- Document ingestion helpers ----------
def add_docs_to_session(session_id: str, docs: list[Document], file_name: str):
    if not docs:
        return
    collection = get_collection(session_id)
    documents = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [f"{file_name}_{i}" for i in range(len(docs))]
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

def remove_file_data_from_session(session_id: str, file_name: str):
    coll = get_collection(session_id)
    results = coll.query(query_texts=[""], n_results=10000)
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
        coll.delete(ids=ids_to_delete)
    uploads_dir = session_uploads_dir(session_id)
    fpath = os.path.join(uploads_dir, file_name)
    if os.path.exists(fpath):
        os.remove(fpath)

# ---------- Image extraction (per-session) ----------
def extract_images_with_context(pdf_path: str, session_id: str):
    uploads_dir = session_uploads_dir(session_id)
    os.makedirs(uploads_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_context_data = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_no, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")
        text_blocks = sorted(text_blocks, key=lambda b: b[1])

        for img_no, img in enumerate(images, start=1):
            try:
                xref = img[0]
                smask = img[1]
                bbox = img[1:5]
                img_bottom = bbox[3]

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

                image_context_data.append(
                    {
                        "page_no": page_no,
                        "image_path": img_path,
                        "bbox": bbox,
                        "context": context_text if context_text else "No nearby text found.",
                    }
                )
                main_pix = None
            except Exception as e:
                print(f"Error processing image on page {page_no}: {e}")
                continue

    json_path = os.path.join(session_uploads_dir(session_id), f"{pdf_name}_images.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_context_data, f, indent=2, ensure_ascii=False)
    return json_path, image_context_data

# ---------- Retrieval & rerank ----------
def query_docs_for_session(session_id: str, query: str, n=10):
    coll = get_collection(session_id)
    return coll.query(query_texts=[query], n_results=n)

def rerank_local(prompt: str, docs: list[str], ids: list[str], top_k: int = 3):
    if not docs:
        return "", []
    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = model.rank(prompt, docs, top_k=min(top_k, len(docs)))
        final_text = ""
        citations = []
        for i, r in enumerate(ranks, start=1):
            idx = r["corpus_id"]
            final_text += f"{docs[idx]} [{i}]\n\n"
            citations.append((i, ids[idx]))
        return final_text, citations
    except Exception as e:
        final_text = ""
        citations = []
        for i, d in enumerate(docs[:top_k], start=1):
            final_text += f"{d} [{i}]\n\n"
            citations.append((i, ids[i - 1]))
        return final_text, citations

# ---------- LLM call (streaming) ----------
def call_llm_stream(context: str, question: str, model_name: str = DEFAULT_LLAMA_MODEL):
    response = ollama.chat(
        model=model_name,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer clearly using only the context provided. Include numbered citations referring to context chunks.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    )
    for chunk in response:
        if not chunk.get("done"):
            yield chunk["message"]["content"]

# ---------- Streamlit UI & session management ----------
if "active_session" not in st.session_state:
    sessions_list = list_sessions()
    if sessions_list:
        st.session_state.active_session = sessions_list[0]["id"]
    else:
        st.session_state.active_session = create_session("Main Chat")

if "sessions_list_cache" not in st.session_state:
    st.session_state.sessions_list_cache = list_sessions()

with st.sidebar:

    new_chat_title = st.text_input("New Chat Title (optional)", key="new_chat_title")
    if st.button("🆕 Create New Chat"):
        sid = create_session(new_chat_title.strip() or None)
        st.session_state.active_session = sid
        st.session_state.sessions_list_cache = list_sessions()
        st.rerun()

    st.markdown("---")
    st.subheader("Workspaces")
    sessions = list_sessions()
    for s in sessions:
        title = s["title"] or s["id"]
        display = f"🔸 {title} ({s['id']})"
        if st.button(display, key=f"open_{s['id']}"):
            st.session_state.active_session = s["id"]
            st.rerun()

    st.markdown("---")
    st.subheader("Manage Session")
    if st.button("🗑️ Delete Current Session"):
        cur = st.session_state.active_session
        if cur:
            delete_session(cur)
            st.session_state.sessions_list_cache = list_sessions()
            sessions_after = list_sessions()
            if sessions_after:
                st.session_state.active_session = sessions_after[0]["id"]
            else:
                st.session_state.active_session = create_session("Main Chat")
            st.rerun()

    if st.button("🗑️ Delete All Sessions (Reset App)"):
        for s in list_sessions():
            delete_session(s["id"])
        st.session_state.sessions_list_cache = list_sessions()
        st.session_state.active_session = create_session("Main Chat")
        st.rerun()

st.header("💬 Recall")
active = st.session_state.active_session
st.subheader(f"Active Session: {load_session_title(active) or active}")

col1, col2 = st.columns([1, 2])

# Left column
with col1:
    st.markdown("### 📂 Ingestion")
    uploaded_pdf = st.file_uploader("Upload PDF / DOCX", type=["pdf", "docx"], key=f"up_pdf_{active}")
    uploaded_audio = st.file_uploader("Upload Audio (wav/mp3)", type=["wav", "mp3"], key=f"up_audio_{active}")
    process_btn = st.button("⚡ Process Uploads", key=f"process_{active}")
    remove_file = st.text_input("Remove file by name (uploads folder)", key=f"remove_file_{active}")
    if st.button("Remove File from Session", key=f"removefile_btn_{active}"):
        if remove_file.strip():
            remove_file_data_from_session(active, remove_file.strip())
            st.success(f"Removed file {remove_file.strip()} from session {active}")
        else:
            st.warning("Enter the file name (with extension) to remove from this session.")

    st.markdown("---")
    st.markdown("### 🔎 Session Tools")
    if st.button("Show Session Files"):
        up_dir = session_uploads_dir(active)
        files = os.listdir(up_dir) if os.path.exists(up_dir) else []
        st.write(files)

    if st.button("Show Session History"):
        h = load_history(active)
        st.write(h)

# Process uploads
if process_btn:
    try:
        os.makedirs(session_uploads_dir(active), exist_ok=True)
        if uploaded_pdf:
            file_path = os.path.join(session_uploads_dir(active), uploaded_pdf.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.read())
            text_docs = extract_text_and_tables(file_path)
            text_chunks = chunk_documents(text_docs)
            add_docs_to_session(active, text_chunks, uploaded_pdf.name)
            json_path, image_context_data = extract_images_with_context(file_path, active)
            image_docs = []
            for img in image_context_data:
                doc = Document(
                    page_content=img["context"],
                    metadata={
                        "type": "image",
                        "image_name": img["image_path"],
                        "page_no": img["page_no"],
                        "source_file": uploaded_pdf.name,
                    },
                )
                image_docs.append(doc)
            add_docs_to_session(active, image_docs, uploaded_pdf.name)
            st.success(f"Processed {len(text_chunks)} text chunks & {len(image_docs)} images for session {active}")
            st.info(f"Image metadata JSON saved at: {json_path}")
        if uploaded_audio:
            file_path = os.path.join(session_uploads_dir(active), uploaded_audio.name)
            uploaded_audio.seek(0)
            with open(file_path, "wb") as f:
                f.write(uploaded_audio.read())
            text = transcribe_audio(file_path)
            audio_doc = Document(
                page_content=text,
                metadata={"type": "audio", "source_file": uploaded_audio.name, "audio_path": file_path},
            )
            add_docs_to_session(active, [audio_doc], uploaded_audio.name)
            st.success(f"Transcribed & indexed audio for session {active}")
    except Exception as e:
        st.error(f"Processing failed: {e}\n{traceback.format_exc()}")

# Right column
with col2:
    st.markdown("### ⚙️ Select LLM for this session")
    if "session_llm" not in st.session_state:
        st.session_state["session_llm"] = {}
    llm_options = [
        "koesn/llama3-8b-instruct:latest",
        "koesn/llama2-7b-instruct:latest",
        "nomic/llama2-13b-chat:latest",
    ]
    selected_llm = st.selectbox(
        "Choose LLM for this session",
        llm_options,
        index=llm_options.index(
            st.session_state["session_llm"].get(active, DEFAULT_LLAMA_MODEL)
        ),
    )
    st.session_state["session_llm"][active] = selected_llm

    # Query input
    query_key = f"query_area_{active}"
    stage_key = f"{query_key}_stage"

    # Initialize
    if query_key not in st.session_state:
        st.session_state[query_key] = ""

    # Apply staged value if exists
    if stage_key in st.session_state:
        st.session_state[query_key] = st.session_state.pop(stage_key)

    query_text = st.text_area(
        "Ask a question across this session's data...",
        value=st.session_state[query_key],
        key=query_key,
        height=120,
    )
    st.session_state[f"query_{active}"] = query_text

    # 🎙️ Voice query
    st.markdown("🎙️ Record voice query (optional)")
    audio_blob = mic_recorder(
        start_prompt="Start", stop_prompt="Stop", key=f"mic_{active}"
    )
    if audio_blob:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_blob["bytes"])
            tmp_path = tmp.name
        try:
            transcribed = transcribe_audio(tmp_path)
            st.session_state[stage_key] = transcribed
            st.rerun()
        except Exception as e:
            st.error(f"Transcription failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Chat history and ask button
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []
    if st.session_state.get("last_active_session") != active:
        st.session_state.current_conversation = []
        st.session_state.last_active_session = active

    if st.button("Ask (Search + Generate)", key=f"ask_{active}"):
        if not (query_text and query_text.strip()):
            st.warning("Please enter a query or record audio.")
        else:
            # -------------------- RETRIEVE DOCUMENTS --------------------
            result = query_docs_for_session(active, query_text, n=10)
            docs = result.get("documents", [])[0] if result.get("documents") else []
            ids = result.get("ids", [])[0] if result.get("ids") else []

            # -------------------- RERANK --------------------
            reranked_text, citations = rerank_local(query_text, docs, ids)

            # -------------------- LLM RESPONSE --------------------
            answer_chunks = []
            response_placeholder = st.empty()
            for chunk in call_llm_stream(
                    reranked_text, query_text, model_name=st.session_state["session_llm"][active]
            ):
                answer_chunks.append(chunk)
                response_placeholder.markdown("".join(answer_chunks) + "▌")

            final_answer = "".join(answer_chunks)
            response_placeholder.markdown(final_answer)

            # -------------------- SAVE TO HISTORY --------------------
            history = load_history(active)
            history.append({"question": query_text, "answer": final_answer})
            save_history(active, history)
            st.session_state.current_conversation.append(
                {"question": query_text, "answer": final_answer}
            )

            # -------------------- DISPLAY CITATIONS --------------------
            st.subheader("📜 Citations")
            collection = get_collection(active)
            for num, cid in citations:
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
                    if paragraph_text:
                        st.markdown(f"**Context:** {paragraph_text}")

                else:  # text/table
                    st.markdown(f"""
                        <div style="border:1px solid #aaa; border-radius:6px; padding:8px; margin-bottom:8px;">
                            <div style="font-weight:bold; background-color:#e0e0e0; padding:4px 6px; border-radius:4px;">
                                {pdf_name} - Page {page_no}
                            </div>
                            <div style="margin-top:4px;">{paragraph_text}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # Display session conversation as collapsible
    st.markdown("### 📝 Conversation History")
    history_entries = load_history(active)[-10:]
    for i, entry in enumerate(history_entries):
        with st.expander(f"Q: {entry['question']}", expanded=False):
            st.markdown(f"**A:** {entry['answer']}")
            # Use a unique key with index + hashed question to avoid duplicate key errors
            load_key = f"loadq_{i}_{hash(entry['question'])}"
            if st.button("Load Question into input", key=load_key):
                st.session_state[stage_key] = entry["question"]
                st.rerun()
