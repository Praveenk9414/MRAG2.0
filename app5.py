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
    # sort by created desc
    sessions = sorted(sessions, key=lambda s: s["created"], reverse=True)
    return sessions

def create_session(title: str | None = None) -> str:
    sid = uuid.uuid4().hex[:8]
    path = session_dir(sid)
    os.makedirs(path, exist_ok=True)
    os.makedirs(session_uploads_dir(sid), exist_ok=True)
    os.makedirs(session_chroma_path(sid), exist_ok=True)
    # init empty history
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
    """
    Return a chromadb.PersistentClient collection tied to the session path.
    Each session has its own Chroma persistent DB folder.
    """
    chroma_path = session_chroma_path(session_id)
    os.makedirs(chroma_path, exist_ok=True)
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=DEFAULT_EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=chroma_path)
    # unique collection name per session:
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

# Remove session-specific documents by filename
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
    # remove uploaded file
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
                # handle alpha/composite
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
        # CrossEncoder.rank returns list of dicts with corpus_id
        final_text = ""
        citations = []
        for i, r in enumerate(ranks, start=1):
            idx = r["corpus_id"]
            final_text += f"{docs[idx]} [{i}]\n\n"
            citations.append((i, ids[idx]))
        return final_text, citations
    except Exception as e:
        # fallback: simple top-k by original order
        final_text = ""
        citations = []
        for i, d in enumerate(docs[:top_k], start=1):
            final_text += f"{d} [{i}]\n\n"
            citations.append((i, ids[i - 1]))
        return final_text, citations

# ---------- LLM call (streaming) ----------
def call_llm_stream(context: str, question: str):
    """
    Generator that yields text chunks from Ollama chat streaming.
    """
    response = ollama.chat(
        model=DEFAULT_LLAMA_MODEL,
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
        # Ollama response item shape: chunk["done"], chunk["message"]["content"]
        if not chunk.get("done"):
            yield chunk["message"]["content"]
    # final chunk may be omitted by API; generator ends

# ---------- Streamlit UI & session management ----------
# initialize session info in st.session_state
if "active_session" not in st.session_state:
    # if no sessions exist, create one
    sessions_list = list_sessions()
    if sessions_list:
        st.session_state.active_session = sessions_list[0]["id"]
    else:
        st.session_state.active_session = create_session("Main Chat")

if "sessions_list_cache" not in st.session_state:
    st.session_state.sessions_list_cache = list_sessions()

# Sidebar: sessions list + new chat controls
with st.sidebar:
    st.title("💬 Chats (Sessions)")

    # New chat: ask for optional title
    new_chat_title = st.text_input("New Chat Title (optional)", key="new_chat_title")
    if st.button("🆕 Create New Chat"):
        sid = create_session(new_chat_title.strip() or None)
        st.session_state.active_session = sid
        st.session_state.sessions_list_cache = list_sessions()
        st.rerun()

    st.markdown("---")
    st.subheader("All Sessions")
    sessions = list_sessions()
    for s in sessions:
        # display title and created time
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
            # pick another existing session or create new
            sessions_after = list_sessions()
            if sessions_after:
                st.session_state.active_session = sessions_after[0]["id"]
            else:
                st.session_state.active_session = create_session("Main Chat")
            st.rerun()

    if st.button("🗑️ Delete All Sessions (Reset App)"):
        # careful destructive action
        for s in list_sessions():
            delete_session(s["id"])
        # recreate base
        st.session_state.sessions_list_cache = list_sessions()
        st.session_state.active_session = create_session("Main Chat")
        st.rerun()

# Main layout
st.header("📚 Multimodal RAG — Session Isolated Chats")

active = st.session_state.active_session
st.subheader(f"Active Session: {load_session_title(active) or active}")

col1, col2 = st.columns([1, 2])

# Left column: upload and session-local resources
with col1:
    st.markdown("### 📂 Upload & Index (This session only)")
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

# Process uploads (in left column)
if process_btn:
    try:
        os.makedirs(session_uploads_dir(active), exist_ok=True)
        # PDF/DOC processing
        if uploaded_pdf:
            file_path = os.path.join(session_uploads_dir(active), uploaded_pdf.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.read())

            # Extract text & chunk using your helpers
            text_docs = extract_text_and_tables(file_path)
            text_chunks = chunk_documents(text_docs)
            # add to session chroma
            add_docs_to_session(active, text_chunks, uploaded_pdf.name)

            # extract images and context and index them
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

        # Audio processing
        if uploaded_audio:
            file_path = os.path.join(session_uploads_dir(active), uploaded_audio.name)
            uploaded_audio.seek(0)
            with open(file_path, "wb") as f:
                f.write(uploaded_audio.read())

            # Transcribe + index
            text = transcribe_audio(file_path)
            audio_doc = Document(
                page_content=text,
                metadata={"type": "audio", "source_file": uploaded_audio.name, "audio_path": file_path},
            )
            add_docs_to_session(active, [audio_doc], uploaded_audio.name)
            st.success(f"Transcribed & indexed audio for session {active}")

    except Exception as e:
        st.error(f"Processing failed: {e}\n{traceback.format_exc()}")

# Right column: chat area (isolated per session)
with col2:
    st.markdown("### 💬 Chat")
    # load session history (list of QA objects)
    history = load_history(active)

    # 'New' chat within session means a new conversation in the same session - we interpret as clearing current UI but keeping saved history
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = []  # each item: {"role": "user"/"assistant", "text": "..."}
    # if user switches sessions, ensure current conversation is reset (so UI empty)
    if st.session_state.get("last_active_session") != active:
        st.session_state.current_conversation = []
        st.session_state.last_active_session = active

    # Chat input and mic recorder
    query_text = st.text_area("Ask a question across this session's data...", key=f"query_{active}", height=120)
    st.markdown("🎙️ Record voice query (optional)")
    audio_blob = mic_recorder(start_prompt="Start", stop_prompt="Stop", key=f"mic_{active}")

    # If audio recorded, transcribe and fill into text area
    if audio_blob:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_blob["bytes"])
            tmp_path = tmp.name
        try:
            transcribed = transcribe_audio(tmp_path)
            # overwrite query_text and also put into session state text area
            st.session_state[f"query_{active}"] = transcribed
            st.success(f"Transcribed: {transcribed}")
            query_text = transcribed
        except Exception as e:
            st.error(f"Transcription failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Ask button
    if st.button("Ask (Search + Generate)", key=f"ask_{active}"):
        if not (query_text and query_text.strip()):
            st.warning("Please enter a query or record audio.")
        else:
            # RETRIEVE
            result = query_docs_for_session(active, query_text, n=10)
            docs = result.get("documents", [])[0] if result.get("documents") else []
            ids = result.get("ids", [])[0] if result.get("ids") else []

            # build context using rerank
            context, citations = rerank_local(query_text, docs, ids, top_k=5)

            # stream LLM output into placeholder
            st.subheader("Answer")
            placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in call_llm_stream(context, query_text):
                    full_answer += chunk
                    placeholder.markdown(full_answer)
            except Exception as e:
                st.error(f"LLM call failed: {e}\n{traceback.format_exc()}")

            # Save to session history (append)
            history_entry = {
                "id": uuid.uuid4().hex[:8],
                "question": query_text,
                "answer": full_answer,
                "citations": citations,
            }
            history.append(history_entry)
            save_history(active, history)
            st.success("Saved this Q&A to session history.")

            # Show citations below
            st.subheader("Citations / Sources")
            coll = get_collection(active)
            for (num, cid) in citations:
                try:
                    res = coll.get(ids=[cid])
                    paragraph_text = res["documents"][0]
                    meta = res["metadatas"][0]
                except Exception:
                    paragraph_text = "Could not load"
                    meta = {}
                doc_type = meta.get("type", "text")
                fname = meta.get("source_file", "Unknown")
                page_no = meta.get("page_no", "-")

                if doc_type == "audio":
                    st.markdown(f"**[{num}] Audio — {fname}**")
                    if meta.get("audio_path"):
                        st.audio(meta.get("audio_path"))
                elif doc_type == "image":
                    st.markdown(f"**[{num}] Image — {fname} (page {page_no})**")
                    img_path = meta.get("image_name")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=300)
                else:
                    st.markdown(f"**[{num}] {fname} — page {page_no}**")
                    st.write(paragraph_text)

    # Show session history (list entries) but not preload into input
    st.markdown("---")
    st.markdown("#### Session History")
    hist = load_history(active)
    for entry in reversed(hist[-20:]):  # show last 20
        st.markdown(f"**Q:** {entry['question']}")
        if st.button(f"Show A — {entry['id']}", key=f"show_{entry['id']}"):
            st.markdown(f"**A:** {entry['answer']}")
            # show citations if any
            if entry.get("citations"):
                st.markdown("**Citations:**")
                coll = get_collection(active)
                for (num, cid) in entry["citations"]:
                    try:
                        res = coll.get(ids=[cid])
                        paragraph_text = res["documents"][0]
                        meta = res["metadatas"][0]
                    except Exception:
                        paragraph_text = "Could not load"
                        meta = {}
                    st.markdown(f"- [{num}] {meta.get('source_file','Unknown')} (page {meta.get('page_no','-')})")
        if st.button(f"Load Q into input — {entry['id']}", key=f"load_{entry['id']}"):
            st.session_state[f"query_{active}"] = entry["question"]
            st.rerun()

# footer
st.markdown("---")
st.caption("Each session stores its own uploads, indices and chat history under ./sessions/<session_id>/ — isolated from other sessions.")
