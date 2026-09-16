import os
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Page config (must be the first Streamlit command) ──
st.set_page_config(
    page_title="LinguaDoc Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load API Key from Streamlit Secrets ──
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if not GROQ_API_KEY:
    st.error(
        "🔑 GROQ_API_KEY is missing.\n\n"
        "Go to your app's **Settings → Secrets** on Streamlit Cloud and add:\n\n"
        "```toml\nGROQ_API_KEY = \"your-key-here\"\n```\n\n"
        "Then wait for the app to reboot."
    )
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Your Google Drive file IDs ──
GDRIVE_FILES = [
    ("1CqlwwJ9pWlLPyU2VJ3kCvuC5nPAE907H", "python_for_linguists.pdf"),
    # Add more files here as needed:
    # ("YOUR_NEXT_FILE_ID", "filename.pdf"),
]


def download_from_gdrive(file_id, save_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
    if confirm_token:
        url = url + f"&confirm={confirm_token}"
        response = session.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def load_pdf_as_documents(path):
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": i + 1}
            ))
    return docs


@st.cache_resource(show_spinner="Setting up knowledge base — this happens once...")
def build_vectorstore():
    os.makedirs("docs", exist_ok=True)
    all_documents = []
    total_pages = 0
    for file_id, filename in GDRIVE_FILES:
        save_path = f"docs/{filename}"
        if not os.path.exists(save_path):
            download_from_gdrive(file_id, save_path)
        docs = load_pdf_as_documents(save_path)
        all_documents.extend(docs)
        total_pages += len(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, total_pages, len(chunks)


def ask_question(message, vectorstore):
    if not message.strip():
        return "Please type a question."
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(message)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided document context.\n"
            "- Answer ONLY based on the context provided below.\n"
            "- If the answer is not in the context, say: "
            "'I could not find this information in the provided documents.'\n"
            "- Be clear, concise, and accurate."
        )
        user_prompt = f"Context from documents:\n{context}\n\nQuestion: {message}"
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-1: #0b0a17;
    --bg-2: #17122b;
    --bg-3: #241b3f;
    --accent-1: #f7971e;
    --accent-2: #ffd200;
    --glass: rgba(255,255,255,0.045);
    --glass-border: rgba(255,255,255,0.09);
    --text-dim: rgba(255,255,255,0.55);
}

.stApp {
    background: radial-gradient(circle at 15% 0%, var(--bg-3) 0%, var(--bg-2) 45%, var(--bg-1) 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default streamlit chrome clutter */
#MainMenu, footer, header {visibility: hidden;}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14102a 0%, #0b0a17 100%) !important;
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
.sb-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.sb-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim) !important;
    margin-bottom: 0.25rem;
}
.sb-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: var(--accent-2) !important;
}

/* ---- Hero header ---- */
.hero-header {
    text-align: center;
    padding: 2.6rem 2rem 1.8rem;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 22px;
    margin-bottom: 1.6rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.hero-header h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.15 !important;
}
.hero-header p {
    color: var(--text-dim) !important;
    font-size: 1.02rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
}

/* ---- Example question chips ---- */
div[data-testid="column"] .stButton button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: rgba(255,255,255,0.8) !important;
    border-radius: 999px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    height: 100% !important;
}
div[data-testid="column"] .stButton button:hover {
    border-color: var(--accent-2) !important;
    color: var(--accent-2) !important;
    background: rgba(255,210,0,0.08) !important;
    transform: translateY(-1px);
}

/* ---- Chat messages ---- */
div[data-testid="stChatMessage"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 0.3rem 0.6rem;
    margin-bottom: 0.7rem;
}
div[data-testid="stChatMessageContent"] p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 0.96rem;
    line-height: 1.55;
}

/* ---- Chat input box ---- */
div[data-testid="stChatInput"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
}
div[data-testid="stChatInput"] textarea {
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Footer ---- */
.footer-note {
    text-align: center;
    color: rgba(255,255,255,0.22);
    font-size: 0.76rem;
    margin-top: 1.4rem;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,210,0,0.4); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  BUILD KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════
vectorstore, total_pages, total_chunks = build_vectorstore()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📚 LinguaDoc")
    st.markdown(
        '<div class="sb-card"><div class="label">Document</div>'
        '<div class="value">Python for Linguists</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-card"><div class="label">Model</div>'
        '<div class="value">LLaMA 3.3 · 70B</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-card"><div class="label">Vector DB</div>'
        '<div class="value">FAISS</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sb-card"><div class="label">Indexed</div>'
        f'<div class="value">{total_pages} pages · {total_chunks} chunks</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>📚 LinguaDoc Assistant</h1>
    <p>Ask anything about your documents — powered by Groq · FAISS · LLaMA 3.3</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CHAT STATE
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

# ══════════════════════════════════════════════════════════════
#  EXAMPLE PROMPTS (only shown before the conversation starts)
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(
        '<p style="color:rgba(255,255,255,0.4); font-size:0.85rem; '
        'text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem;">'
        'Try asking</p>',
        unsafe_allow_html=True,
    )
    examples = [
        "What is this book about?",
        "How do you write a for loop in Python?",
        "What does the book say about text processing?",
        "How can Python be used for linguistic analysis?",
        "What is tokenization?",
    ]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            clicked = ex
    if clicked:
        st.session_state.messages.append({"role": "user", "content": clicked})
        with st.spinner("Thinking..."):
            answer = ask_question(clicked, vectorstore)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📚"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ══════════════════════════════════════════════════════════════
#  CHAT INPUT
# ══════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Thinking..."):
            answer = ask_question(prompt, vectorstore)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-note">
    Built with Streamlit · Groq · FAISS · LangChain
</div>
""", unsafe_allow_html=True)
