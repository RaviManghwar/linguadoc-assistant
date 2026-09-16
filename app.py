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
    """Returns (answer_text, sorted_list_of_page_numbers_used)."""
    if not message.strip():
        return "Please type a question.", []
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(message)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        pages = sorted({
            doc.metadata.get("page")
            for doc in relevant_docs
            if doc.metadata.get("page") is not None
        })
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
        return response.choices[0].message.content, pages
    except Exception as e:
        return f"Error: {str(e)}", []


# ══════════════════════════════════════════════════════════════
#  STYLING — one locked-in dark theme, no toggle, no light-mode edge cases
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent-1: #f7971e;
    --accent-2: #ffd200;
    --glass: rgba(255,255,255,0.05);
    --glass-hover: rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.10);
    --glass-border-strong: rgba(255,255,255,0.16);
    --text-main: rgba(255,255,255,0.94);
    --text-dim: rgba(255,255,255,0.58);
    --text-faint: rgba(255,255,255,0.36);
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #241b3f 0%, #17122b 45%, #0b0a17 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    color-scheme: dark;
}

.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
    color: var(--text-main);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="collapsedControl"] {
    color: var(--accent-1) !important;
}
[data-testid="collapsedControl"] svg {
    fill: var(--accent-1) !important;
}

[data-testid="stAlert"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
}
[data-testid="stAlert"] p {
    color: var(--text-main) !important;
}

/* ══════════════════ SIDEBAR ══════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16112b 0%, #0c0a18 100%) !important;
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0 1.4rem 0;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 1.2rem;
}
.sb-brand .icon {
    font-size: 1.7rem;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(247,151,30,0.18), rgba(255,210,0,0.10));
    border: 1px solid rgba(255,210,0,0.25);
    border-radius: 12px;
}
.sb-brand .title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--text-main);
    line-height: 1.1;
}
.sb-brand .subtitle {
    font-size: 0.72rem;
    color: var(--text-faint);
    letter-spacing: 0.04em;
}

.sb-section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-faint) !important;
    margin: 0.85rem 0 0.4rem 0.1rem;
    font-weight: 600;
}

.sb-list {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0 0.9rem;
    margin-bottom: 0.5rem;
}
.sb-list-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--glass-border);
    gap: 0.6rem;
}
.sb-list-row:last-child { border-bottom: none; }
.sb-list-row .label {
    font-size: 0.72rem;
    color: var(--text-dim) !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    white-space: nowrap;
}
.sb-list-row .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-2) !important;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
}

.sb-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    transition: all 0.2s ease;
}
.sb-card:hover {
    border-color: var(--glass-border-strong);
    background: var(--glass-hover);
}
.sb-card .emoji {
    font-size: 1.15rem;
    width: 32px;
    text-align: center;
    flex-shrink: 0;
}
.sb-card .text-wrap { min-width: 0; }
.sb-card .label {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-faint) !important;
    margin-bottom: 0.15rem;
}
.sb-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: var(--accent-2) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.sb-stats-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.sb-stat {
    flex: 1;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.55rem 0.5rem;
    text-align: center;
}
.sb-stat .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 500;
    color: var(--accent-2) !important;
    display: block;
}
.sb-stat .lbl {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-faint) !important;
}

/* ══════════════════ BUTTONS ══════════════════ */
.stApp button {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp button[kind="secondary"],
.stApp [data-testid*="BaseButton" i] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-main) !important;
    transition: all 0.2s ease !important;
}
.stApp button[kind="secondary"]:hover,
.stApp [data-testid*="BaseButton" i]:hover {
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(247,151,30,0.10) !important;
}
section[data-testid="stSidebar"] button {
    border-radius: 10px !important;
}

/* Example question chips */
div[data-testid*="Horizontal" i] button,
div[data-testid*="column" i] button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-main) !important;
    border-radius: 999px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    white-space: normal !important;
    height: 100% !important;
}
div[data-testid*="Horizontal" i] button:hover,
div[data-testid*="column" i] button:hover {
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(247,151,30,0.10) !important;
    transform: translateY(-1px);
}

/* ══════════════════ HERO HEADER ══════════════════ */
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

/* ══════════════════ CHAT MESSAGES ══════════════════ */
div[data-testid="stChatMessage"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.85rem;
    animation: fadeIn 0.25s ease;
}
div[data-testid="stChatMessageContent"] p {
    color: var(--text-main) !important;
    font-size: 0.96rem;
    line-height: 1.6;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: linear-gradient(135deg, rgba(247,151,30,0.08), rgba(255,255,255,0.03));
    border-color: rgba(255,210,0,0.18);
    margin-right: 10%;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(255,255,255,0.03);
    border-color: var(--glass-border-strong);
    margin-left: 10%;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
}

.source-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-faint);
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    margin-top: 0.4rem;
}

/* Streamlit wraps the chat input in its own bottom container —
   theme that too, or it shows as a flat black bar regardless of
   how the input box itself is styled. */
[data-testid*="Bottom" i] {
    background: linear-gradient(180deg, rgba(11,10,23,0) 0%, rgba(11,10,23,0.92) 55%, #0b0a17 100%) !important;
}

/* ══════════════════ CHAT INPUT ══════════════════ */
[data-testid*="ChatInput" i] {
    background: linear-gradient(#211a3a, #211a3a) padding-box,
                linear-gradient(135deg, rgba(247,151,30,0.55), rgba(255,210,0,0.35)) border-box !important;
    border: 1.5px solid transparent !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35) !important;
    padding: 0.15rem 0.3rem !important;
}
/* Strip every nested wrapper's own border/background/radius so only
   the single outer border above is visible — Streamlit nests an
   inner pill-shaped div around the textarea that otherwise shows
   through as a second, ugly border. */
[data-testid*="ChatInput" i] * {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 0 !important;
}
[data-testid*="ChatInput" i]:focus-within {
    background: linear-gradient(#241d40, #241d40) padding-box,
                linear-gradient(135deg, var(--accent-1), var(--accent-2)) border-box !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.4), 0 0 0 3px rgba(247,151,30,0.15) !important;
}
[data-testid*="ChatInput" i] textarea {
    color: #ffffff !important;
    background: transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid*="ChatInput" i] textarea::placeholder {
    color: rgba(255,255,255,0.4) !important;
}
/* Re-apply the send button's own look on top of the blanket reset above */
[data-testid*="ChatInput" i] button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    border-radius: 50% !important;
}
[data-testid*="ChatInput" i] button svg {
    fill: #1a1a2e !important;
}

/* ══════════════════ FOOTER ══════════════════ */
.footer-note {
    text-align: center;
    color: var(--text-faint);
    font-size: 0.76rem;
    margin-top: 1.4rem;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}

/* ══════════════════ SCROLLBAR ══════════════════ */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-1); }
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
    st.markdown("""
    <div class="sb-brand">
        <div class="icon">📚</div>
        <div>
            <div class="title">LinguaDoc</div>
            <div class="subtitle">Document Q&A Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Knowledge base</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-card"><div class="emoji">📄</div>'
        '<div class="text-wrap"><div class="label">Document</div>'
        '<div class="value">Python for Linguists</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sb-stats-row">
            <div class="sb-stat"><span class="num">{total_pages}</span><span class="lbl">Pages</span></div>
            <div class="sb-stat"><span class="num">{total_chunks}</span><span class="lbl">Chunks</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section-label">Engine</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sb-list">
            <div class="sb-list-row"><span class="label">🧠 Model</span><span class="value">LLaMA 3.3 · 70B</span></div>
            <div class="sb-list-row"><span class="label">⚡ Vector DB</span><span class="value">FAISS</span></div>
            <div class="sb-list-row"><span class="label">🔗 Powered by</span><span class="value">Groq API</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_questions = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.markdown(
        f"""
        <div class="sb-list">
            <div class="sb-list-row"><span class="label">💬 Questions asked</span><span class="value">{n_questions}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🗑️  Clear conversation", use_container_width=True):
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


def render_message(role, content, pages=None):
    avatar = "🧑‍💻" if role == "user" else "📚"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if pages:
            page_str = ", ".join(f"p.{p}" for p in pages)
            st.markdown(f'<span class="source-tag">📖 Source: {page_str}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  EXAMPLE PROMPTS (only shown before the conversation starts)
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(
        '<p style="color:var(--text-faint); font-size:0.85rem; '
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
        st.session_state.messages.append({"role": "user", "content": clicked, "pages": None})
        with st.spinner("Thinking..."):
            answer, pages = ask_question(clicked, vectorstore)
        st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("pages"))

# ══════════════════════════════════════════════════════════════
#  CHAT INPUT
# ══════════════════════════════════════════════════════════════
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "pages": None})
    render_message("user", prompt)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Thinking..."):
            answer, pages = ask_question(prompt, vectorstore)
        st.markdown(answer)
        if pages:
            page_str = ", ".join(f"p.{p}" for p in pages)
            st.markdown(f'<span class="source-tag">📖 Source: {page_str}</span>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-note">
    Built with Streamlit · Groq · FAISS · LangChain
</div>
""", unsafe_allow_html=True)
