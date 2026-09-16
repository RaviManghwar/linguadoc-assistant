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
#  THEME
# ══════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg_grad": "radial-gradient(circle at 15% 0%, #241b3f 0%, #17122b 45%, #0b0a17 100%)",
        "sidebar_grad": "linear-gradient(180deg, #14102a 0%, #0b0a17 100%)",
        "glass": "rgba(255,255,255,0.045)",
        "glass_border": "rgba(255,255,255,0.09)",
        "glass_border_strong": "rgba(255,255,255,0.14)",
        "text_main": "rgba(255,255,255,0.92)",
        "text_dim": "rgba(255,255,255,0.55)",
        "text_faint": "rgba(255,255,255,0.35)",
        "sidebar_text": "rgba(255,255,255,0.85)",
        "input_text": "#ffffff",
        "scrollbar": "rgba(255,255,255,0.15)",
        "assistant_bg": "linear-gradient(135deg, rgba(247,151,30,0.08), rgba(255,255,255,0.03))",
        "assistant_border": "rgba(255,210,0,0.18)",
        "user_bg": "rgba(255,255,255,0.03)",
        "shadow": "0 8px 32px rgba(0,0,0,0.25)",
        "toggle_icon": "🌙",
    },
    "light": {
        "bg_grad": "radial-gradient(circle at 15% 0%, #fff8ec 0%, #fbf1e0 45%, #f6e9d3 100%)",
        "sidebar_grad": "linear-gradient(180deg, #fffaf2 0%, #f8ecd8 100%)",
        "glass": "rgba(255,255,255,0.55)",
        "glass_border": "rgba(60,40,10,0.10)",
        "glass_border_strong": "rgba(60,40,10,0.16)",
        "text_main": "rgba(35,25,10,0.90)",
        "text_dim": "rgba(35,25,10,0.55)",
        "text_faint": "rgba(35,25,10,0.35)",
        "sidebar_text": "rgba(35,25,10,0.85)",
        "input_text": "#231a0a",
        "scrollbar": "rgba(60,40,10,0.18)",
        "assistant_bg": "linear-gradient(135deg, rgba(247,151,30,0.14), rgba(255,255,255,0.5))",
        "assistant_border": "rgba(200,130,0,0.30)",
        "user_bg": "rgba(255,255,255,0.6)",
        "shadow": "0 8px 28px rgba(120,90,40,0.12)",
        "toggle_icon": "☀️",
    },
}
T = THEMES[st.session_state.theme]

# ══════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --accent-1: #f7971e;
    --accent-2: #ffd200;
    --glass: {T["glass"]};
    --glass-border: {T["glass_border"]};
}}

.stApp {{
    background: {T["bg_grad"]} !important;
    font-family: 'DM Sans', sans-serif !important;
}}

#MainMenu, footer, header {{visibility: hidden;}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: {T["sidebar_grad"]} !important;
    border-right: 1px solid var(--glass-border);
}}
section[data-testid="stSidebar"] * {{ color: {T["sidebar_text"]} !important; }}
.sb-card {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}}
.sb-card .label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {T["text_dim"]} !important;
    margin-bottom: 0.25rem;
}}
.sb-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: var(--accent-1) !important;
}}

/* ---- Hero header ---- */
.hero-header {{
    text-align: center;
    padding: 2.6rem 2rem 1.8rem;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 22px;
    margin-bottom: 1.6rem;
    backdrop-filter: blur(14px);
    box-shadow: {T["shadow"]};
}}
.hero-header h1 {{
    font-family: 'Fraunces', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.15 !important;
}}
.hero-header p {{
    color: {T["text_dim"]} !important;
    font-size: 1.02rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
}}

/* ---- Buttons (universal fallback for any Streamlit button testid) ---- */
.stApp button {{
    font-family: 'DM Sans', sans-serif !important;
}}
.stApp button[kind="secondary"],
.stApp [data-testid*="BaseButton" i] {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: {T["text_main"]} !important;
    transition: all 0.2s ease !important;
}}
.stApp button[kind="secondary"]:hover,
.stApp [data-testid*="BaseButton" i]:hover {{
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(247,151,30,0.10) !important;
}}

/* ---- Sidebar buttons (theme toggle, clear conversation) ---- */
section[data-testid="stSidebar"] button {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: {T["sidebar_text"]} !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] button:hover {{
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
}}

/* ---- Example question chips ---- */
div[data-testid*="Horizontal" i] button,
div[data-testid*="column" i] button {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: {T["text_main"]} !important;
    border-radius: 999px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    white-space: normal !important;
    height: 100% !important;
}}
div[data-testid*="Horizontal" i] button:hover,
div[data-testid*="column" i] button:hover {{
    border-color: var(--accent-1) !important;
    color: var(--accent-1) !important;
    background: rgba(247,151,30,0.10) !important;
    transform: translateY(-1px);
}}

/* ---- Chat messages ---- */
div[data-testid="stChatMessage"] {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.85rem;
    animation: fadeIn 0.25s ease;
}}
div[data-testid="stChatMessageContent"] p {{
    color: {T["text_main"]} !important;
    font-size: 0.96rem;
    line-height: 1.6;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
    background: {T["assistant_bg"]};
    border-color: {T["assistant_border"]};
    margin-right: 10%;
}}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: {T["user_bg"]};
    border-color: {T["glass_border_strong"]};
    margin-left: 10%;
}}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
}}

/* ---- Source citation caption ---- */
.source-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {T["text_faint"]};
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    margin-top: 0.4rem;
}}

/* ---- Chat input box ---- */
[data-testid*="ChatInput" i] {{
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}}
[data-testid*="ChatInput" i]:focus-within {{
    border-color: rgba(247,151,30,0.5) !important;
    box-shadow: 0 0 0 3px rgba(247,151,30,0.12) !important;
}}
[data-testid*="ChatInput" i] textarea {{
    color: {T["input_text"]} !important;
    background: transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid*="ChatInput" i] textarea::placeholder {{
    color: {T["text_faint"]} !important;
}}
[data-testid*="ChatInput" i] textarea:focus {{
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid*="ChatInput" i] button {{
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    border: none !important;
}}
[data-testid*="ChatInput" i] button svg {{
    fill: #1a1a2e !important;
}}

/* ---- Footer ---- */
.footer-note {{
    text-align: center;
    color: {T["text_faint"]};
    font-size: 0.76rem;
    margin-top: 1.4rem;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}}

/* ---- Scrollbar ---- */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {T["scrollbar"]}; border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-1); }}
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
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown("### 📚 LinguaDoc")
    with top_r:
        if st.button(T["toggle_icon"], key="theme_toggle", help="Switch theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

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
        f'<p style="color:{T["text_faint"]}; font-size:0.85rem; '
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
