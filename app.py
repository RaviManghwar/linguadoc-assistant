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
    layout="centered",
)

# ── Load API Key from Streamlit Secrets ──
# On Streamlit Cloud this comes from the app's "Secrets" settings, not os.environ.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
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
    print(f"Downloaded: {save_path}")


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


# ── Cache the vectorstore so it's built ONCE per app instance, not on every rerun ──
@st.cache_resource(show_spinner="Setting up knowledge base — this happens once...")
def build_vectorstore():
    os.makedirs("docs", exist_ok=True)
    all_documents = []
    for file_id, filename in GDRIVE_FILES:
        save_path = f"docs/{filename}"
        if not os.path.exists(save_path):
            download_from_gdrive(file_id, save_path)
        docs = load_pdf_as_documents(save_path)
        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


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


# ── Custom CSS (same visual theme as your Gradio app) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.hero-header {
    text-align: center;
    padding: 2.5rem 2rem 1.5rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
}
.hero-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.2 !important;
}
.hero-header p {
    color: rgba(255,255,255,0.6) !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
}
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,210,0,0.25);
    border-radius: 50px;
    padding: 0.4rem 1.1rem;
    color: rgba(255,255,255,0.75);
    font-size: 0.82rem;
    display: inline-block;
}
.stat-pill span {
    color: #ffd200;
    font-weight: 600;
}
.footer-note {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.78rem;
    margin-top: 1.2rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="hero-header">
    <h1>📚 LinguaDoc Assistant</h1>
    <p>Ask anything about your documents — powered by Groq · FAISS · LLaMA 3.3</p>
</div>
<div class="stats-bar">
    <div class="stat-pill">📄 Document: <span>Python for Linguists</span></div>
    <div class="stat-pill">🧠 Model: <span>LLaMA 3.3 · 70B</span></div>
    <div class="stat-pill">⚡ Vector DB: <span>FAISS</span></div>
</div>
""", unsafe_allow_html=True)

# ── Build (or fetch cached) vectorstore ──
vectorstore = build_vectorstore()

# ── Chat state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Example prompts ──
examples = [
    "What is this book about?",
    "How do you write a for loop in Python?",
    "What does the book say about text processing?",
    "How can Python be used for linguistic analysis?",
    "What is tokenization?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": ex})
        answer = ask_question(ex, vectorstore)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Render chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(prompt, vectorstore)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ──
st.markdown("""
<div class="footer-note">
    Built with Streamlit · Groq · FAISS · LangChain
</div>
""", unsafe_allow_html=True)
