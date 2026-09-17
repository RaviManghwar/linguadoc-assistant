# 📚 LinguaDoc Assistant

A simple chatbot that answers questions about a PDF document. Ask it something, and it looks through the document, finds the right parts, and gives you an answer based on what's actually written there — not a guess.

Think of it like a smart search bar that talks back.

## What it does

- You type a question about the document (right now it's set up with *Python for Linguists*, but you can point it at any PDF).
- It searches the document for the most relevant pages.
- It sends those pages to an AI model, which reads them and writes you a clear answer.
- It even tells you which page numbers the answer came from, so you can go check for yourself.

## How it works (the simple version)

1. **The PDF gets downloaded** from Google Drive the first time the app starts.
2. **The text gets chopped into small chunks** (a few hundred words each) so it's easier to search through.
3. **Each chunk gets turned into numbers** (called "embeddings") that capture what it's about. This is done with a free, open model from Hugging Face.
4. **All those chunks get stored in a search index** called FAISS, which is really fast at finding "chunks that are similar to this question."
5. **When you ask a question**, the app finds the 4 most relevant chunks and hands them to an AI model (running on Groq, which is very fast) along with your question.
6. **The AI answers using only what's in those chunks** — if the answer isn't in the document, it says so instead of making something up.

## What it's built with

| Piece | What it's for |
|---|---|
| [Streamlit](https://streamlit.io) | The website/app interface you see and click around in |
| [Groq](https://groq.com) | Runs the AI model (LLaMA 3.3) super fast |
| [FAISS](https://github.com/facebookresearch/faiss) | Searches through the document quickly |
| [LangChain](https://www.langchain.com) | Glues the PDF-reading, chunking, and searching steps together |
| [Hugging Face](https://huggingface.co) embeddings | Turns text into numbers so it can be searched |

## Running it yourself

### 1. Get the code

Clone this repo or download the files (`app.py` and `requirements.txt`).

### 2. Install what it needs

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

You'll need a free API key from [Groq](https://console.groq.com). Once you have it:

**If running locally:** create a file called `.streamlit/secrets.toml` in the project folder and add:

```toml
GROQ_API_KEY = "your-key-here"
```

**If deploying on Streamlit Community Cloud:** go to your app's **Settings → Secrets** and paste the same line in there instead.

### 4. Run it

```bash
streamlit run app.py
```

That's it — it'll open in your browser.

## Deploying it online (Streamlit Community Cloud)

1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo, and set the main file to `app.py`.
4. Add your `GROQ_API_KEY` under **Advanced settings → Secrets**.
5. Click **Deploy** and wait a minute or two.

The first time anyone opens the app, it'll take a bit longer to load — it's downloading the PDF and building the search index. After that first time, it's cached, so it stays fast.

## Using your own PDF instead

Open `app.py` and find this part near the top:

```python
GDRIVE_FILES = [
    ("1CqlwwJ9pWlLPyU2VJ3kCvuC5nPAE907H", "python_for_linguists.pdf"),
]
```

Replace the file ID and name with your own PDF's Google Drive file ID. You can add more than one PDF here too — just add another line.

## A couple of things to know

- The free tier of Streamlit Cloud has limited memory, so if you load a *lot* of very large PDFs, it might run out of room. One or two average-sized PDFs should be fine.
- If the app says the `GROQ_API_KEY` is missing, double check your secrets are saved correctly and that the app has finished restarting.
- Answers are based only on the document — if you ask something the document doesn't cover, it'll tell you it couldn't find that information instead of making things up.

## Built with

Streamlit · Groq · FAISS · LangChain · Hugging Face
