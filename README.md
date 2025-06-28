# 🎥 YouTube Video Q&A App (RAG-powered with LangChain & Streamlit)

This tool lets you **ask questions about any English YouTube video**, using its transcript and a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain, FAISS, and OpenAI.

---

## 🚀 Features

- 🔗 Input any **YouTube video URL**
- ❓ Ask a natural language **question** about the video
- 🧠 Uses **YouTubeTranscriptAPI** to fetch subtitles
- 🪄 Leverages **LangChain**, **FAISS**, and **OpenAI GPT-3.5** to find answers
- 💻 Built with **Streamlit** for easy interaction
- 📦 Managed using **Poetry**

---

## 📦 Setup (Poetry)

### 1. Clone this repo
```bash
git clone https://github.com/gvsreekar/yt-chatbot.git
cd yt-chatbot
```
### 2. Install dependencies using Poetry
```bash
poetry install
```
### 3.Add environment Variables
OPENAI_API_KEY=your_openai_api_key_here

### 4. Run the App
```bash
poetry run streamlit run app.py
```
