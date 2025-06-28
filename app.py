import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
import re

load_dotenv()

# ------------------------- UI ------------------------- #
st.title("Tool for Answering Questions on Any English YouTube Video")

url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Enter your question:")

submit = st.button("Submit")

# ---------------------- Helper Functions ---------------------- #
def extract_video_id(youtube_url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", youtube_url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ------------------------ RAG Pipeline ------------------------ #
if submit and url and question:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please provide a valid link.")
    else:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            vector_store = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
            prompt = PromptTemplate(
                template=""" You are a helpful assistant.
                Answer the below question using only the context provided. If the context is insufficient
                just say you don't know.
                {context}
                {question}
                """,
                input_variables=['context', 'question']
            )

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()
            final_chain = parallel_chain | prompt | llm | parser

            answer = final_chain.invoke(question)
            st.subheader("Answer:")
            st.write(answer)

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
