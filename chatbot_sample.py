from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
load_dotenv()

video_id = "Gfr50f6ZBvo"

try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

vector_store = FAISS.from_documents(documents=chunks,embedding=OpenAIEmbeddings())

retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

llm = ChatOpenAI(model='gpt-3.5-turbo',temperature = 0.5)

prompt = PromptTemplate(
    template=""" You are a helpful assistant.
    Answer the below question using only the context provided. If the context is insufficient
    just say you don't know.
    {context}
    {question}
""",
input_variables=['context','question']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
   'context':retriever | RunnableLambda(format_docs),
   'question':RunnablePassthrough()
}
)
parser = StrOutputParser()

final_chain = parallel_chain | prompt | llm | parser

print(final_chain.invoke("Who is Demis?"))

