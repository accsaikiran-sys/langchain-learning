from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_classic.chains import ConversationChain
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_texts(["Initial context"], embeddings)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

memory = VectorStoreRetrieverMemory(
    retriever=retriever
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.invoke("My name is John and I live in Toronto")
conversation.invoke("I work at Google")
conversation.invoke("Where do I live?")