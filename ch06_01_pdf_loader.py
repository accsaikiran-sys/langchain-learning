import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

pdf="sample.pdf"

loader=PyPDFLoader(pdf)

document=loader.load()

print("Pages Loaded: ",len(document))

splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(document)

print("Chunks:",len(chunks))

query="Summarize this pdf in simple words"

full_text="\n\n".join(chunk.page_content for chunk in chunks)

response=llm.invoke(full_text + query)

print(response.content)
