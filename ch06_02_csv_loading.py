import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_classic.text_splitter import CharacterTextSplitter

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key
)

csv_path="sample.csv"

loader=CSVLoader(csv_path)

documents=loader.load()

print(len(documents))

splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=50
)

chunks=splitter.split_documents(documents)

query="give me a summary of the csv data"
print("Total chunks:", len(chunks))

full_text = "\n".join(doc.page_content for doc in chunks)

response = llm.invoke(full_text+query)

print(response.content)
