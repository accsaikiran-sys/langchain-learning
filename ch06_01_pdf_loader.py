import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = "SaskGov-cv.pdf"

def build_retriever():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def main():
    # Make sure you set your Google API key
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("Please set GEMINI_API_KEY environment variable.")

    retriever = build_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    print("PDF ready! Ask questions. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\nAnswer:", qa.run(q))

if __name__ == "__main__":
    main()