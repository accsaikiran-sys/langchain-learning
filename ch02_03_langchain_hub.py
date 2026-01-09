import os
from dotenv import load_dotenv
from langchain_core.hub import HubTool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Load a prebuilt chain from LangChain Hub
hub_chain = HubTool.from_hub("examples/simple-qa-chain")

# Run the chain
response = hub_chain.run({"query": "Who invented the lightbulb?"})
print(response)