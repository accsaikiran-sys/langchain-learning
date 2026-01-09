from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
import os
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key = api_key
)

template = """you are a helpful assistant, answer the user's question in 1 precise line.

Question: {query}
"""

prompt = PromptTemplate.from_template(template)

query = input("Enter query: ")

final_prompt = prompt.format(query = query)


response = llm.invoke(final_prompt)

print(response.content)