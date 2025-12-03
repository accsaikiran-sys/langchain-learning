import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

memory=ConversationEntityMemory(llm=llm)

prompt=PromptTemplate(
    input_variables=["history","input"],
    template="""
Conversation so far:
{history}

User:{input}
AI:
"""
)

chain=LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

print(chain.run("Hi my name is kiran and i live in africa"))
print("="*50)
print(chain.run("i work with python and langchain"))
print("="*50)
print(chain.run("where do i live?"))
print("="*50)
print(chain.run("what do i work on?"))
print("="*50)
