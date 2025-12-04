"""
Created:20251203
https://chatgpt.com/c/692fd62c-ded4-8325-af62-2ea1357aa16c

"""


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

memory=ConversationKGMemory(llm=llm)

conversation=ConversationChain(llm=llm, memory=memory)

print(conversation.predict(input="hi my name is kiran"))
print("="*50)
print(conversation.predict(input="i work on langchain and python"))
print("="*50)
