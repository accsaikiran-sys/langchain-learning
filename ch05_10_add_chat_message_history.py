"""
Created: 20251203


"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key,
    convert_system_message_to_human=True
)

prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant"),
    ("placeholder","{history}"),
    ("human","{input}")
])

chain = prompt | llm

store = {} # ---> session_id

def get_history(session_id: str):
    if session_id not in store:
        store[session_id]=ChatMessageHistory() # ---> it will create a new chat history
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

session = {"configurable":{"session_id": "user_1"}}

print(conversation.invoke({"input" : "hi, i am akiran"}, config = session))

print("-"*50)

print(conversation.invoke({"input": "what is my name?"}, config=session))
print("-"*50)

session_2= {"configurable":{"session_id": "user_2"}}


print(conversation.invoke({"input": "who am i ?"}, config=session_2))
