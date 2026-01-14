"""
Created:20251203
https://chatgpt.com/c/692fd62c-ded4-8325-af62-2ea1357aa16c
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationEntityMemory
from langchain_classic.prompts import PromptTemplate

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
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

# print(chain.run("Hi my name is kiran and i live in africa"))
# print("="*50) #delimmiter purpose
# print(chain.run("i work with python and langchain"))
# print("="*50)
# print(chain.run("where do i live?"))
# print("="*50)
# print(chain.run("what do i work on?"))
# print("="*50)
def main():
    
    try:
        while True:
            print("type exit or quit to exit the chat")
            user_input=input("You:").strip()
            if user_input.lower() in ("exit","quit"):
                print("Exit Success")
                break
            if not user_input:
                continue
            response=chain.run(input=user_input)
            print("AI:",response)
            print("***********************************************\n**********")
            print(memory.entity_store.store)
    except KeyboardInterrupt:
        print("Exit Successful")


if __name__=='__main__':
    main()