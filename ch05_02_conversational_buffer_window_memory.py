"""
Created: 20251202
https://course.kactii.com/course/hustlecamp-s3
https://chatgpt.com/c/692fb5e0-fcd4-8333-b435-72b14fea58ce
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

memory=ConversationBufferWindowMemory(k=1)

chain=ConversationChain(llm=llm,memory=memory)

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
    except KeyboardInterrupt:
        print("Exit Successful")


if __name__=='__main__':
    main()