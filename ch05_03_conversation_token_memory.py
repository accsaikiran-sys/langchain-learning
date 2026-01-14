"""

Created: 20251202
https://chatgpt.com/c/692fc826-132c-8325-bd9c-0f7010a9a8b8

"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationTokenBufferMemory

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

memory=ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_message=True
)

conversation=ConversationChain(
    llm=llm,
    memory=memory
)

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
            response=conversation.run(input=user_input)
            print("AI:",response)
            print("***********************************************\n**********")
    except KeyboardInterrupt:
        print("Exit Successful")


if __name__=='__main__':
    main()