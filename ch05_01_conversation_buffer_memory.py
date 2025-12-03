import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

memory=ConversationBufferMemory()

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