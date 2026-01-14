import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryMemory


load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    api_key=api_key
)

memory=ConversationSummaryMemory(llm=llm)

conversation=ConversationChain(llm=llm,memory=memory)

def main():
    print("type exit or quit to stop")
    try:
        while True:
            user_input=input("You:").strip()

            if user_input.lower() in {"exit","quit"}:
                print("Exiting.....")
                break
            if not user_input:
                continue
            response=conversation.invoke(input=user_input)
            print("AI:",response)
            print("==========================================================================================")
    except KeyboardInterrupt:
        print("Exiting")

if __name__=='__main__':
    main()