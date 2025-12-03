import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

parser=DatetimeOutputParser()

formatted_instructions=parser.get_format_instructions()

prompt=PromptTemplate(
    template="Provide the date for the next christmas.\n{formatted_instructions}",
    partial_variables={"formatted_instructions":formatted_instructions}
)

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

formatted_prompt=prompt.format()

result=llm.invoke(formatted_prompt)

parsed=parser.parse(result.content)

print(parsed)
