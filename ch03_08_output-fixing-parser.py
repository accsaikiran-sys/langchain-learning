from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")

parser = PydanticOutputParser(pydantic_object=Person)

format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="Extract the following details:\nName: John Doe\nAge: thirty three\n{formatted_instructions}",
    partial_variables={"formatted_instructions": format_instructions}
)

fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key
    )
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

formatted_prompt = prompt.format()

response = llm.invoke(formatted_prompt)

fixed_response = fixing_parser.parse(response.content)

print(fixed_response)
