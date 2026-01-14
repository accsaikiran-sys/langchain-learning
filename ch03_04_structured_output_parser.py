import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)

# Define the structure we want
class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    city: str = Field(description="City where person lives")
    hobbies: List[str] = Field(description="List of person's hobbies")

# Create the parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Create prompt with format instructions
prompt = PromptTemplate(
    template="Extract the following information about a person:\n{format_instructions}\n\nText: John Smith is 25 years old and lives in New York. He enjoys reading, swimming, and playing guitar.",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Generate and parse
response = llm.invoke(prompt.format())
print("Raw response:")
print(response.content)
print("\nParsed structure:")
parsed = parser.parse(response.content)
print(parsed)
print(f"\nName: {parsed.name}")
print(f"Age: {parsed.age}")
print(f"City: {parsed.city}")
print(f"Hobbies: {parsed.hobbies}")