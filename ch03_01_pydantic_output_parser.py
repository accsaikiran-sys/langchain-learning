"""
Pydantic Output Parser with LangChain

This script demonstrates how to get STRUCTURED data from an AI model.
Instead of getting plain text, we tell the AI to return data in a specific format
(like a form with name, age, and city fields).
"""

# Import necessary libraries
import os
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI
from langchain_core.prompts import PromptTemplate  # For creating prompt templates
from langchain_core.output_parsers import PydanticOutputParser  # Converts AI text into structured data
from pydantic import BaseModel, Field  # For defining data structure/schema
from dotenv import load_dotenv  # For loading API keys

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Define the structure we want the AI to return
# This is like creating a form with specific fields
class PersonInfo(BaseModel):
    name: str = Field(description="name of the person")  # Must be text (string)
    age: int = Field(description="the name of the person")  # Must be a number (integer)
    city: str = Field(description="city where they are from")  # Must be text (string)


# Initialize the Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Create a parser that will convert AI's text response into our PersonInfo structure
# This ensures we get data in the exact format we defined above
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Create a prompt template that tells the AI:
# 1. What to do (extract person information)
# 2. How to format the output (using format_instructions from the parser)
# 3. Where the input text will go (user_input placeholder)
template = """
Extract structured information about a person from the text below
{format_instructions}

TEXT:
{user_input}
"""

# Create the PromptTemplate object
# input_variables: what we'll provide when using this template
# partial_variables: automatically filled values (the parser's formatting instructions)
prompt = PromptTemplate(
    template=template,
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

user_text = "John is 27 years old and lives in Saskatoon"

final_prompt = prompt.format(user_input=user_text)

response = llm.invoke(final_prompt)

print(response.content)

person = parser.parse(response.content)

print(person)