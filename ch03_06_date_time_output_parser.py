
"""
Key Notes

DateTimeOutputParser is no longer part of LangChain.

Use Pydantic models with datetime fields instead.

The LLM should output JSON with ISO date format (YYYY-MM-DD) to make parsing reliable.

This method works for any structured data containing dates, times, or timestamps.
"""


import os
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Step 1: Define a Pydantic model with a datetime field
class PersonDOB(BaseModel):
    name: str
    dob: datetime

    # Optional: Convert string to datetime if needed
    @validator("dob", pre=True)
    def parse_date(cls, v):
        return datetime.fromisoformat(v.strip())

# Step 2: Create a Pydantic parser
parser = PydanticOutputParser(pydantic_object=PersonDOB)

# Step 3: Prompt the LLM to return JSON
prompt = PromptTemplate(
    template="""
Provide Elon Musk's name and date of birth in JSON format:
{{"name": "Elon Musk", "dob": "YYYY-MM-DD"}}
""",
    input_variables=[]
)

# Step 4: Run LLM and parse output
response = llm.invoke(prompt.format())
parsed_output = parser.parse(response.content)

print(parsed_output)
print(parsed_output.name, parsed_output.dob)  # dob is a datetime object
