import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.output_parsers import OutputFixingParser, PydanticOutputParser

# Load GEMINI_API_KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GEMINI_API_KEY in your environment")

# Define structured output
class Person(BaseModel):
    name: str
    age: int

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=api_key
)

# Example text that will cause type errors
text = "Charlie is forty-two years old."

# ---------------------------
# Without OutputFixingParser
# ----------------------------
raw_output = llm.invoke([
    ("system", "Extract structured data in JSON format."),
    ("user", f"Extract name and age from: {text}")
])

print("=== Without OutputFixingParser ===")
print(raw_output.content)

# If you try to parse this raw output directly using Pydantic:
try:
    Person.parse_raw(raw_output.content)
except Exception as e:
    print("Error while parsing:", e)


# -----------------------
# With OutputFixingParser
# -----------------------
pydantic_parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

parsed_output = fixing_parser.parse(raw_output.content)
print("\n=== With OutputFixingParser ===")
print(parsed_output)