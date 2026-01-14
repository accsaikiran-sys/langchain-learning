# fix_my_recipe.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int
    instructions: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", api_key=os.getenv("GEMINI_API_KEY"))

# 1. base parser
base_parser = PydanticOutputParser(pydantic_object=Recipe)

# 2. fixing wrapper
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

prompt = PromptTemplate(
    template=(
        "Give me a simple cookie recipe in JSON with keys: name, ingredients, prep_time_minutes, instructions. "
        "Add a friendly note before and after the JSON block. Don't worry about perfect syntax."
    ),
    input_variables=[]
)

# raw LLM call so we can print the messy string
raw_llm_response = llm.invoke(prompt.format())
print("RAW LLM TEXT:\n", raw_llm_response.content, "\n")

# parsed / fixed object
parsed = fixing_parser.parse(raw_llm_response.content)
print("FIXED OBJECT:\n", parsed)
print("Recipe name:", parsed.name)
print("Prep time:", parsed.prep_time_minutes, "minutes")