import os
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Step 1: Define enum
class ColorEnum(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# Step 2: Define Pydantic model using enum
class FavoriteColor(BaseModel):
    color: ColorEnum

# Step 3: Create parser
parser = PydanticOutputParser(pydantic_object=FavoriteColor)

# Step 4: Prepare prompt
prompt = PromptTemplate(
    template="Pick the favorite color from red, green, or blue and respond in JSON format like {{'color':'red'}}.",
    input_variables=[]
)

# Step 5: Run LLM and parse output
response = llm.invoke(prompt.format())
parsed_output = parser.parse(response.content)

print(parsed_output)
print(parsed_output.color)  # guaranteed to be one of ColorEnum values
