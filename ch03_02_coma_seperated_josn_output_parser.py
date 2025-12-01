"""
JSON Output Parser with LangChain

This script demonstrates how to get JSON formatted output from an AI model.
JSON is a common data format that's easy to work with in programming.
"""

# Import necessary libraries
import os
from langchain_core.prompts import PromptTemplate  # For creating prompt templates
from langchain_core.output_parsers.json import JsonOutputParser  # Converts AI text into JSON
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI
from dotenv import load_dotenv  # For loading API keys

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Create a JSON parser
# This will convert the AI's text response into a Python dictionary (JSON format)
parser = JsonOutputParser()

# Create a prompt template that asks the AI to return data in JSON format
# The AI should return an object with keys: language(string), year(int)
template = """
return a JSON object with keys : language(string),year(int).
do not include any extra text.
"""

# Convert the template string into a PromptTemplate object
prompt = PromptTemplate.from_template(template)

# Format the prompt (in this case, no variables to fill in)
final_prompt = prompt.format()

# Send the prompt to the AI and get a response
response = llm.invoke(final_prompt)

# Parse the AI's text response into a Python dictionary (JSON)
parsed = parser.parse(response.content)

# Print the parsed JSON data
print(parsed)
