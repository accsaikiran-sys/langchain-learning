"""Created: 20251129
This script demonstrates how to use LangChain with Google's Gemini AI model.
It creates a simple chatbot that answers questions using a prompt template.
"""

# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI integration
from langchain_core.prompts import PromptTemplate  # Tool to create reusable prompt templates
import os  # For accessing environment variables
from dotenv import load_dotenv  # For loading API keys from .env file

# Load environment variables from .env file (keeps API keys secure)
load_dotenv()

# Get the Gemini API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini AI model
# model: specifies which version of Gemini to use
# api_key: your authentication key to access the API
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

# Create a prompt template with a placeholder {query}
# This template defines how the AI should behave and where to insert the user's question
template = """you are a helpful assistant 
answer the user's question in simple words

Question: {query}
"""

# Convert the template string into a PromptTemplate object
# This allows us to easily insert different questions into the same template
prompt = PromptTemplate.from_template(template)

# Get input from the user
user_question = input("Question: ")

# Format the prompt by replacing {query} with the actual question
# Example: if question = "what is langchain?", {query} becomes "what is langchain?"
final_prompt = prompt.format(query=user_question)

# Send the formatted prompt to the AI model and get a response
response = llm.invoke(final_prompt)

# Print the AI's response to the console
print(response.content)