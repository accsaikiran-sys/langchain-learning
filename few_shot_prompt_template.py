"""
Few-Shot Prompting with LangChain and Gemini AI

This script demonstrates "few-shot learning" - teaching the AI by showing it examples
before asking it to answer a new question. This helps the AI understand the style 
and format you want for answers.
"""

# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate  # Tools for creating prompts
import os  # For accessing environment variables
from dotenv import load_dotenv  # For loading API keys from .env file


# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Create a list of example Q&A pairs to teach the AI
# These examples show the AI what kind of answers we want (simple, short explanations)
examples = [
    {
        "question": "what is AI",
        "answer": "AI is technology that responses with reasoning or human like intelligence"
    },
    {
        "question": "what is python?",
        "answer": "Python is a programming language"
    }
]

# Create a template for how each example should be formatted
# {question} and {answer} are placeholders that will be filled with actual values
example_template = """
Q: {question}
A: {answer}
"""

# Convert the example template into a PromptTemplate object
# This tells LangChain which variables to look for in the template
example_prompt = PromptTemplate(
    template=example_template,
    input_variables=["question", "answer"]
)

template = """you are a helpful assistant 
Use the example above to answer the new question
{exaples}

Q: {new_question}
A:
"""

# Create a FewShotPromptTemplate that combines everything
# This will show the AI the examples, then ask it a new question
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,  # The example Q&A pairs we created above
    example_prompt=example_prompt,  # How to format each example
    prefix="you are an expert teacher. use simple explanations",  # Instructions before examples
    suffix="now answer this question:\nQ: {new_question}\nA:",  # Text after examples, with placeholder for new question
    input_variables=["new_question"]  # The variable we'll provide when using this template
)

# Format the complete prompt by inserting our actual question
# This creates a full prompt with: prefix + examples + suffix with our question
final_prompt = few_shot_prompt.format(new_question="what is langchain?")

# Send the complete prompt to the AI and get a response
response = llm.invoke(final_prompt)

# Print only the text content of the response (not metadata)
print(response.content)