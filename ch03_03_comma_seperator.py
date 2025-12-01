import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=api_key
)

template="""
give me  5 fruit names
"""

prompt = PromptTemplate.from_template(template)

final_prompt = prompt.format()

response=llm.invoke(final_prompt)

print(response.content)

parser=CommaSeparatedListOutputParser()

result = parser.parse(response.content)

print(result)