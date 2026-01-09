import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=api_key

)

examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer": "Einstein"
    },
    {
        "question": "When was Naver's founder born?",
        "answer": " June 22, 1967"
    },
    {
        "question": "Who was the king who ruled in the year that Yulgok Yi I's mother was born?",
        "answer": "Yeonsangun"
    },
    {
        "question": "Are the directors of Oldboy and Parasite from the same country?",
        "answer": "Yes"
    },
]

example_template ="""
Q:{question}
A:{answer}
"""

main_template= """
you are an simple ai assisstant.
Using the examples above answer the new question
{examples}

now answer this

Q: {new_question}
A:
"""


few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template(example_template),
    prefix="you are a expert teacher. use this simple explanations",
    suffix="now answer this question: {new_question}",
    input_variables=["new_question"]
)

final_prompt = few_shot_prompt.format(new_question="What is langchain?")

response=llm.invoke(final_prompt)

print(response.content)