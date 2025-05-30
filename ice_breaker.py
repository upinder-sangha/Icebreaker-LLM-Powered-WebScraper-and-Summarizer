from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile

# Load environment variables from .env
load_dotenv()

if __name__ == "__main__":
    print("Hello world!")

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary of the person
    2. Two interesting facts about them
    """

    summary_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Create Groq LLM instance
    # llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    llm = ChatOllama(model="llama3.2")

    # Chain together the prompt and LLM
    chain = summary_prompt | llm | StrOutputParser()
    linkedIn_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/upinder-sangha/", mock=True)

    # Run the chain with input
    res = chain.invoke(input={"information": linkedIn_data})

    print(res)
