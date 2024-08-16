from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.qa import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def setup_qa_system(docsearch):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    custom_prompt_template = """
    Consider the following research paper excerpt:

    {context}

    Now, based on this information, answer the following question:

    {question}

    Provide a concise and accurate response below:
    If the question requires a chart or figure, please provide the following:
    - Chart Type (e.g., bar chart, line chart, scatter plot)
    - Data (in a simple format like comma-separated values)
    - Axis Labels (for x and y axes)
    - Chart Title

    Otherwise, provide a concise and accurate text response below:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa
