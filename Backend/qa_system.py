from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def setup_qa_system(docsearch):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    custom_prompt_template = """
    You are a spot data analyst. You will be given an excerpt from a dataset or research paper. Your task is to quickly analyze the information and provide a precise and actionable response. Follow these guidelines:Data Analysis Task:

    {context}

    Now, based on this information, answer the following question:

    {question}
    Based on the given data, please provide a concise and insightful analysis, adhering to the following:
    1.Immediate Insight**: Identify key patterns, trends, or outliers in the data.
    2. Actionable Recommendations**: Suggest any actions, decisions, or next steps based on your analysis.
    3.Data Visualization** (if applicable): 
        - Suggested Chart Type (e.g., bar chart, line chart, scatter plot)
        - Key Data Points (in comma-separated format)
        - Axis Labels (for x and y axes)
        - Suggested Chart Title
    4.Text Summary** (if applicable): Provide a brief text-based summary highlighting key findings, anomalies, or recommendations.
    
    Be clear, direct, and focus on delivering insights that can drive decisions quickly

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
