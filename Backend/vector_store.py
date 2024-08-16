from pymongo import MongoClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import os
from dotenv import load_dotenv

load_dotenv()

def get_mongo_client():
    uri = os.getenv("MONGODB_URI")
    return MongoClient(uri)

def setup_vector_store(texts):
    client = get_mongo_client()
    db_name = "langchain_demo"
    collection_name = "RagApplication"
    collection = client[db_name][collection_name]

    embeddings = FastEmbedEmbeddings()
    docsearch = MongoDBAtlasVectorSearch.from_documents(texts, embeddings, collection=collection)
    return docsearch
