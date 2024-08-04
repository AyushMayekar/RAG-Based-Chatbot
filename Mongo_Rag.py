import os, pymongo
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(
    title="MongoDB RAG",
    description="A simple API using FastAPI and RAG",
)

client = MongoClient("mongodb+srv://ayush224947101:AYUSH21@cluster0.mq8dx3f.mongodb.net/?appName=Cluster0")

#* Define collection and index name
db_name = "ayush"
collection_name = "ayush"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"

# Loading the environment variables
groq_api_key = os.environ['groq_api_key']

# Defining the LLM
llm = ChatGroq(groq_api_key = groq_api_key, 
                model_name = 'mixtral-8x7b-32768')

# Defining the Prompt
Prompt = PromptTemplate.from_template("""
Act as an intelligent assistance and answer the questions asked on the basis of provided context only, respond a generalised message of no context
if the answer cannot be found in the context provided, design every response in a very appropriate and professional manner
<context>
{context}
</context>
Question : {input}
""")

# Initializing retrieval chain
retrieval_chain = None

# Initializing a flag
# Check if embeddings already exist in MongoDB
if atlas_collection.count_documents({}) > 0:  # Adjust this condition based on your collection structure
    loaded_embeddings = True
    print("Embeddings already loaded in MongoDB.")
else:    
    loaded_emebeddings = False

# Function to load,split and embed the document
def load_embeddings():
    # Loading the context
    loader = PyPDFLoader(r"C:\Users\ayush\Desktop\vs\Python\langchain_tut\RAG\attention.pdf")
    docs = loader.load()

    # Splitting the loaded docs
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    split_docs = splitter.split_documents(docs)

    # Creating embeddings of the splitted data
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    # Creating a vector store form MongoDB Atlas
    vector_store = MongoDBAtlasVectorSearch.from_documents(documents = split_docs,
                                                        embedding = embeddings,
                                                        collection = atlas_collection,
                                                        index_name = vector_search_index)
        

    # Creating a document chain
    global llm, Prompt
    doc_chain = create_stuff_documents_chain(llm = llm, prompt = Prompt, output_parser= StrOutputParser())

    # Defining the vector store as retriever
    retriever = vector_store.as_retriever( search_type = "similarity",
    search_kwargs = { "k": 10 })

    # Creating a retrieval chain
    global retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # Updating the flag
    global loaded_emebeddings
    loaded_emebeddings = True

# Check if the Document is embedded ie. embeddings are loaded
if not loaded_emebeddings:
    load_embeddings()

# Define a route
@app.post("/api")
async def invoke_retrieval(request: Request):
    input_data = await request.json()
    query = input_data.get("input", {}).get("input")
    if query:
        response = retrieval_chain.invoke({"input": query})
        return {"response": response['answer']}
    return {"response": "No query provided"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")