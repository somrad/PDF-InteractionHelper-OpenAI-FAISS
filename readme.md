# Overview

A simple class PDFInteractionHelper to help with the following:

## Load the file

- load a PDF document - we use a paper "REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS" in a file "./pdf_source/2210.03629.pdf"
- chunk it 
- encode it using the OpenAI Encoding
- load it in FAISS vector database

## Query the file using Open AI

- Loads the FAISS vector data stored earlier
- RetrievalQA is used get the relevant context and send the query and context to LLM

# Getting Started

## Environment set up
- Set OPENAI_API_KEY in .env file

## pipenv set up
- run pipenv shell
- run pipenv install 

### packages used

- langchain = "*"
- openai = "*"
- pypdf = "*"
- black = "*"
- tiktoken = "*"
- faiss-cpu = "*"



# Running

python main.py
