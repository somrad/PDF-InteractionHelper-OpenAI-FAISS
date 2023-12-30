import os
from langchain.llms.openai import OpenAI

from PDFInteractionHelper import PDFInteractionHelper


if __name__ == "__main__":
    helper = PDFInteractionHelper("./pdf_source/2210.03629.pdf", "faiss_react_index")

    # Sample question 1: Get a gist of the document
    #    helper.Query('GIve me the gist of React in 3 sentences')

    # Sample question 2: Ask LLM to generate a few questions based on the document
    helper.Query("GIve me 5 questions about React based on the given context")
