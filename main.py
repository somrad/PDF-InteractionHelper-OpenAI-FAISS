import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI


class PDFInteractionHelper():

    embeddings = OpenAIEmbeddings()

    
    def __init__(self, pdf_filename: str, index_store_location:str) -> None:
        self.pdf_filename = pdf_filename
        self.index_store_location = index_store_location


        self.loader = PyPDFLoader(pdf_filename)
        self.document = self.loader.load()
        print('----Document Loaded from file-----')
        # print(self.document)
        
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
        self.docs = text_splitter.split_documents(self.document)

        self.vectorstore = FAISS.from_documents(self.docs,self.embeddings)
        self.vectorstore.save_local(index_store_location)
        print('----Document Loaded to vector store-----')


    def Query(self,question: str) -> str:
        print('Query: ----Vector store from Local-----')

        new_vectorStore = FAISS.load_local(self.index_store_location, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(),chain_type='stuff', 
                                              retriever=new_vectorStore.as_retriever())
        result = self.qa.run(question)
        print(result)
        return result


if __name__ == '__main__':
   helper = PDFInteractionHelper('./2210.03629.pdf',"faiss_react_index")
#    helper.Query('GIve me the gist of React in 3 sentences')
   helper.Query('GIve me 5 questions about React based on the given context')
