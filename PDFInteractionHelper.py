from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import logging


class PDFInteractionHelper:
    embeddings = OpenAIEmbeddings()

    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)

    def __init__(self, pdf_filename: str, index_store_location: str) -> None:
        self.pdf_filename = pdf_filename
        self.index_store_location = index_store_location

        self.loader = PyPDFLoader(pdf_filename)
        self.document = self.loader.load()
        logging.info("----Document Loaded from file-----")
        # print(self.document)

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separator="\n"
        )
        self.docs = text_splitter.split_documents(self.document)

        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.vectorstore.save_local(index_store_location)
        logging.info("----Document Loaded to vector store-----")

    def Query(self, question: str) -> str:
        logging.info("Query: ----Vector store from Local-----")

        new_vectorStore = FAISS.load_local(self.index_store_location, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=new_vectorStore.as_retriever(),
        )
        result = self.qa.run(question)
        logging.info(result)
        return result
