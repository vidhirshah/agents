import os
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader


# Entire Rag pipeline
class RagPipeline:
    def __init__(self, chromaDBPath, collection_name, model_name):
        self.chromaDBPath = chromaDBPath
        self.collection_name = collection_name
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def websiteLoader(self, url: str, chunkSize: int = 500, overLap: int = 200):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        os.environ['LANGCHAIN_API_KEY'] = ""
    
        # Load website
        loader = WebBaseLoader(
            web_paths=(url,),
            # bs_kwargs=dict(
            #     parse_only=bs4.SoupStrainer("table")
            # ),
        )
        blog_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunkSize, 
            chunk_overlap=overLap)
        # Make splits
        splits = text_splitter.split_documents(blog_docs)

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.chromaDBPath,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("Connection Established .. ")
        print("\n\n\nConnection ... Done ....\n\n",self.chromaDBPath)
        # Add to the DB
        vector_store.add_documents(splits)
        print("Added to the DB .. ")

        return "Success"
    
    def searchQuery(self,query:str):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        os.environ['LANGCHAIN_API_KEY'] = ""
        os.environ["GOOGLE_API_KEY"] = ""

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.chromaDBPath,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        template = """Answer the question based only on the following context:
                    {context}
                    Question: {question}
                    """

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        outputParser = StrOutputParser()
        retrived_data = retriever.invoke(query)
        retrived_chunks = []
        for i, data in enumerate(retrived_data, 1):
            temp = {}
            temp['chunk_id'] = data.id
            temp['chunk_content'] = data.page_content
            retrived_chunks.append(temp)
        return retrived_chunks