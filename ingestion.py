import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("C:\\Users\\akshaykatti\\langchain-course\\mediumblog1.txt",
                        encoding="utf-8",
                        autodetect_encoding=True)
    document = loader.load()

    print("splitting...")
    #each document has chunk of 1000 words
    #for 1k chunk size, 20 sub documents were created
    #for 1.5k chunk size, 13 sub documents were created
    #for 2.5k chunk size, 8 sub documents were created
    # higher the no of chunk size -> lower the no of sub documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #list of documents
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")
    print(f"chunking & splitting completed")
    print("***")

    #embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting to vector database...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
