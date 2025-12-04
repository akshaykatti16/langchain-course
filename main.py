import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("hi")
    llm = ChatOpenAI(model="gpt-5-mini")
    pdf_path = "2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    #pdf split into 30+ documents
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    #further these 33 documents split into chunks of 100+ sub documents , suitable for vector store

    docs = text_splitter.split_documents(documents=documents)
    print(len(docs))

    embeddings = OpenAIEmbeddings()
    #FAISS is local vector store, from facebook made open source 
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    
    #pinecone vector store code
    # PineconeVectorStore.from_documents(
    #     texts, embeddings, index_name=os.environ["INDEX_NAME"]
    # )
    

    #load vector database for 
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    #gets relevant documents
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    #format retrieved docs
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res)
