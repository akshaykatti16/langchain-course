import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-5-mini")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    
    #initialize pinecone
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    #using LCEL
    
    template="""Use the following pieces of context to answer the questions.
    If you dont know the answer, just say you dont know.
    use 3-4 sentences maximum and keep the answer concise.
    {context}
    
    Question : {question}
    
    Helpful Answer : 
    
    """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {
            "context" : vectorstore.as_retriever() | format_docs,
            "question" : RunnablePassthrough()
        }
        | custom_rag_prompt
        | llm

    )

    res = rag_chain.invoke(query)

    print(res)
