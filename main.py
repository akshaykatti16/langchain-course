from dotenv import load_dotenv, find_dotenv, dotenv_values
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#for custom search tool
from tavily import TavilyClient
#for readymade tool
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field
import os

load_dotenv()

#to format agent output
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

#source class is nested here below, this is actual response
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="Thr agent's answer to the query")
    #list of source class objects
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

tavily = TavilyClient()

# --- Pydantic schema for tool input ---
class TavilyQuery(BaseModel):
    """Schema for Tavily search tool"""
    query: str = Field(..., description="Search query to look up online")
    max_results: int = Field(3, description="Number of search results to return")

@tool(args_schema=TavilyQuery)
def tavily_search(query: str, max_results: int = 3):
    """
    Uses Tavily search engine to fetch the most relevant results.
    """
    print(f"Searching Tavily for: {query}")
    return tavily.search(query=query, max_results=max_results)

#custom search agent
@tool
def search(query : str) -> str:
    """
    Executes a search query using a LangChain agent and returns the most relevant hits.
    Args - search query
    Retrun - search results
    """
    print(f"searching for {query}")
    #return "You have weather information now!!"
    return tavily.search(query=query)

llm = ChatOpenAI(model='gpt-5-mini')
#tools = [search]
#tools=[TavilySearch]
# Use this tool instead of TavilySearch
tools = [tavily_search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    #print(os.getenv("OPENAI_API_KEY"))
    #result = agent.invoke({"messages":HumanMessage(content="What is the weather in tokyo")})
    result = agent.invoke({"messages":HumanMessage(content="give me senior ai engineer jobs in pune location india")})
    print(result)


if __name__ == "__main__":
    main()
