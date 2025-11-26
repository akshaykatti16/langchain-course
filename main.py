from dotenv import load_dotenv, find_dotenv, dotenv_values
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#for custom search tool
from tavily import TavilyClient
#for readymade tool
from langchain_tavily import TavilySearch
import os

load_dotenv()
tavily = TavilyClient()

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
tools=[TavilySearch]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    #print(os.getenv("OPENAI_API_KEY"))
    #result = agent.invoke({"messages":HumanMessage(content="What is the weather in tokyo")})
    result = agent.invoke({"messages":HumanMessage(content="search for top 3 fastest flights between new york and london")})
    print(result)


if __name__ == "__main__":
    main()

# class Source(BaseModel):
#     """Schema for a source used by the agent"""

#     url: str = Field(description="The URL of the source")


# class AgentResponse(BaseModel):
#     """Schema for agent response with answer and sources"""

#     answer: str = Field(description="Thr agent's answer to the query")
#     sources: List[Source] = Field(
#         default_factory=list, description="List of sources used to generate the answer"
#     )


# llm = ChatOpenAI(model="gpt-5")
# tools = [TavilySearch()]
# agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


# def main():
#     print("Hello from langchain-course!")
#     result = agent.invoke(
#         {
#             "messages": HumanMessage(
#                 content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
#             )
#         }
#     )
#     print(result)


# if __name__ == "__main__":
#     main()
