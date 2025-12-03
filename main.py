import os
from typing import List

from dotenv import dotenv_values, find_dotenv, load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
# for readymade tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
# for custom search tool
from tavily import TavilyClient

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
#with_structured_ouput // alternative to pydantic output parser
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")

# for pydantic output
#output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# react_prompt_WITH_FORMAT_INSTRUCTIONS = PromptTemplate(
#     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
#     input_variables=["input","agent_scratchpad","tool_names"]
#     ).partial(format_instruction=output_parser.get_format_instructions())

# react_prompt_WITH_FORMAT_INSTRUCTIONS = PromptTemplate(
#     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
#     input_variables=["input", "agent_scratchpad"],
# ).partial(
#     format_instructions=output_parser.get_format_instructions()
# )

#correct one
# react_prompt_WITH_FORMAT_INSTRUCTIONS = PromptTemplate.from_template(
#     REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
# ).partial(
#     tool_names=", ".join([tool.name for tool in tools]),
#     format_instructions=output_parser.get_format_instructions(),
# )

#for with_structured_output
react_prompt_WITH_FORMAT_INSTRUCTIONS = PromptTemplate.from_template(
    REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
).partial(
    tool_names=", ".join([tool.name for tool in tools]),
    #empty string
    format_instructions="",
)

agent = create_react_agent(llm=llm, 
                           tools=tools, 
                           # from hub
                           # prompt=react_prompt
                           prompt=react_prompt_WITH_FORMAT_INSTRUCTIONS)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# output parser with LECL
extract_output = RunnableLambda(lambda x : x["output"])

#not required for with_structured_output 
#parse_output = RunnableLambda(lambda x : output_parser.parse(x))

#chain = agent_executor | extract_output | parse_output
#type(result) will be dict

#chain = agent_executor | extract_output | parse_output
#type(result) will be AgentResponse (pydantic output)

#for with_structured_output
chain = agent_executor | extract_output | structured_llm

def main():
    print("Hello from langchain-course!")
    # print(os.getenv("OPENAI_API_KEY"))
    result = chain.invoke(
        input={
            "input": "give me 3 senior ai engineer linkedin job postings for bay area location, sorted by latest time of its posting"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
