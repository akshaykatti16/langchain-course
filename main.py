from dotenv import load_dotenv, find_dotenv, dotenv_values
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_ollama import ChatOllama
import os

def main():
    print("Hello from langchain-course!")
    load_dotenv()
    #print(os.getenv("OPENAI_API_KEY"))
    information = """Avul Pakir Jainulabdeen Abdul Kalam (/ˈʌbdʊl kəˈlɑːm/ ⓘ UB-duul kə-LAHM; 15 October 1931 – 27 July 2015) was an Indian aerospace scientist and statesman who served as the president of India from 2002 to 2007.

    Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) and Indian Space Research Organisation (ISRO) and was intimately involved in India's civilian space programme and military missile development efforts. He was known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology. He also played a pivotal organisational, technical, and political role in Pokhran-II nuclear tests in 1998, India's second such test after the first test in 1974.

    Kalam was elected as the president of India in 2002 with the support of both the ruling Bharatiya Janata Party and the then-opposition Indian National Congress. He was widely referred to as the "People's President". He engaged in teaching, writing and public service after his presidency. He was a recipient of several awards, including the Bharat Ratna, India's highest civilian honour.

    While delivering a lecture at IIM Shillong, Kalam collapsed and died from an apparent cardiac arrest on 27 July 2015, aged 83. Thousands attended the funeral ceremony held in his hometown of Rameswaram, where he was buried with full state honours. A memorial was inaugurated near his home town in 2017.
    """
    
    prompt_definition = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    PromptTemplateDefintion = PromptTemplate(input_variables=["infomation"],template=prompt_definition)

    llm = ChatOpenAI(temperature=0,model='gpt-5-mini')
    #langchain syntax
    chain = PromptTemplateDefintion | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
