from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # return "The weather in Jagtial is sunny"
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by a agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Popular high-performance Groq model
    temperature=0
)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the Hyderabad on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
