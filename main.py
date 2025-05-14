import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

#llm = ChatOpenAi(model = "gpt-4o-mini")
#
groq_api_key = os.getenv("CHATGROQ_API_KEY")
llm = ChatGroq(
    model="llama3-8b-8192",  # or mixtral-8x7b or gemma-7b-it
    api_key=groq_api_key
)

# Call the LLM
#response = llm.invoke("What is life?")
#print(response.content)
#prompt  template
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) 
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query":"what is capital of France?"})
print(raw_response)

try:
  structured_response = parser.parse(raw_response.get("output"))
  print(structured_response.topic)
except:
   print("error parsing response" ,e,"Raw respnse",raw_response)