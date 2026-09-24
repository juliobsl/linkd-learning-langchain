from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)
agent = create_agent(model=model, checkpointer=InMemorySaver())

config = {
    "configurable": {"thread_id": "user-123"}
}

response = agent.invoke(
    {"messages": [HumanMessage(content="Oi, meu nome é julio")]},
    config=config
)
print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [HumanMessage(content="Qual é o meu nome?")]},
    config=config
)
print(response["messages"][-1].content)