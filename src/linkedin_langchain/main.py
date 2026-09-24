from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

model = init_chat_model(model="gpt-4o-mini",temperature=0.7)

# historico - chat
messages = [
    # instruções, role prompt, etc
    SystemMessage(content="Você é um assistente que fala português e responde em forma de poemas japanês haiku"),
    # prompt do usuário
    HumanMessage(content="Como ser mais produtivo?")
]

response = model.invoke(messages)

print(response)