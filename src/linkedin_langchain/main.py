from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.tools import tool

load_dotenv()


@tool
def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculates the Body Mass Index (BMI) given weight in kilograms and height in meters.
    Arguments:
        weight (float): The weight of the individual in kilograms.
        height (float): The height of the individual in meters.
    Returns:
        float: The calculated BMI value rounded to two decimal places.
    """
    bmi = weight / (height ** 2)
    return round(bmi, 2)



model = init_chat_model(model="gpt-4.1-mini",temperature=0.7)
model_with_tools = model.bind_tools([calculate_bmi])

# historico - chat
messages = [
    HumanMessage(content="imc de uma pessoa com 70 quilos e 175 de altura")
]

response = model_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    result = calculate_bmi.invoke(tool_call)
    messages.append(result)

final = model_with_tools.invoke(messages)
print(final)