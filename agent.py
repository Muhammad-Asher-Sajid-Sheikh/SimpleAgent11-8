import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langchain_core.agents import AgentExecutor, create_tool_calling_agent

@tool
def get_Current_Time() -> str:
    '''
    This tool can be used to get the current date time in string form.
    '''
    return datetime.now()


@tool
def calculate_Area(length: float, width : float) -> float:
    '''Calculates the area using length and width'''
    return length * width


@tool
def reverse_string(text : str) -> str:
    '''Reverses the given text string'''
    return text[::-1]


tools = [get_Current_Time, calculate_Area, reverse_string]
tool_map = {tool.name : tool for tool in tools}

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature = 0.3,
    google_api_key = 'Your api key'
)


# llm_with_tools = llm.bind_tools(tools)

query = (
    "what is the area of rectangle with the length of 7m and width of 4m"
)

messages = [HumanMessage(content=query)]

# msg -> llm -> response -> msg [previous msgs resp]

# PT

# response = llm_with_tools.invoke(messages)
# messages.append(response)

# print(response)


# if response.tool_calls:

#     for tool_Call in response.tool_calls:
#         tool_name = tool_Call['name']
#         tool_arg = tool_Call['args']
#         tool_id = tool_Call['id']

#         selectedtool = tool_map[tool_name]
#         tool_results = selectedtool.invoke(tool_arg)

#         print(tool_results)

#         messages.append(ToolMessage(content = str(tool_results), tool_call_id = tool_id))


agent = create_tool_calling_agent(llm, tools, promptTemplate)
agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose = True)

response = agentExecutor.invoke({"input" : query})
