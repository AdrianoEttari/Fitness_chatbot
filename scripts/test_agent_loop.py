from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from app.rag.exercises import search_exercises


llm = ChatOllama(
    model="gpt-oss:latest",
    temperature=0.5,
)

tools = [search_exercises]

llm_with_tools = llm.bind_tools(tools)

tools_by_name = {
    tool.name: tool
    for tool in tools
}


messages = [
    HumanMessage(
        content="I need intermediate exercises for the chest."
    )
]


# --------------------------------------------------
# STEP 1: Ask the LLM what to do
# --------------------------------------------------

response = llm_with_tools.invoke(messages)

messages.append(response)


print("FIRST LLM RESPONSE:")
print(response)

print("\nTOOL CALLS:")
print(response.tool_calls)


# --------------------------------------------------
# STEP 2: Execute the requested tools
# --------------------------------------------------

for tool_call in response.tool_calls:

    tool = tools_by_name[tool_call["name"]]

    result = tool.invoke(tool_call["args"])

    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
    )

    messages.append(tool_message)


# --------------------------------------------------
# STEP 3: Give the tool result back to the LLM
# --------------------------------------------------

final_response = llm_with_tools.invoke(messages)


print("\nFINAL RESPONSE:")
print(final_response.content)