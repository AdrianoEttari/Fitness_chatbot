from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
#### If langchain_ollama import doesn't work then write in terminal: 
# echo $SSL_CERT_FILE
# python -c "import os; p=os.environ.get('SSL_CERT_FILE'); print('SSL_CERT_FILE =', p); print('exists =', os.path.exists(p) if p else None)"
# unset SSL_CERT_FILE
# python -c "from langchain_ollama import ChatOllama; print('LangChain Ollama import: OK')"
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.rag.exercises import search_exercises

from langchain_core.messages import HumanMessage

# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    """
    In the state there are all the information that are shared among the nodes in the graph.
    """
    messages: Annotated[
        list[AnyMessage],
        add_messages, # add_messages è un reducer (it says to LangGraph how to save the new messages:
        # old messages + new messages -> updated messages.)
    ]


# ============================================================
# 2. LLM
# ============================================================

llm = ChatOllama(
    model="mistral-nemo:latest",
    temperature=0.5,
)


# ============================================================
# 3. TOOLS
# ============================================================

tools = [
    search_exercises,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4. AGENT NODE
# ============================================================

def agent_node(state: AgentState):
    """
    Calls the LLM using the current conversation state.
    """

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 5. TOOL NODE
# ToolNode identifies the tool, extracts the arguments of the tool, executes the tool and builds the tool message.
# ToolNode è un nodo predefinito di LangGraph che gestisce l'esecuzione delle tool call prodotte dall'LLM e restituisce i relativi risultati nello State.
# ============================================================

tool_node = ToolNode(tools) 


# ============================================================
# 6. ROUTING LOGIC
# ============================================================

def should_continue(state: AgentState):
    """
    Decides whether the graph should execute tools
    or terminate.
    """

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# ============================================================
# 7. GRAPH DEFINITION
# ============================================================

graph_builder = StateGraph(AgentState)


# Nodes
# The first argument in the .add_node() is the name, the second argument is the function 
# that will be executed when the graph reaches that node.
graph_builder.add_node(
    "agent",
    agent_node,
)

graph_builder.add_node(
    "tools",
    tool_node,
)


# Entry point
graph_builder.set_entry_point(
    "agent"
)


# Conditional edge:
#
# agent
#   │
#   ├── tool call → tools
#   │
#   └── no tool   → END

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)


# After executing tools, go back to the agent
graph_builder.add_edge(
    "tools",
    "agent",
)


# ============================================================
# 8. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()

# ============================================================
# 9. SIMPLE TEST FUNCTION
# ============================================================

def run_graph(user_input: str):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        }
    )

    return result


if __name__ == "__main__":

    result = run_graph(
        "Give me intermediate exercises for the chest."
        # "Hello, how are you?"
    )

    for message in result["messages"]:
        print("\n--------------------")
        print(type(message).__name__)
        print(message)