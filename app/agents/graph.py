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

from langchain_core.messages import HumanMessage, SystemMessage
from app.domain.user import UserProfile, FitnessGoal, ExperienceLevel
from app.domain.training import TrainingPlan

# ============================================================
# 1. STATE
# ============================================================

class TrainingState(TypedDict):
    user: UserProfile

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]
    
    training_plan: TrainingPlan | None # It must be in the state so that will be easier 
    # for the validator to do its work (it will just evaluate the state).

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

def agent_node(state: TrainingState):

    user = state["user"]

    system_message = SystemMessage(
        content=f"""
        You are the Training Agent of a fitness planning system.

        - Age: {user.age}
        - Height: {user.height_cm} cm
        - Weight: {user.weight_kg} kg
        - Goal: {user.goal}
        - Experience: {user.experience}
        - Training days per week: {user.training_days_per_week}
        - Training duration: {user.training_duration_minutes} minutes
        - Injuries or limitations: {user.injuries_or_limitations}

        You are responsible for creating workout plans.
        
        You have access to tools that contain information
        about exercises.
        Use the available tools when you need exercise information.
        Do not invent exercise information when it can be retrieved
        from the knowledge base.
        """
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    # response = llm_with_tools.invoke(messages)
    response = llm.invoke(messages)

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
# 6. PLANNER NODE
# ============================================================

structured_llm = llm.with_structured_output(TrainingPlan)

PLANNER_SYSTEM_PROMPT = """
    You are the training plan generator.

    Your task is to create a personalized training plan
    using the information available in the conversation.

    The output must strictly follow the TrainingPlan schema.

    Do not add explanations outside the structured output.
    """

def planner_node(state: TrainingState):
    
    planner_messages = [
        SystemMessage(
            content = PLANNER_SYSTEM_PROMPT
        ),
        *state["messages"],
    ]
    
    
    response = structured_llm.invoke(
        planner_messages
    )
    
    return {
        "training_plan": response
    }

# ============================================================
# 7. ROUTING LOGIC
# ============================================================

def should_continue(state: TrainingState):
    """
    Decides whether the graph should execute tools
    or terminate.
    """

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "planner"


# ============================================================
# 8. GRAPH DEFINITION
# ============================================================

graph_builder = StateGraph(TrainingState)


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

graph_builder.add_node(
    "planner",
    planner_node,
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
#   └── no tool   → planner

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "planner": "planner",
    },
)


# After executing tools, go back to the agent
graph_builder.add_edge(
    "tools",
    "agent",
)

graph_builder.add_edge(
    "planner",
    END,
)

# ============================================================
# 9. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()

# ============================================================
# 10. SIMPLE TEST FUNCTION
# ============================================================

def run_graph(user_input: str, user: UserProfile, training_plan: TrainingPlan):

    result = graph.invoke(
        {   
            "user": user,
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "training_plan": training_plan,
        }
    )

    return result


if __name__ == "__main__":
    user = UserProfile(
        age=28,
        height_cm=180,
        weight_kg=80,
        goal=FitnessGoal.MUSCLE_GAIN,
        experience=ExperienceLevel.INTERMEDIATE,
        training_days_per_week=4,
        training_duration_minutes=60,
        dietary_preferences=["omnivore"],
        dietary_restrictions=[],
        injuries_or_limitations=[],
    )

    training_plan = None
    
    result = run_graph(
        # "Give me intermediate exercises for the chest.",
        "Hello, how are you?",
        # "Hello. I would like a chest routine for my chest day workout.",
        user,
        training_plan
        
    )

    for message in result["messages"]:
        print("\n--------------------")
        print(type(message).__name__)
        print(message)