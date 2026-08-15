from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, ToolMessage
#### If langchain_ollama import doesn't work then write in terminal: 
# echo $SSL_CERT_FILE
# python -c "import os; p=os.environ.get('SSL_CERT_FILE'); print('SSL_CERT_FILE =', p); print('exists =', os.path.exists(p) if p else None)"
# unset SSL_CERT_FILE
# python -c "from langchain_ollama import ChatOllama; print('LangChain Ollama import: OK')"
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import json
from app.rag.exercises import search_exercises, ExerciseInfo

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
    
    exercise_database: list[ExerciseInfo]
    
    training_plan: TrainingPlan | None # It must be in the state so that will be easier 
    # for the validator to do its work (it will just evaluate the state).
    
    tool_iterations: int

# ============================================================
# 2. LLM
# ============================================================

llm = ChatOllama(
    model="mistral-nemo:latest",
    # model = "mistral:latest",
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
    TRAINING_SYSTEM_PROMPT = """
    You are the Training Agent in a multi-agent fitness planning system.

    Your role is to gather and reason about the information required
    to create a personalized training plan.

    You have access to tools containing exercise information.

    Use the tools whenever you need reliable information about exercises.

    Your output is NOT the final training plan.

    Do NOT:
    - generate a workout plan;
    - list a complete workout;
    - prescribe sets, reps or rest periods unless this information
    is required by a tool call;
    - provide the final answer to the user.

    When using search_exercises:
    - provide muscle_group only when a specific muscle group is required;
    - if no specific muscle group is required, omit muscle_group;
    - never use "all" as a muscle_group.

    When you have gathered enough information, simply indicate that
    the information is ready for the planner.

    The Planner Agent is responsible for generating the final
    TrainingPlan.
    """

    messages = [
        TRAINING_SYSTEM_PROMPT,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

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
# 6. PROCESS TOOL NODE
# save the result of the tool node in the state
# ============================================================

def process_tool_results(state: TrainingState):
    """
    The result of the tool, if the model picked any, is appended in the state argument "exercise_database".
    """
    
    last_message = state["messages"][-1]

    if not isinstance(last_message, ToolMessage):
        return {}

    exercises_data = json.loads(last_message.content)
    
    new_exercises = [
        ExerciseInfo.model_validate(exercise)
        for exercise in exercises_data
    ]
    
    updated_database = [
            *state["exercise_database"],
            *new_exercises
        ]
    # print("\n===EXERCISE DATABASE===")
    # for exercise in updated_database:
    #     print("-", exercise)
    # print("="*20)
    return {
        "exercise_database": updated_database,
        "tool_iterations": state["tool_iterations"]+1,
    }
    
# ============================================================
# 7. PLANNER NODE
# ============================================================

structured_llm = llm.with_structured_output(TrainingPlan)

def planner_node(state: TrainingState):
    user = state["user"]
    
    exercises = state["exercise_database"]
    
    PLANNER_SYSTEM_PROMPT = f"""
    You are the training plan generator.

    Your task is to create a personalized training plan
    using the information available in the conversation.
    
    The following exercises have been retrieved from the exercise database.

    You MUST ONLY use exercises from this list:

    {exercises}
    
    Moreover, you plan must be suited for a user with the following features:
    - Age: {user.age}
    - Height: {user.height_cm} cm
    - Weight: {user.weight_kg} kg
    - Goal: {user.goal}
    - Experience: {user.experience}
    - Training days per week: {user.training_days_per_week}
    - Training duration: {user.training_duration_minutes} minutes
    - Injuries or limitations: {user.injuries_or_limitations}
    
    The output must strictly follow the TrainingPlan schema.

    Do not add explanations outside the structured output.
    """
    
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
# 8. ROUTING LOGIC
# ============================================================
MAX_TOOL_ITERATIONS = 5

def should_continue(state: TrainingState):
    """
    Decides whether the graph should execute tools
    or terminate.
    """

    last_message = state["messages"][-1]

    if state["tool_iterations"] >= MAX_TOOL_ITERATIONS:
        return "planner"
    
    if last_message.tool_calls:
        return "tools"

    return "planner"


# ============================================================
# 9. GRAPH DEFINITION
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

graph_builder.add_node(
    "process_tool_results",
    process_tool_results,
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

graph_builder.add_edge(
    "tools",
    "process_tool_results",
)

graph_builder.add_edge(
    "process_tool_results",
    "agent",
)

graph_builder.add_edge(
    "planner",
    END,
)

# ============================================================
# 10. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()

# ============================================================
# 11. SIMPLE TEST FUNCTION
# ============================================================

def run_graph(user_input: str, user: UserProfile,
              training_plan: TrainingPlan, exercise_database: list,
              tool_iterations: int):

    result = graph.invoke(
        {   
            "user": user,
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "exercise_database": exercise_database,
            "training_plan": training_plan,
            "tool_iterations":tool_iterations
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
    exercise_database = []
    tool_iterations=0
    
    result = run_graph(
        # "Give me intermediate exercises for the chest.",
        "I am an intermediate athlete and I want to build muscle. I need a 4-day workout plan.",
        # "Hello. I would like a chest routine for my chest day workout.",
        user,
        training_plan,
        exercise_database,
        tool_iterations
        
    )
    print(result["training_plan"])
    print(type(result["training_plan"]))
    
    for message in result["messages"]:
        print("\n--------------------")
        print(type(message).__name__)
        print(message)
    breakpoint()
    print(result)