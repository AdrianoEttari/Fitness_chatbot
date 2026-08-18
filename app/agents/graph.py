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
from app.rag.exercises import search_exercises, ExerciseInfo, MuscleGroup, ExerciseSearchResult

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
    
    exercise_search_results: list[ExerciseSearchResult]
    
    searched_muscle_groups: list[MuscleGroup]
    
    training_plan: TrainingPlan | None # It must be in the state so that will be easier 
    # for the validator to do its work (it will just evaluate the state).
    
    tool_iterations: int
    
    planner_error: str | None
    planner_attempts: int

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
    You are the Training Research Agent.

    Your task is to retrieve exercises from the exercise database
    that can be used to build a personalized training plan.

    IMPORTANT RULES:

    1. When calling search_exercises, you MUST specify exactly ONE
    muscle group.

    2. Never call search_exercises with muscle_group=None.

    3. Never search for all muscle groups at once.

    4. If you need exercises for different muscle groups,
    perform separate tool calls.

    For example:

    search_exercises(
        muscle_group="chest",
        difficulty="intermediate"
    )

    Then, if you need exercises for the back:

    search_exercises(
        muscle_group="back",
        difficulty="intermediate"
    )

    Do NOT do:

    search_exercises(
        muscle_group=None,
        difficulty="intermediate"
    )
    
    5. If a tool returns an error, you MUST analyze the error and try the tool again with corrected arguments when possible. Do not simply explain the error to the user.
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
    The result of the tool, if the model picked any, is appended in the state argument "exercise_search_results".
    """
    # Find the AIMessage that contains the tool call
    tool_call_message = None
    
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
            break
    
    if tool_call_message is None:
        return {}
    
    # Take every id of each tool in AIMessage
    tool_call_ids = {
        call["id"]
        for call in tool_call_message.tool_calls
    }
    
    tool_messages = [
        message
        for message in state["messages"]
        if (
            isinstance(message, ToolMessage)
            and message.tool_call_id in tool_call_ids
        )    
    ]
    
    new_search_results = []
    
    for tool_message in tool_messages:
        if tool_message.status == "error":
            continue
    
        # Find the tool call corresponding to this ToolMessage
        # next(generator, None) returns the first element of the generator that satisfies the if condition and if there are no other elements it returns None. Notice that we're looping for each tool message, so each tool_message will be taken at the end, even if tool_call considers just one of them.
        tool_call = next(
            (
                call
                for call in tool_call_message.tool_calls
                if call["id"] == tool_message.tool_call_id
            ),
            None,
        )

        if tool_call is None:
            continue

        muscle_group = tool_call["args"]["muscle_group"]
        difficulty = tool_call["args"].get("difficulty")

        exercises_data = json.loads(tool_message.content)
        
        new_exercises = [
            ExerciseInfo.model_validate(exercise)
            for exercise in exercises_data
        ]
    
        search_result = ExerciseSearchResult(
            muscle_group=MuscleGroup(muscle_group),
            difficulty=difficulty,
            exercises=new_exercises
        )
        new_search_results.append(search_result)

    updated_database = [
            *state["exercise_search_results"],
            *new_search_results,
        ]
    return {
        "exercise_search_results": updated_database,
        "tool_iterations": state["tool_iterations"]+1,
    }
    
# ============================================================
# 7. PLANNER NODE
# ============================================================

structured_llm = llm.with_structured_output(TrainingPlan)

def planner_node(state: TrainingState):
    user = state["user"]
    exercises = state["exercise_search_results"]
    previous_error = state["planner_error"]
    attempt = state["planner_attempts"] + 1

    error_feedback=""
    if previous_error is not None:
        error_feedback = f"""
        IMPORTANT:
        A previous attempt to generate the training plan was rejected.

        The validation error was:
        {previous_error}

        You MUST correct this error in the new plan.
        """
        

    PLANNER_SYSTEM_PROMPT = f"""
    You are the training plan generator.

    Your task is to create a personalized training plan
    using the information available in the conversation.

    IMPORTANT RULES:
    * The number of workout days must be equal to {user.training_days_per_week}.
    * You MUST ONLY use exercises from this list, but you don't have to use them all: {exercises}
    * Do not use rest days! just workout days. 
    * The output must strictly follow the TrainingPlan schema.
    
    A workout day MAY contain exercises targeting multiple related muscle groups.

    You should combine complementary muscle groups when appropriate.
    For example:
    - chest + triceps
    - back + biceps
    - shoulders + triceps
    - quadriceps + hamstrings + glutes

    Do not assume that one muscle group must correspond to exactly one workout day.
    
    Moreover, you plan must be suited for a user with the following features:
    - Age: {user.age}
    - Height: {user.height_cm} cm
    - Weight: {user.weight_kg} kg
    - Goal: {user.goal}
    - Experience: {user.experience}
    - Training days per week: {user.training_days_per_week}
    - Training duration: {user.training_duration_minutes} minutes
    - Injuries or limitations: {user.injuries_or_limitations}


    Do not add explanations outside the structured output.
    
    {error_feedback}
    """
    
    planner_messages = [
        SystemMessage(
            content = PLANNER_SYSTEM_PROMPT
        ),
        *state["messages"],
    ]
    
    try:
        response = structured_llm.invoke(
            planner_messages
        )
        return {
            "training_plan": response,
            "planner_error": None,
            "planner_attempts": attempt,
            }
    except Exception as e:
        print("\n=== PLANNER VALIDATION ERROR ===")
        print(e)
        print("================================\n")

        return {
            "training_plan": None,
            "planner_error": str(e),
            "planner_attempts": attempt,
        }
    
MAX_PLANNER_ATTEMPTS=5    
    
def should_retry_planner(state: TrainingState):
    if state["planner_error"] is None:
        return "validate"
    
    if state["planner_attempts"] >= MAX_PLANNER_ATTEMPTS:
        print("\nPlanner failed after maximum attempts.")
        return "validate"
    
    return "retry"

def validate_training_plan(state: TrainingState):
    training_plan = state["training_plan"]
    search_results = state["exercise_search_results"]

    if training_plan is None:
        return {
            "planner_error": "Training plan is missing."
        }

    # Collect all exercise names returned by the tools
    available_exercises = {
        exercise.name
        for search_result in search_results
        for exercise in search_result.exercises
    }

    invalid_exercises = []

    for workout in training_plan.workouts:
        for exercise in workout.exercises:
            if exercise.name not in available_exercises:
                invalid_exercises.append(exercise.name)
    
    # There is no need of writing state["planner_error"]=...
    # The reason is that I put in the function argument state: TrainingState
    # So, the final dictionary will be the update of the state.
    if invalid_exercises:
        return {
            "planner_error": (
                "The following exercises were not found in the "
                f"exercise database: {invalid_exercises}"
            )
        }

    return {
        "planner_error": None
    }

def validation_result(state: TrainingState):
    if state["planner_error"] is None:
        return "end"
    
    if state["planner_attempts"] >= MAX_PLANNER_ATTEMPTS:
        print("\nPlanner failed after maximum attempts.")
        return "end"
    
    return "retry"
        
# ============================================================
# 8. ROUTING LOGIC
# ============================================================

def should_continue(state: TrainingState):
    """
    Decides whether the graph should execute tools
    or terminate.
    """
    MAX_TOOL_ITERATIONS = 5
    
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

graph_builder.add_node(
    "validate_training_plan",
    validate_training_plan
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

graph_builder.add_conditional_edges(
    "planner",
    should_retry_planner,
    {
        "retry":"planner",
        "validate": "validate_training_plan",
    }
)

graph_builder.add_conditional_edges(
    "validate_training_plan",
    validation_result,
    {
        "retry":"planner",
        "end": END
    }
)


# ============================================================
# 10. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()

# ============================================================
# 11. SIMPLE TEST FUNCTION
# ============================================================

def run_graph(user_input: str, user: UserProfile,
              training_plan: TrainingPlan, exercise_search_results: list,
              searched_muscle_groups: list, tool_iterations: int, planner_error: str,
              planner_attempts: int):

    result = graph.invoke(
        {   
            "user": user,
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],
            "exercise_search_results": exercise_search_results,
            "searched_muscle_groups": searched_muscle_groups,
            "training_plan": training_plan,
            "tool_iterations":tool_iterations,
            "planner_error": planner_error,
            "planner_attempts": planner_attempts
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
    exercise_search_results = []
    searched_muscle_groups = []
    tool_iterations=0
    planner_error = None
    planner_attempts=0
    
    result = run_graph(
        # "Give me intermediate exercises for the chest.",
        "I am an intermediate athlete and I want to build muscle. I need a 4-day workout plan.",
        # "Hello. I would like a chest routine for my chest day workout.",
        user,
        training_plan,
        exercise_search_results,
        searched_muscle_groups,
        tool_iterations,
        planner_error,
        planner_attempts
        
    )
    print(result["training_plan"])
    print(type(result["training_plan"]))
    
    # for message in result["messages"]:
    #     print("\n--------------------")
    #     print(type(message).__name__)
    #     print(message)
    breakpoint()
    print(result)