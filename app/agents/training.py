from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.domain.training import TrainingPlan
from app.domain.user import UserProfile
from app.rag.exercises import search_exercises

    
    
def run_training_agent(
    user: UserProfile,
) -> list:

    llm = ChatOllama(
        model="gpt-oss:latest",
        temperature=0.5,
    )

    tools = [
        search_exercises,
    ]

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    llm_with_tools = llm.bind_tools(tools)

    agent_prompt = f"""
    You are a fitness training research agent.

    Your job is to gather the information needed to create
    a personalized workout plan for the user.

    User profile:

    {user.model_dump_json(indent=2)}

    You have access to tools containing information about exercises.

    Use a tool when you need information that is not available
    in the current conversation.

    You are NOT required to use a tool.

    If you already have enough information, do not call any tool
    and finish your analysis.

    Do not invent information that could be obtained from a tool.
    """

    messages = [
        HumanMessage(
            content=agent_prompt
        )
    ]

    tool_results = []

    MAX_ITERATIONS = 5

    for _ in range(MAX_ITERATIONS):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # The LLM decided that no more tools are necessary.
        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:

            tool = tools_by_name[
                tool_call["name"]
            ]

            result = tool.invoke(
                tool_call["args"]
            )

            tool_results.append(result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return tool_results

def generate_training_plan(
    user: UserProfile,
    tool_results: list,
) -> TrainingPlan:

    llm = ChatOllama(
        model="gpt-oss:latest",
        temperature=0.5,
    )

    structured_llm = llm.with_structured_output(
        TrainingPlan
    )
        
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert fitness programming assistant.

                Create a personalized workout plan for the user.

                Use the information retrieved by the training research agent.

                Do not invent exercises when suitable exercises are
                available in the retrieved information.

                Return only a valid TrainingPlan.
                """
            ),
            (
                "human",
                """
                USER PROFILE:

                {user}

                RETRIEVED INFORMATION:

                {tool_results}
                """
            ),
        ]
    )
    
    chain = prompt | structured_llm
    
    return chain.invoke(
        {
            "user": user.model_dump_json(
                indent=2
            ),
            "tool_results": str(tool_results),
        }
    )
    
def create_training_plan(
    user: UserProfile,
) -> TrainingPlan:

    tool_results = run_training_agent(
        user
    )

    training_plan = generate_training_plan(
        user,
        tool_results,
    )

    return training_plan