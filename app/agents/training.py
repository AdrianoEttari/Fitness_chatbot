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


def create_training_plan(user: UserProfile) -> TrainingPlan:

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

    messages = [
        HumanMessage(
            content=f"""
            Create a personalized workout plan for this user:

            {user.model_dump_json(indent=2)}

            You have access to tools that can provide information
            about available exercises.

            Use the tools whenever you need additional information.
            If you already have enough information, do not use any tool.
            """
        )
    ]

    # -----------------------------------------
    # Agent loop
    # -----------------------------------------
    MAX_ITERATIONS = 5 # otherwise the model my loop on the same tool forever
    
    for _ in range(MAX_ITERATIONS):

        response = llm_with_tools.invoke(messages)

        # messages.append(response)

        if not response.tool_calls:
            break
        
        tool_results = []
        
        # Execute requested tools
        for tool_call in response.tool_calls:

            tool = tools_by_name[tool_call["name"]]

            result = tool.invoke(
                tool_call["args"]
            )
            
            tool_results.append(result)
            
            # messages.append(
            #     ToolMessage(
            #         content=str(result),
            #         tool_call_id=tool_call["id"],
            #     )
            # )

    # -----------------------------------------
    # Generate structured TrainingPlan
    # -----------------------------------------

    structured_llm = llm.with_structured_output(
        TrainingPlan
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert fitness programming assistant.

                Create a personalized training plan using the
                information gathered during the previous analysis.

                Return only the requested structured TrainingPlan.
                """
            ),
            (
                "human",
                """
                User profile:

                {user}

                Information gathered by the agent:

                {context}
                """
            ),
        ]
    )

    chain = final_prompt | structured_llm

    return chain.invoke(
        {
            "user": user.model_dump_json(indent=2),
            "context": str(tool_results),
        }
    )