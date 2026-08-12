from typing import List

from pydantic import BaseModel

from magentic import (
    prompt,
    OpenaiChatModel
)

from autogen_agentchat.agents import AssistantAgent

from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

from config import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    TOP_K_RESULTS
)



groq_magentic_model = OpenaiChatModel(
    LLM_MODEL_NAME,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


class Flashcard(BaseModel):
    question: str
    answer: str


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_option_index: int
    explanation: str



@prompt(
    """
    Create {count} study flashcards based ONLY on the
    following study material.

    Requirements:
    - Create exactly {count} flashcards.
    - Focus on important concepts, definitions and key points.
    - Do not invent information outside the provided material.
    - Make the questions useful for exam preparation.
    - Avoid duplicate questions.

    Study material:

    {context}
    """,
    model=groq_magentic_model
)
def generate_flashcards_with_magentic(
    context: str,
    count: int = 6
) -> List[Flashcard]:
    ...


@prompt(
    """
    Create {count} multiple-choice questions based ONLY on
    the following study material.

    Requirements:

    - Create exactly {count} questions.
    - Each question must have exactly 4 options.
    - Only one option should be correct.
    - Do not reveal the correct answer in the question.
    - Make questions useful for exam preparation.
    - Cover different concepts from the material.
    - Avoid duplicate or nearly identical questions.
    - Set correct_option_index to the zero-based index
      of the correct option.
    - Provide a clear explanation for the correct answer.
    - Do not use information that is not present in the
      provided study material.

    Study material:

    {context}
    """,
    model=groq_magentic_model
)
def generate_quiz_with_magentic(
    context: str,
    count: int = 10
) -> List[QuizQuestion]:
    ...



def create_study_agent(vector_store):
    """
    Creates an AutoGen study assistant that has access
    to the uploaded PDF through a FAISS search tool.
    """


    def search_pdf_notes(
        query: str
    ) -> str:
        """
        Searches the uploaded study PDF for relevant
        information using FAISS.
        """

        docs = vector_store.similarity_search(
            query,
            k=TOP_K_RESULTS
        )


        if not docs:
            return (
                "No relevant information was found "
                "in the uploaded study material."
            )


        return "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )



    model_client = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME,

        api_key=GROQ_API_KEY,

        base_url="https://api.groq.com/openai/v1",

        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },

        # Groq does not support the name field
        include_name_in_message=False
    )

    agent = AssistantAgent(
        name="Study_Assistant",

        model_client=model_client,

        tools=[
            search_pdf_notes
        ],

        system_message=(
            "You are an AI Study Assistant. "

            "You have access to the user's uploaded "
            "study material through the search_pdf_notes tool. "

            "When the user's question is related to the "
            "uploaded study material, use the "
            "search_pdf_notes tool to retrieve relevant "
            "information before answering. "

            "Base answers about the uploaded study material "
            "on the retrieved content. "

            "Do not invent information that is not supported "
            "by the retrieved study material. "

            "Answer clearly, accurately and concisely."
        )
    )


    return agent



def create_general_agent():
    """
    Creates a general-purpose AutoGen agent.

    This agent is used when the user has not uploaded
    a PDF yet.
    """

    model_client = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME,

        api_key=GROQ_API_KEY,

        base_url="https://api.groq.com/openai/v1",

        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },

        include_name_in_message=False
    )


    agent = AssistantAgent(
        name="General_Assistant",

        model_client=model_client,

        system_message=(
            "You are a helpful AI assistant. "

            "Answer general questions clearly and accurately. "

            "For calculations, programming questions, "
            "definitions, general knowledge and explanations, "
            "provide a direct and useful answer."
        )
    )


    return agent



async def run_agent_query(
    agent,
    user_query: str
) -> str:
    """
    Runs an AutoGen agent asynchronously.

    If Groq fails while trying to generate a tool call,
    return a friendly message instead of exposing the
    technical traceback to the user.
    """

    try:

        result = await agent.run(
            task=user_query
        )


        return result.messages[-1].content


    except Exception as e:

        error_message = str(e)


        # Handle Groq tool-call failure
        if "tool_use_failed" in error_message:

            return (
                "⚠️ **I couldn't find this information "
                "in your study material.**\n\n"
                "Please ask a question related to "
                "the uploaded PDF."
            )


        # Handle other errors
        return (
            "⚠️ **Sorry, I couldn't process your question "
            "right now.**\n\n"
            "Please try again."
        )