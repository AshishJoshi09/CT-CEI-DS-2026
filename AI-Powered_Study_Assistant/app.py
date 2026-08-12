import asyncio

import streamlit as st

from document_processor import (
    process_uploaded_pdf
)

from vector_store import (
    create_vector_store
)

from agent_system import (
    create_study_agent,
    create_general_agent,
    run_agent_query,
    generate_flashcards_with_magentic,
    generate_quiz_with_magentic
)



st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="",
    layout="wide"
)



if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "agent" not in st.session_state:
    st.session_state.agent = None


if "messages" not in st.session_state:
    st.session_state.messages = []


if "quiz_items" not in st.session_state:
    st.session_state.quiz_items = None



with st.sidebar:

    st.header("Study Material")


    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )


    if uploaded_file and st.button(
        "Process Document"
    ):

        with st.spinner(
            "Indexing PDF with FAISS..."
        ):


            chunks = process_uploaded_pdf(
                uploaded_file
            )

            vector_store = create_vector_store(
                chunks
            )

            agent = create_study_agent(
                vector_store
            )

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.agent = (
                agent
            )


            st.session_state.messages = []

            st.session_state.quiz_items = None


            st.success(
                f"Processed successfully! "
                f"{len(chunks)} text chunks created."
            )


st.title(
    "TeachYou"
)


st.caption(
    " AI-Powered Study Assistant"
)



tab_chat, tab_flashcards, tab_quiz = st.tabs(
    [
        "Agent Chat",
        "Flashcards",
        "Practice Quiz"
    ]
)


with tab_chat:

    st.subheader(
        "Ask Your Study Assistant"
    )


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    if user_question := st.chat_input(
        "Ask a question about your study material..."
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )


        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        if st.session_state.agent is None:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Groq is thinking..."
                ):

                    try:

                        general_agent = (
                            create_general_agent()
                        )


                        response = asyncio.run(
                            run_agent_query(
                                general_agent,
                                user_question
                            )
                        )


                        st.markdown(
                            response
                        )


                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response
                            }
                        )


                    except Exception:

                        st.error(
                            "Sorry, I couldn't process "
                            "your question."
                        )


        else:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching your study material..."
                ):

                    response = asyncio.run(
                        run_agent_query(
                            st.session_state.agent,
                            user_question
                        )
                    )


                    st.markdown(
                        response
                    )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )



with tab_flashcards:

    st.subheader(
        "Generate Structured Flashcards"
    )


    if st.session_state.vector_store is None:

        st.info(
            "Upload and process a PDF first "
            "to generate flashcards."
        )


    else:

        if st.button(
            "Generate Flashcards"
        ):

            with st.spinner(
                "Magentic is generating flashcards..."
            ):


                docs = (
                    st.session_state
                    .vector_store
                    .similarity_search(
                        "important concepts definitions "
                        "key points summary",
                        k=6
                    )
                )


                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )


                cards = (
                    generate_flashcards_with_magentic(
                        context,
                        count=6
                    )
                )


                for i, card in enumerate(
                    cards,
                    1
                ):

                    with st.expander(
                        f"Flashcard {i}: "
                        f"{card.question}"
                    ):

                        st.write(
                            f"**Answer:** "
                            f"{card.answer}"
                        )



with tab_quiz:

    st.subheader(
        "Practice Quiz"
    )


    if st.session_state.vector_store is None:

        st.info(
            "Upload and process a PDF first "
            "to generate a practice quiz."
        )


    else:

        if st.button(
            "Generate Practice Quiz"
        ):

            with st.spinner(
                "Magentic is generating your quiz..."
            ):


                docs = (
                    st.session_state
                    .vector_store
                    .similarity_search(
                        "main topics concepts definitions "
                        "important points",
                        k=8
                    )
                )

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )

                quiz_items = (
                    generate_quiz_with_magentic(
                        context,
                        count=10
                    )
                )

                st.session_state.quiz_items = (
                    quiz_items
                )


        if st.session_state.quiz_items:

            st.markdown(
                "### Test Your Knowledge"
            )


            st.info(
                "Select an answer and submit it "
                "to see whether you are correct."
            )


            for idx, question in enumerate(
                st.session_state.quiz_items,
                1
            ):


                st.markdown(
                    f"### Q{idx}. "
                    f"{question.question}"
                )

                user_choice = st.radio(
                    "Choose your answer:",
                    question.options,
                    index=None,
                    key=f"quiz_q_{idx}"
                )

                if st.button(
                    "Submit Answer",
                    key=f"btn_q_{idx}"
                ):


                    if user_choice is None:

                        st.warning(
                            "Please select an answer first."
                        )


                    else:

                        correct_answer = (
                            question.options[
                                question.correct_option_index
                            ]
                        )


                        if user_choice == correct_answer:

                            st.success(
                                "✅ Correct!"
                            )


                        else:

                            st.error(
                                "Incorrect."
                            )


                            st.write(
                                f"**Correct answer:** "
                                f"{correct_answer}"
                            )


                        # -----------------------------------------
                        # EXPLANATION
                        # -----------------------------------------

                        st.info(
                            f"**Explanation:** "
                            f"{question.explanation}"
                        )


                st.divider()


st.markdown("---")


st.caption(
    "TeachYou , learn your way"
)