import os
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

# =====================================================
# HUGGING FACE TOKEN
# =====================================================

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# =====================================================
# LOAD MODEL
# =====================================================

endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3
)
llm = ChatHuggingFace(llm=endpoint)


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Agentic AI Code Reviewer",
    page_icon="🤖"
)

st.title("🤖 Agentic AI Code Generator")

st.write(
    """
Generator Agent → Reviewer Agent → Tester Agent
"""
)

# =====================================================
# GRAPH STATE
# =====================================================

class GraphState(TypedDict):

    prompt: str
    generated_code: str
    review_result: str
    test_result: str
    iterations: int


# =====================================================
# AGENT 1
# =====================================================

def code_generator(state):

    prompt = state["prompt"]

    generator_prompt = f"""
Generate Python code.

Task:

{prompt}

Return ONLY executable Python code.
"""

    response = llm.invoke(generator_prompt)

    code = response.content

    return {
        "generated_code": code,
        "iterations": state.get("iterations", 0) + 1
    }


# =====================================================
# AGENT 2
# =====================================================

def code_reviewer(state):

    code = state["generated_code"]

    review_prompt = f"""
Review this code.

Check:

- Syntax
- Logic
- Imports
- Python best practices

Return ONLY:

APPROVED

or

REJECTED: reason

Code:

{code}
"""

    response = llm.invoke(review_prompt)

    return {

        "review_result": response.content

    }


# =====================================================
# AGENT 3
# =====================================================

def code_tester(state):

    code = state["generated_code"]

    try:

        exec_globals = {}

        exec(code, exec_globals)

        result = "TEST PASSED"

    except Exception as e:

        result = f"TEST FAILED: {e}"

    return {

        "test_result": result

    }


# =====================================================
# ROUTING
# =====================================================

MAX_ITERATIONS = 3


def review_decision(state):

    review = state["review_result"]

    if "APPROVED" in review:

        return "approved"

    if state["iterations"] >= MAX_ITERATIONS:

        return "failed"

    return "retry"


# =====================================================
# GRAPH
# =====================================================

workflow = StateGraph(GraphState)

workflow.add_node(
    "generator",
    code_generator
)

workflow.add_node(
    "reviewer",
    code_reviewer
)

workflow.add_node(
    "tester",
    code_tester
)

workflow.set_entry_point(
    "generator"
)

workflow.add_edge(
    "generator",
    "reviewer"
)

workflow.add_conditional_edges(
    "reviewer",
    review_decision,
    {

        "approved": "tester",

        "retry": "generator",

        "failed": END

    }
)

workflow.add_edge(
    "tester",
    END
)

app = workflow.compile()

# =====================================================
# STREAMLIT INPUT
# =====================================================

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Generate Python code to reverse a string"
)

if st.button("Run Agents"):

    if prompt:

        with st.spinner(
            "Agents working..."
        ):

            result = app.invoke({

                "prompt": prompt,

                "iterations": 0

            })

        st.subheader(
            "Generated Code"
        )

        st.code(

            result.get(
                "generated_code",
                ""
            ),

            language="python"

        )

        st.subheader(
            "Reviewer Result"
        )

        st.write(

            result.get(
                "review_result",
                ""
            )

        )

        st.subheader(
            "Tester Result"
        )

        st.success(

            result.get(
                "test_result",
                ""
            )

        )