import os
import streamlit as st
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator  # model_validator 임포트
import random

# Define the GraphState class
class GraphState(TypedDict):
    question: str  # 질문
    state: str  # categorized의 결과
    answer: str  # Query만

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0.0,
    max_tokens=300,
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)

class Categorized(BaseModel):
    common_conversation: bool = Field(
        description="""사용자가 일반적인 대화를 원하면 True, 그렇지 않으면 False를 대답해줘"""
    )
    SQLQuery: bool = Field(
        description="""사용자가 쿼리문을 요청하거나, 데이터에 대한 정보를 원하면 True, 그렇지 않으면 False를 대답해줘."""
    )

    @model_validator(mode="after")
    def check_only_one_true(cls, values):
        if values.common_conversation == values.SQLQuery:
            raise ValueError("common_conversation과 SQLQuery 중 하나만 True여야 합니다.")
        return values

# Define the behavior_Classification function to check relevance and set the branching condition in "answer"
def behavior_Classification(state: GraphState) -> GraphState:
    # Set up the structured LLM output to use the Categorized schema
    structured_llm = llm.with_structured_output(Categorized)
    response = structured_llm.invoke(state["question"])
    
    # Determine which field is True and use that as the answer
    if response.common_conversation:
        state["state"] = "common_conversation"
    elif response.SQLQuery:
        state["state"] = "SQLQuery"
    
    return state

# Define the llm_answer function to get an answer from the LLM
def common_answer(state: GraphState) -> GraphState:
    state["answer"] = llm.invoke(state["question"])
    return state

# 외부 상태 변수로 toggle 설정
toggle_flag = False

def is_valid_sql_query_text(state: GraphState) -> GraphState:
    global toggle_flag  # 외부 상태 변수를 사용하여 값 교체

    # Toggle 상태에 따라 True 또는 False를 번갈아 할당
    state["answer"] = "True" if toggle_flag else "False"
    toggle_flag = not toggle_flag  # 다음 호출에서 값을 반대로 설정

    return state

def transfer_TextToSQL(state: GraphState) -> GraphState:
    # 프롬프트 템플릿 정의
    template = """
    Question: {question}

    Table Schema
    ch_acq_TB (
        event_date DATE, -- 이벤트 발생 날짜 (예: 20231016)
        source_medium STRING, -- 유저 유입 경로 (UTM 기반) (예: youtube/video)
        campaign STRING, -- 상세 이벤트명 (UTM 기반) (예: EastAfrica_Nov2022)
        content STRING, -- 이벤트 내용 (UTM 기반) (예: girlseducation)
        term STRING, -- 타겟팅 검색어 (예: 유저가 검색했으면 하는 검색어)
        page_location STRING, -- 페이지 URL
        user_id STRING, -- 정기 후원자 ID 또는 일시 후원자 ID (예: 2023102263)
        session STRING, -- GA 세션 ID (예: user_pseudo_id + ga_session_id)
        user STRING, -- 고유한 유저 ID (예: 110417617.169768)
        page_bounce INTEGER, -- 페이지 이탈 횟수 (예: 0, 1, 2)
        begin_checkout INTEGER, -- 후원 신청 페이지 도달 시 발생하는 이벤트 (예: null, 1)
        regular_purchase STRING, -- 정기 후원 여부
        once_purchase STRING, -- 일시 후원 여부
        regular_value INTEGER, -- 정기 후원 금액
        once_value INTEGER -- 일시 후원 금액
    );

    Generate an appropriate SQL query based on the question.
    Don’t hesitate to create subqueries if necessary.
    You can Answer Only SQL query.

    SQL Answer: """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(
        temperature=0.0,
        max_tokens=300,
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    # LLM Chain 객체 생성
    chain = prompt | model | StrOutputParser()
    state["answer"] = chain.invoke({"question": state["question"]})
    return state
# Define the rewrite function to improve the question prompt

def rewrite_question(state: GraphState) -> GraphState:
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant to enhance questions related to customer requirements for data analysis and advertising. "
                "The task involves improving questions based on the following customer objectives and data requirements."
            ),
            (
                "human",
                """
                Here are the key customer requirements:\n"
                "- **Channel Analysis**:\n"
                "  - Step 1: Identifying donation amount, count, and source traffic (Understand data types and relationships)\n"
                "  - Step 2: Budget planning for ads, understanding media offerings (exposure, click-through rate), and conversion to donations.\n"
                "- **Landing Page Analysis**\n"
                "- **Dataset Availability**:\n"
                "  - Planned statistics for all customer data, implementation time constraints.\n"
                "  - Risks include small sample size, possible unintended exposure.\n\n"
                "Given these, the initial question was:\n ------- \n{question}\n ------- \n"
                Analysis를 참고하여 쿼리문을 만들기 위한 좋은 질문을 작성해보세요. 질문은 칼럼값을 잘 참고하게끔 하는게 좋습니다.
                Please be sure to answer the question, and only answer the question Korean.
                """
            ),
        ]
    )

    model = ChatOpenAI(
        temperature=0.0,
        max_tokens=300,
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": question})
    return GraphState(question=response, state = state["state"], answer=state["answer"])

def main():
    st.title("Graph Workflow Execution")

    workflow = StateGraph(GraphState)

    # Add nodes to the workflow
    workflow.add_node("categorized", behavior_Classification)
    workflow.add_node("common", common_answer)
    workflow.add_node("is_valid_sql_query_text", is_valid_sql_query_text)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("transfer_TextToSQL", transfer_TextToSQL)

    # Add conditional edges for the categorized node
    workflow.add_conditional_edges(
            "categorized",
            lambda state: state["state"],  # Use the "answer" key to determine branching
            {
                "common_conversation": "common",
                "SQLQuery": "is_valid_sql_query_text",
            },
        )

    # Add conditional edges for the TextToSQL node to handle query-related responses
    workflow.add_conditional_edges(
            "is_valid_sql_query_text",
            lambda state: state["answer"],
            {
                "True": "transfer_TextToSQL",
                "False": "rewrite_question"
            },
        )
    workflow.add_edge("rewrite_question", "is_valid_sql_query_text")
    workflow.add_edge("transfer_TextToSQL", END)

    # Set entry point
    workflow.set_entry_point("categorized")

    # Compile the workflow
    app = workflow.compile()

    config = RunnableConfig(
        recursion_limit=10, configurable={"thread_id": "CORRECTIVE-RAG"}
    )

    question = st.text_input("Enter a question:", "10월 31일 youtube/video 방문자수.")
    if st.button("Run Workflow"):
        inputs = GraphState(question=question, state="", answer="")
        
        try:
            for output in app.stream(inputs, config=config):
                current_node = next(iter(output.keys()))  # 현재 노드 이름

                # 노드 상태 출력 (1개의 열에 노드 상태를 출력)
                st.markdown(f"### Current Node: {current_node}")

                # 3개의 열로 나누어 Question, State, Answer 출력
                output_areas = st.columns(3)

                # Question 상태 출력
                with output_areas[0]:
                    st.subheader("Question")
                    st.write(output[current_node]["question"])

                # State 상태 출력
                with output_areas[1]:
                    st.subheader("State")
                    st.write(output[current_node]["state"])

                # Answer 상태 출력
                with output_areas[2]:
                    st.subheader("Answer")
                    st.write(output[current_node]["answer"])

                # 각 노드 후 구분선
                st.markdown("---")

        except GraphRecursionError:
            st.info("Recursion limit of 10 reached without hitting a stop condition.")


if __name__ == "__main__":
    main()
