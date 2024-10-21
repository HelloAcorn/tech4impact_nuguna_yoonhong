from openai import AsyncOpenAI
import asyncio
import re
import sqlite3
import time  # Import the time module

# OpenAI API 키 설정
client = AsyncOpenAI()

async def text_to_sql_query(question):
    # LLM에 전달할 프롬프트 생성
    instruction = """
    You are an expert database administrator.
    Given a natural language question and the table schema, generate an SQL query that answers the question.

    Table Schema:
    CREATE TABLE campaign_board (
        campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,        -- 캠페인 고유 ID
        title TEXT NOT NULL,                                  -- 캠페인 제목
        description TEXT,                                     -- 캠페인 설명
        created_by TEXT NOT NULL,                             -- 작성자 (캠페인 등록자)
        start_date DATE,                                      -- 캠페인 시작일
        end_date DATE,                                        -- 캠페인 종료일
        total_funds REAL DEFAULT 0,                           -- 모금된 총 금액
        view_count INTEGER DEFAULT 0,                         -- 게시글 조회수
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,       -- 게시글 생성일
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP        -- 게시글 수정일
    );

    Generate an appropriate SQL query based on the question.

    Question: """ + question + """

    SQL Query:
    """

    # OpenAI GPT-4 모델 호출
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruction},  # 시스템 인스트럭션으로 제공
            {"role": "user", "content": question}          # 유저의 실제 프롬프트 제공
        ],
        max_tokens=150,
        temperature=0,
    )
    
    # 생성된 SQL 쿼리 추출
    sql_query = response.choices[0].message.content

    return sql_query

async def clean_sql_query(sql_query):
    sql_query = re.sub(r'^```(?:sql)?', '', sql_query, flags=re.IGNORECASE).strip()
    sql_query = re.sub(r'```$', '', sql_query).strip()
    return sql_query

async def execute_query(sql_query):
    conn = sqlite3.connect('test_database.db')
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        return rows

    except sqlite3.Error as e:
        return f"An error occurred: {e.args[0]}"

    finally:
        conn.close()

async def process_question(question):
    start_time = time.time()  # Start time for the process_question function

    # SQL 쿼리 생성
    sql_query = await text_to_sql_query(question)
    clean_query = await clean_sql_query(sql_query)

    # SQL 쿼리 실행
    result = await execute_query(clean_query)

    end_time = time.time()  # End time for the process_question function
    duration = end_time - start_time  # Calculate duration

    print(f"Execution time for '{question}': {duration:.2f} seconds")
    
    # 반환된 SQL 쿼리, 정리된 쿼리, 그리고 결과를 함께 반환
    return clean_query, result

question_texts = [
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
                "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
                "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
        "행사 모금이 가장 많이된 상위 5개의 제목을 알려줄래?",
        "조회수가 가장 높은 캠페인들은 무엇인가요?",
        "이번 달에 시작한 캠페인 목록을 보여주세요.",
        "admin이 만든 캠페인 중 모금액이 100000 이상인 것은?",
        "캠페인 종료일이 가까운 순서대로 3개를 알려주세요.",
        "모금액이 50000 미만인 캠페인은 무엇인가요?",
        "캠페인 제목에 '환경'이 포함된 것들을 보여주세요.",
        "오늘 생성된 캠페인들은 무엇이 있나요?",
        "view_count가 100 이상인 캠페인들의 제목과 조회수를 알려주세요.",
        "모금액 대비 조회수가 높은 캠페인을 알려줄 수 있나요?",
    ]
    
async def main():
    overall_start_time = time.time()  # Start time for the entire main function

    # 비동기적으로 모든 질문 처리
    results = await asyncio.gather(*(process_question(question) for question in question_texts))

    # 각 질문에 대해 출력
    for question, (clean_query, result) in zip(question_texts, results):
        print(f"Q: {question}")
        print(f"Query: {clean_query}")
        print(f"A: {result}\n")

    overall_end_time = time.time()  # End time for the entire main function
    overall_duration = overall_end_time - overall_start_time  # Calculate overall duration
    print(f"Total execution time for all questions: {overall_duration:.2f} seconds")

# In Jupyter or other environments where event loop is already running
if __name__ == "__main__":
    try:
        # This will work when no loop is running
        asyncio.run(main())
    except RuntimeError:
        print("The event loop is already running. This code must be executed in a compatible environment.")
