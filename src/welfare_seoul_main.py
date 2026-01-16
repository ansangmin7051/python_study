from vector_db import WelfareVectorDB
from generator import RAGGenerator

seoul_welfare_data = [
    {
        "id": "WELF_SEOUL_001",
        "category": "청년",
        "title": "서울시 청년수당",
        "content": "서울시에 거주하며 최종학력 졸업 후 미취업 상태인 만 19~34세 청년에게 월 50만원씩 최대 6개월간 활동지원금을 지급합니다. 단, 중위소득 150% 이하 가구여야 합니다."
    },
    {
        "id": "WELF_SEOUL_002",
        "category": "주거",
        "title": "서울시 청년월세지원",
        "content": "서울시 거주 만 19~39세 무주택 청년 1인 가구에게 월 최대 20만원, 총 12회(240만원)의 월세를 지원합니다. 임차보증금 5천만원 이하 및 월세 60만원 이하 건물에 거주해야 합니다."
    },
    {
        "id": "WELF_SEOUL_003",
        "category": "출산/육아",
        "title": "서울형 산후조리경비 지원",
        "content": "서울시에 거주하는 모든 산모에게 출생아 1인당 100만원 상당의 바우처를 지급합니다. 산후조리원 이용뿐만 아니라 산후우울증 검사, 의약품 구매 등에도 사용 가능합니다."
    },
    {
        "id": "WELF_SEOUL_004",
        "category": "어르신",
        "title": "서울시 어르신 대중교통 이용지원",
        "content": "서울시에 거주하는 만 65세 이상 어르신에게 지하철 무임승차 혜택을 제공하며, 버스 이용 시에도 일정 금액의 마일리지를 환급해 드리는 지버지(지하철·버스 지원) 서비스를 운영합니다."
    },
    {
        "id": "WELF_SEOUL_005",
        "category": "일자리",
        "title": "서울시 뉴딜일자리",
        "content": "시민 생활에 필요한 공공서비스 분야에서 일하며 실무 경력을 쌓을 수 있도록 최대 23개월간 일자리를 제공하고, 서울형 생활임금을 적용하여 급여를 지급합니다."
    },
    {
        "id": "WELF_SEOUL_006",
        "category": "장애인",
        "title": "장애인 버스요금 지원",
        "content": "서울시에 거주하는 장애인과 보호자 1인에게 서울 버스 및 서울과 연계된 경기/인천 버스 이용 요금을 월 최대 5만원까지 실비 지원합니다."
    }
]

def main():
    # 1. 초기화 (Initialization, 초기화)
    db = WelfareVectorDB()
    gen = RAGGenerator()

    # 2. 데이터 준비 (데이터가 없을 때만 실행하도록 구성 가능)
    db.create_collection()
    db.upsert_documents(seoul_welfare_data)
    print("데이터 업서트 완료!")

    # 3. 질문 처리 (Query Process, 쿼리 프로세스)
    user_question = "서울시에서 청년들에게 지원 하는 복지 정책의 내용에 대해서 알려줘"
    
    print(f"질문: {user_question}\n")
    print("관련 정보를 찾는 중...")
    
    context = db.search_relevant_documents(user_question)
    
    if context:
        print("답변 생성 중...\n")
        answer = gen.generate_answer(user_question, context)
        print("="*30)
        print("AI 분석 결과:")
        print(answer)
        print("="*30)
    else:
        print("관련된 정보를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()