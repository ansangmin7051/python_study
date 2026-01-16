import requests
from config import Config

class RAGGenerator:
    def __init__(self):
        self.url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL

    def generate_answer(self, question, context):
        prompt = f"""
        [지시사항]
        아래 제공된 [데이터]만 참고하여 질문에 답하세요. 
        반드시 한국어로만 답변하고 영어 사용을 하지마세요.

        [데이터]
        {context}

        [질문]
        {question}
        """
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"에러 발생: {str(e)}"