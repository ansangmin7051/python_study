class Config:
    COLLECTION_NAME = "welfare_seoul"
    DENSE_EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"
    SPARSE_EMBEDDING_MODEL_NAME = "naver/splade-v3"
    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:1b"
    PARQUET_PATH = "data/seoul_welfare_data.parquet"
