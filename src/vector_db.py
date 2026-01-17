from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from config import Config
import pandas as pd
import duckdb
import uuid
import torch

class WelfareVectorDB:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.dense_model = SentenceTransformer(Config.DENSE_EMBEDDING_MODEL_NAME)
        self.vector_dim = self.dense_model.get_sentence_embedding_dimension()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.SPARSE_EMBEDDING_MODEL_NAME)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(Config.SPARSE_EMBEDDING_MODEL_NAME)
        self.sparse_model.eval()
        torch.set_grad_enabled(False)

    def create_collection(self):
        self.client.recreate_collection(
            collection_name = Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=self.vector_dim, 
                                               distance = models.Distance.COSINE
                                               ),
            hnsw_config=models.HnswConfigDiff(m=16, 
                                              ef_construct=200
                                              )
        )
        self._create_payload_indexs()

    def hybrid_create_collection(self):
        self.client.recreate_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config={"dense_vector":models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE
            )},
            sparse_vectors_config={
                "sparse_vector":models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
            },
            hnsw_config=models.HnswConfigDiff(
                m=16, 
                ef_construct=200
            )
        )
        self._create_payload_indexs()

    def _create_payload_indexs(self): 
        self.client.create_payload_index(Config.COLLECTION_NAME, 
                                         "category", 
                                         models.PayloadSchemaType.KEYWORD
                                         )
        self.client.create_payload_index(Config.COLLECTION_NAME, 
                                         "title", 
                                         models.TextIndexParams(type="text",
                                                                tokenizer = models.TokenizerType.WORD
                                                                )
                                        )

    def upsert_documents(self, data_list):
        points=[]    
        for item in data_list:
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, item['id']))
            dense_vector = self.dense_model.encode(item['content']).tolist()
            points.append(models.PointStruct(id = pid, 
                                             vector = dense_vector, 
                                             payload={'id' : item['id'], 
                                                      'category' : item['category'], 
                                                      'title' : item['title']}
                                             )
                          )
            
        self.client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=points
        )    

        df = pd.DataFrame([{'id':r['id'], 'content':r['content']} for r in data_list])
        df.to_parquet(Config.PARQUET_PATH, engine = 'pyarrow', index=False)

    def hybrid_upsert_documents(self, data_list):
        points=[]
        for item in data_list:
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, item['id']))
            dense_vector = self.dense_model.encode(item['content']).tolist()
            indices, weights = self._create_sparse_vector(item['content'])

            points.append(
                models.PointStruct(
                    id = pid,
                    vector = {
                        'dense_vector':dense_vector,
                        'sparse_vector':models.SparseVector(
                            indices=indices,
                            values=weights
                        )
                    },
                    payload = {
                        'id' : item['id'], 
                        'category' : item['category'], 
                        'title' : item['title']
                    }
                )
            )
        self.client.upsert(collection_name=Config.COLLECTION_NAME,points=points)    

    def _create_sparse_vector(self, content):
        inputs = self.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        outputs = self.sparse_model(**inputs)
        logits = outputs.logits

        values, _ = torch.max(
            torch.log(1+torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
            dim =1
        )

        # 3. squeeze()는 [1, Vocab]을 [Vocab]으로 만듭니다.
        values = values.squeeze()

        # 4. 0이 아닌 값의 인덱스 추출 (as_tuple=True가 더 안전함)
        indices_tensor = values.nonzero(as_tuple=True)[0]
        
        # 5. 결과 변환
        indices = indices_tensor.cpu().tolist()
        weights = values[indices_tensor].cpu().tolist()
        
        return indices, weights

    def search_relevant_documents(self, query, limit=3):
        query_vector = self.dense_model.encode(query).tolist() 
        search_result = self.client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                should=[
                    models.FieldCondition(key='category',
                                          match=models.MatchValue(value="청년")
                                          ),
                    models.FieldCondition(key='title',
                                          match=models.MatchText(text="청년")
                                          )                      
                ]
            ),
            limit=limit
        ).points  

        if not search_result: return ""

        ids = [s.payload["id"] for s in search_result]
        id_tuple = str(tuple(ids)) if len(ids) > 1 else f"('{ids[0]}')"
        order_case = " ".join([f"WHEN id = '{id_val}' THEN {i}" for i, id_val in enumerate(ids)])

        sql = f"""
            SELECT content FROM read_parquet('{Config.PARQUET_PATH}')
            WHERE id IN {id_tuple}
            ORDER BY CASE {order_case} END
        """
        results = duckdb.query(sql).fetchall()
        return "\n".join([r[0] for r in results])
    
    def hybrid_search_relevant_documents(self, query, limit=3):
        query_vector = self.dense_model.encode(query).tolist() 
        indices, weights = self._create_sparse_vector(query)

        search_result = self.client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                should=[
                    models.FieldCondition(key='category',
                                          match=models.MatchValue(value="청년")
                                          ),
                    models.FieldCondition(key='title',
                                          match=models.MatchText(text="청년")
                                          )                      
                ]
            ),
            limit=limit
        ).points  

        if not search_result: return ""

        ids = [s.payload["id"] for s in search_result]
        id_tuple = str(tuple(ids)) if len(ids) > 1 else f"('{ids[0]}')"
        order_case = " ".join([f"WHEN id = '{id_val}' THEN {i}" for i, id_val in enumerate(ids)])

        sql = f"""
            SELECT content FROM read_parquet('{Config.PARQUET_PATH}')
            WHERE id IN {id_tuple}
            ORDER BY CASE {order_case} END
        """
        results = duckdb.query(sql).fetchall()
        return "\n".join([r[0] for r in results])    