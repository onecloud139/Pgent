import os
import torch
from typing import Optional
from langchain_core.tools import tool
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.llms.deepseek import DeepSeek 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# 配置项
deepseek_key= r"sk-4a"
TXT_DOC_PATH = "./fin_knowledge"
INDEX_STORAGE_PATH = f"{TXT_DOC_PATH}/rag_index_storage"
FAISS_INDEX_PATH = f"{TXT_DOC_PATH}/faiss_index"
_query_engine: Optional[object] = None

def init_rag_engine() -> None:
    """初始化LlamaIndex RAG引擎（单例+索引持久化+全维度加速）"""
    global _query_engine
    if _query_engine is not None:
        return

    Settings.embed_model = HuggingFaceEmbedding(model_name="/mnt/data/models/qwen3-embedding-4b")
    
    llm = DeepSeek(model="deepseek-chat", api_key=deepseek_key)
    Settings.llm = llm

    try:
        if os.path.exists(INDEX_STORAGE_PATH) and os.path.exists(FAISS_INDEX_PATH):
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=INDEX_STORAGE_PATH)
            index = load_index_from_storage(storage_context)
        else:
            documents = SimpleDirectoryReader(TXT_DOC_PATH).load_data()
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=16)
            test_embedding = Settings.embed_model.get_text_embedding("test")
            d = len(test_embedding)

            faiss_index = faiss.IndexHNSWFlat(d, 16)
            faiss_index.hnsw.ef_construction = 40
            faiss_index.hnsw.ef_search = 20
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_documents(
                documents, node_parser=text_splitter, storage_context=storage_context
            )
            os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
            os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
            index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)

        _query_engine = index.as_query_engine(
            similarity_top_k=5, response_mode="compact")
    except Exception as e:
        raise RuntimeError(f"RAG引擎初始化失败：{str(e)}")

@tool("fin_document_rag", description="当用户询问**基础投资原则、风险等级、市场分类**等知识时，使用此工具。它提供来自内部金融文档的背景信息和规则。")
def fin_document_rag(query: str) -> str:
    if _query_engine is None:
        init_rag_engine()
    try:
        response = _query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"RAG工具调用失败：{str(e)}"