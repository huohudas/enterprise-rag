"""
检索优化模块
负责：混合检索（向量+BM25）、RRF重排、元数据过滤
"""

import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalModule:
    """检索优化模块"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        self._setup_retrievers()

    def _setup_retrievers(self):
        """初始化向量检索器和BM25检索器"""
        logger.info("正在初始化检索器...")

        # 向量检索器：基于语义相似度
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器：基于关键词匹配
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )
        logger.info("检索器初始化完成")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索：同时使用向量检索和BM25检索
        用RRF算法融合两个结果，取前top_k个返回
        """
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 3) -> List[Document]:
        """
        带元数据过滤的混合检索
        先检索更多候选，再按过滤条件筛选
        比如只搜索HR分类的文档
        """
        # 多取几个候选，保证过滤后还有足够结果
        candidates = self.hybrid_search(query, top_k=top_k * 3)

        filtered = []
        for doc in candidates:
            if self._match_filters(doc, filters):
                filtered.append(doc)
            if len(filtered) >= top_k:
                break

        return filtered

    def _match_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """判断文档是否符合过滤条件"""
        for key, value in filters.items():
            doc_value = doc.metadata.get(key)
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            else:
                if doc_value != value:
                    return False
        return True

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        RRF重排算法
        公式：每个文档的得分 = 1/(排名+k) 的累加
        同一个文档在两个检索结果里都靠前，最终得分就高
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        # 计算BM25检索结果的RRF分数，累加到同一文档
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        # 按总分从高到低排序
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_objects[doc_id] for doc_id, _ in sorted_items if doc_id in doc_objects]
