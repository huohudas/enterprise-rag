"""
索引构建模块
负责：向量化文档、构建FAISS索引、保存和加载索引
"""

import logging
from pathlib import Path
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self._setup_embeddings()

    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在加载嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("嵌入模型加载完成")

    def build_index(self, chunks: List[Document]) -> FAISS:
        """向量化文档块并构建FAISS索引"""
        logger.info(f"正在构建向量索引，共 {len(chunks)} 个文档块...")
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        logger.info("向量索引构建完成")
        return self.vectorstore

    def save_index(self):
        """保存索引到本地"""
        if not self.vectorstore:
            raise ValueError("请先构建索引")
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"索引已保存到: {self.index_save_path}")

    def load_index(self) -> Optional[FAISS]:
        """从本地加载索引，不存在则返回None"""
        if not Path(self.index_save_path).exists():
            logger.info("未找到已保存的索引，将重新构建")
            return None
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载索引失败: {e}，将重新构建")
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似度搜索"""
        if not self.vectorstore:
            raise ValueError("请先构建或加载索引")
        return self.vectorstore.similarity_search(query, k=k)
