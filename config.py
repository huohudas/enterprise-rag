"""
配置管理文件
集中管理所有系统参数，方便统一修改
"""
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # 数据路径
    data_path: str = "./data"
    
    # 嵌入模型（向量化用）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # 向量索引保存路径
    index_save_path: str = "./vector_index"
    
    # LLM模型
    llm_model: str = "glm-4-flash"
    
    # 生成参数
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # 检索参数
    top_k: int = 3              # 最终返回文档数
    retrieval_k: int = 5        # 每个检索器返回数量

# 默认配置，直接使用这个
DEFAULT_CONFIG = RAGConfig()
