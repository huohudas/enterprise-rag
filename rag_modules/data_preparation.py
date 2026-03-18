"""
数据准备模块
负责：读取文档、按标题分块、提取元数据、建立父子关系
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# 获取日志记录器，用于打印运行信息
logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块"""

    # 文件夹名到中文分类的映射
    # 作用：读取文件时自动识别文档属于哪个部门
    CATEGORY_MAPPING = {
        'hr': 'HR人事',
        'finance': '财务',
        'it': 'IT技术',
    }

    def __init__(self, data_path: str):
        """
        初始化模块
        data_path: 存放文档的根目录路径，比如 ./data
        """
        self.data_path = data_path
        self.documents: List[Document] = []      # 存放完整文档（父文档）
        self.chunks: List[Document] = []          # 存放分块后的小段（子文档）
        self.parent_child_map: Dict[str, str] = {} # 子块ID → 父文档ID 的映射表

    def load_documents(self) -> List[Document]:
        """
        读取data目录下所有markdown文件
        返回：Document对象列表，每个对象包含文件内容和元数据
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")

        documents = []
        data_path_obj = Path(self.data_path)

        # rglob("*.md") 递归搜索所有子目录下的.md文件
        # 比如同时找到 data/hr/员工手册.md 和 data/it/IT使用规范.md
        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 读取文件内容
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 用文件路径生成唯一ID（MD5哈希）
                # 目的：同一个文件每次生成的ID都一样，方便后续父子关系追踪
                relative_path = md_file.relative_to(data_path_obj).as_posix()
                parent_id = hashlib.md5(relative_path.encode()).hexdigest()

                # 创建Document对象
                # page_content：文档正文内容
                # metadata：附带的标签信息（文件路径、ID等）
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        # 给每个文档增加分类、文件名等元数据
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        从文件路径中提取：所属部门分类、文件名
        目的：后续可以按部门过滤，比如只搜索HR相关文档
        """
        file_path = Path(doc.metadata.get('source', ''))

        # 从路径中提取部门分类
        # 比如 data/hr/员工手册.md → 识别出 hr → 映射为 HR人事
        doc.metadata['category'] = '其他'
        for folder_name, category in self.CATEGORY_MAPPING.items():
            if folder_name in file_path.parts:
                doc.metadata['category'] = category
                break

        # 提取文件名（不含扩展名）作为文档标题
        # 比如 员工手册.md → 员工手册
        doc.metadata['doc_name'] = file_path.stem

    def chunk_documents(self) -> List[Document]:
        """
        按Markdown标题对文档进行分块
        返回：分块后的Document列表
        """
        if not self.documents:
            raise ValueError("请先调用 load_documents() 加载文档")

        logger.info("正在进行Markdown结构分块...")
        all_chunks = []

        # 定义按哪些标题级别切分
        # # 是一级标题，## 是二级标题，### 是三级标题
        headers_to_split_on = [
            ("#", "一级标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
        ]

        # 创建Markdown分割器
        # strip_headers=False 表示保留标题文字，方便理解上下文
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        for doc in self.documents:
            # 对每个文档按标题切分
            chunks = splitter.split_text(doc.page_content)

            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(chunks):
                import uuid
                child_id = str(uuid.uuid4())  # 为每个子块生成唯一ID

                # 子块继承父文档的所有元数据，再加上自己的信息
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content)
                })

                # 记录子块到父文档的映射关系
                self.parent_child_map[child_id] = parent_id
                all_chunks.append(chunk)

        self.chunks = all_chunks
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个块")
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索到的子块，找回对应的完整父文档
        原理：子块里存了parent_id，通过parent_id从self.documents里找完整文档
        目的：检索用小块（精准），生成回答用完整文档（信息完整）
        """
        # 统计每个父文档被匹配到几次，作为相关性排序依据
        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 匹配次数+1
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档，避免重复查找
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按匹配次数从多到少排序
        # 匹配次数越多说明这篇文档越相关
        sorted_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True
        )

        parent_docs = [parent_docs_map[pid] for pid in sorted_ids if pid in parent_docs_map]
        logger.info(f"从 {len(child_chunks)} 个子块找到 {len(parent_docs)} 个父文档")
        return parent_docs

    def get_statistics(self) -> Dict:
        """
        返回知识库统计信息
        用于启动时显示知识库基本情况
        """
        if not self.documents:
            return {}

        categories = {}
        for doc in self.documents:
            cat = doc.metadata.get('category', '未知')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
        }
