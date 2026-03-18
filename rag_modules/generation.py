"""
生成集成模块
负责：初始化LLM、查询路由、查询重写、生成回答
"""

import os
import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationModule:
    """生成集成模块"""

    def __init__(self, model_name: str = "glm-4-flash", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._setup_llm()

    def _setup_llm(self):
        """初始化智谱GLM模型，使用OpenAI兼容接口"""
        logger.info(f"正在初始化LLM: {self.model_name}")
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("请在.env文件中设置 ZHIPU_API_KEY")

        # 智谱API兼容OpenAI格式，只需替换base_url和api_key
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        logger.info("LLM初始化完成")

    def query_router(self, query: str) -> str:
        """
        查询路由：判断用户问题属于哪种类型
        返回：'detail'（详细问答）或 'general'（一般问题）
        目的：不同类型用不同的Prompt模板，回答质量更好
        """
        prompt = ChatPromptTemplate.from_template("""
判断用户问题的类型，只返回以下其中一个词：detail 或 general

detail：用户想了解具体规定、流程、标准，比如"年假怎么申请""报销流程是什么"
general：用户在闲聊或问很宽泛的问题，比如"你好""公司怎么样"

用户问题：{query}
类型：""")

        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        result = chain.invoke(query).strip().lower()
        return result if result in ['detail', 'general'] else 'detail'

    def query_rewrite(self, query: str) -> str:
        """
        查询重写：把用户的口语化问题改写成更适合检索的形式
        比如"我想请假"→"员工请假申请流程和审批规定"
        目的：提高检索准确率
        """
        prompt = PromptTemplate(
            template="""请将用户的问题改写为更适合在企业知识库中检索的形式。
如果问题已经很清晰，直接返回原问题。
只返回改写后的问题，不要解释。

原问题：{query}
改写后：""",
            input_variables=["query"]
        )

        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke(query).strip()
        if rewritten != query:
            logger.info(f"查询重写：'{query}' → '{rewritten}'")
        return rewritten

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成回答（普通输出）
        把检索到的文档拼接成上下文，让LLM根据上下文回答问题
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""你是企业知识库智能助手。请根据以下文档内容回答员工的问题。
回答要准确、简洁、实用。如果文档中没有相关信息，请如实说明。

参考文档：
{context}

员工问题：{question}

回答：""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成回答（流式输出）
        逐字输出，用户不需要等待全部生成完才能看到内容
        使用yield关键字，每生成一点就立即输出一点
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""你是企业知识库智能助手。请根据以下文档内容回答员工的问题。
回答要准确、简洁、实用。如果文档中没有相关信息，请如实说明。

参考文档：
{context}

员工问题：{question}

回答：""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 3000) -> str:
        """
        把多个文档拼接成一段上下文字符串
        加入文档来源和分类信息，帮助LLM更好理解内容
        限制总长度避免超过LLM的token限制
        """
        if not docs:
            return "暂无相关文档。"

        parts = []
        total_length = 0

        for i, doc in enumerate(docs, 1):
            doc_name = doc.metadata.get('doc_name', '未知文档')
            category = doc.metadata.get('category', '未知分类')
            header = f"【文档{i}】{doc_name}（{category}）"
            content = f"{header}\n{doc.page_content}\n"

            if total_length + len(content) > max_length:
                break

            parts.append(content)
            total_length += len(content)

        return "\n" + "="*40 + "\n" + ("\n" + "="*40 + "\n").join(parts)
