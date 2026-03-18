"""
主程序入口
把四个模块串联起来，实现完整的RAG问答流程
"""

import os
import logging
from dotenv import load_dotenv
from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalModule,
    GenerationModule
)

# 加载.env文件中的环境变量（API Key等）
load_dotenv()

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseRAGSystem:
    """企业知识库RAG系统主类"""

    def __init__(self):
        self.config = DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

    def initialize(self):
        """初始化所有模块"""
        print("正在初始化系统...")

        self.data_module = DataPreparationModule(self.config.data_path)
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )
        self.generation_module = GenerationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        print("系统初始化完成")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 先尝试加载已有索引，避免重复构建
        vectorstore = self.index_module.load_index()

        # 无论是否有索引，都需要加载文档（用于父子文档检索）
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()

        if vectorstore is None:
            print("未找到已有索引，正在重新构建...")
            vectorstore = self.index_module.build_index(chunks)
            self.index_module.save_index()
            print("索引构建并保存完成")
        else:
            print("已加载已有索引")

        # 初始化检索模块
        self.retrieval_module = RetrievalModule(vectorstore, chunks)

        # 显示知识库统计
        stats = self.data_module.get_statistics()
        print(f"\n知识库统计：")
        print(f"  文档总数：{stats['total_documents']}")
        print(f"  文本块数：{stats['total_chunks']}")
        print(f"  文档分类：{stats['categories']}")
        print("\n知识库构建完成！")

    def ask(self, question: str, stream: bool = True):
        """
        回答用户问题
        完整流程：路由 → 重写 → 检索 → 获取父文档 → 生成回答
        """
        print(f"\n问题：{question}")
        print("-" * 50)

        # 第一步：查询路由，判断问题类型
        route = self.generation_module.query_router(question)
        print(f"问题类型：{route}")

        # 第二步：查询重写，优化检索词
        rewritten = self.generation_module.query_rewrite(question)

        # 第三步：从问题中提取过滤条件
        filters = self._extract_filters(question)

        # 第四步：检索相关子块
        if filters:
            print(f"应用过滤：{filters}")
            chunks = self.retrieval_module.filtered_search(
                rewritten, filters, top_k=self.config.top_k
            )
        else:
            chunks = self.retrieval_module.hybrid_search(
                rewritten, top_k=self.config.top_k
            )

        if not chunks:
            print("未找到相关文档")
            return

        print(f"检索到 {len(chunks)} 个相关文档块")

        # 第五步：获取完整父文档
        parent_docs = self.data_module.get_parent_documents(chunks)
        print(f"对应 {len(parent_docs)} 篇完整文档")

        # 第六步：生成回答
        print("\n回答：")
        if stream:
            for chunk in self.generation_module.generate_answer_stream(question, parent_docs):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            answer = self.generation_module.generate_answer(question, parent_docs)
            print(answer)

    def _extract_filters(self, query: str) -> dict:
        """
        从问题中提取过滤条件
        比如问题含有"IT"就只搜IT文档，含有"报销"就只搜财务文档
        """
        filters = {}
        keyword_map = {
            'HR人事': ['年假', '请假', '考勤', '薪资', '绩效', '入职', '离职', '合同', '婚假', '产假'],
            '财务': ['报销', '差旅', '发票', '费用', '出差', '审批'],
            'IT技术': ['账号', '密码', '系统', '电脑', '网络', '数据', '安全'],
        }
        for category, keywords in keyword_map.items():
            if any(kw in query for kw in keywords):
                filters['category'] = category
                break
        return filters

    def run(self):
        """运行交互式问答"""
        print("=" * 60)
        print("    企业知识库智能问答系统")
        print("=" * 60)

        self.initialize()
        self.build_knowledge_base()

        print("\n开始问答（输入'退出'结束）：")

        while True:
            try:
                question = input("\n请输入问题：").strip()
                if not question or question in ['退出', 'quit', 'exit']:
                    break
                self.ask(question)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"出错了：{e}")

        print("\n感谢使用！")


if __name__ == "__main__":
    system = EnterpriseRAGSystem()
    system.run()
