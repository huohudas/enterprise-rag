"""
RAGAS评测脚本 - 适配 ragas 0.4.3
"""

import os
import sys
sys.path.insert(0, '/workspaces/enterprise-rag')

from dotenv import load_dotenv
load_dotenv('/workspaces/enterprise-rag/.env')

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalModule,
    GenerationModule
)

TEST_CASES = [
    {
        "question": "公司几点上班？",
        "reference": "上班时间为9:00，下班时间为18:00，午休12:00-13:00，每日工作8小时。"
    },
    {
        "question": "年假怎么申请？",
        "reference": "需提前3个工作日通过OA系统提交申请，经直属上级审批后方可休假。"
    },
    {
        "question": "工作3年能有几天年假？",
        "reference": "工作满3年不满10年享有10天年假。"
    },
    {
        "question": "报销需要几天内提交？",
        "reference": "差旅费须在出差结束后7个工作日内提交，日常费用须在发生后30天内提交。"
    },
    {
        "question": "密码需要多久更换一次？",
        "reference": "密码每90天强制更换一次，且不能重复使用最近5次的密码。"
    },
]


def build_system():
    print("正在初始化RAG系统...")
    data_module = DataPreparationModule(DEFAULT_CONFIG.data_path)
    index_module = IndexConstructionModule(
        model_name=DEFAULT_CONFIG.embedding_model,
        index_save_path=DEFAULT_CONFIG.index_save_path
    )
    data_module.load_documents()
    chunks = data_module.chunk_documents()
    vectorstore = index_module.load_index()
    if vectorstore is None:
        vectorstore = index_module.build_index(chunks)
        index_module.save_index()
    retrieval_module = RetrievalModule(vectorstore, chunks)
    generation_module = GenerationModule(
        model_name=DEFAULT_CONFIG.llm_model,
        temperature=DEFAULT_CONFIG.temperature
    )
    print("RAG系统初始化完成")
    return data_module, retrieval_module, generation_module


def collect_rag_outputs(data_module, retrieval_module, generation_module):
    print(f"\n正在对 {len(TEST_CASES)} 个问题运行RAG系统...")
    results = []
    for i, case in enumerate(TEST_CASES):
        question = case["question"]
        print(f"  处理问题 {i+1}/{len(TEST_CASES)}: {question}")
        chunks = retrieval_module.hybrid_search(question, top_k=3)
        parent_docs = data_module.get_parent_documents(chunks)
        generation_module_inst = GenerationModule(
            model_name=DEFAULT_CONFIG.llm_model,
            temperature=DEFAULT_CONFIG.temperature
        )
        answer = generation_module.generate_answer(question, parent_docs)
        contexts = [doc.page_content[:500] for doc in parent_docs]
        results.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": case["reference"]
        })
    return results


def run_evaluation(results):
    print("\n正在运行RAGAS评测...")

    api_key = os.getenv("ZHIPU_API_KEY")

    # 使用新版llm_factory，通过OpenAI兼容接口接入智谱
    openai_client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    eval_llm = llm_factory(
        model="glm-4-flash",
        client=openai_client
    )

    # 嵌入模型使用RAGAS内置的HuggingFace支持
    from ragas.embeddings import HuggingFaceEmbeddings
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    dataset = EvaluationDataset.from_list(results)

    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ContextRecall(llm=eval_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    return result


def main():
    data_module, retrieval_module, generation_module = build_system()
    results = collect_rag_outputs(data_module, retrieval_module, generation_module)
    eval_result = run_evaluation(results)

    print("\n" + "="*60)
    print("RAGAS 评测结果")
    print("="*60)
    print(eval_result)

    import json
    try:
        scores = {
            "faithfulness": float(eval_result["faithfulness"]),
            "answer_relevancy": float(eval_result["answer_relevancy"]),
            "context_recall": float(eval_result["context_recall"]),
        }
        with open("evaluation/ragas_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"\n评测结果已保存到 evaluation/ragas_scores.json")
    except Exception as e:
        print(f"保存分数时出错: {e}")


if __name__ == "__main__":
    main()
