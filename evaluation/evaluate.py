"""
RAGAS评测脚本 - 适配 ragas 0.4.3
只评测 Faithfulness 和 ContextRecall 两个核心指标
"""

import os
import sys
import json
sys.path.insert(0, '/workspaces/enterprise-rag')

from dotenv import load_dotenv
load_dotenv('/workspaces/enterprise-rag/.env')

from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, ContextRecall

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
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    eval_llm = llm_factory("glm-4-flash", client=client)

    faithfulness = Faithfulness()
    context_recall = ContextRecall()

    dataset = EvaluationDataset.from_list(results)
    metrics = [faithfulness, context_recall]
    result = evaluate(dataset=dataset, metrics=metrics, llm=eval_llm)
    return result


def main():
    data_module, retrieval_module, generation_module = build_system()
    results = collect_rag_outputs(data_module, retrieval_module, generation_module)
    eval_result = run_evaluation(results)

    print("\n" + "="*60)
    print("RAGAS 评测结果")
    print("="*60)

    import numpy as np
    faithfulness_score = float(np.nanmean(eval_result["faithfulness"]))
    context_recall_score = float(np.nanmean(eval_result["context_recall"]))

    print(f"\n  忠实度 Faithfulness：{faithfulness_score:.2f}  （建议>0.8）")
    print(f"  上下文召回 ContextRecall：{context_recall_score:.2f}  （建议>0.7）")

    if faithfulness_score < 0.8:
        print("\n  ⚠️  忠实度偏低，建议优化Prompt，加强约束LLM只使用文档内容回答")
    if context_recall_score >= 0.7:
        print("\n  ✅  上下文召回表现良好，检索质量达标")

    scores = {
        "faithfulness": faithfulness_score,
        "context_recall": context_recall_score
    }
    with open("evaluation/ragas_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果已保存到 evaluation/ragas_scores.json")


if __name__ == "__main__":
    main()
