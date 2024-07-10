# DIRAS: Efficiently and Effectively Annotate Reliable Information Retrieval Benchmark for Your RAG Appliciation
Official repository for the paper DIRAS: Efficient LLM-Assisted Annotation of Document Relevance in Retrieval Augmented Generation. [Link to Paper](https://arxiv.org/abs/2406.14162)

DIRAS solves the following pain points:
- Pain Point \#1: Information Retrieval is critical for RAG performance, but there lacks application-case-specific benchmark. -- *DIRAS leverages LLMs to do it for you!*
- Pain Point \#2: Annotating document relevance using SOTA API-based LLM is expensive, resulting in unthoroughly annotating (query, document) pairs. -- *DRIAS finetunes Open-sourced LLMs which is (1) efficient: can annotate all (query, document) combinations in minutes (2) effective: achieving GPT-4-level performance!*
- Pain Point \#3: Different RAG applications have (subtly) different definitions about what is relevant/irrelevant. -- *DIRAS takes granular and domain-specific relevance definitions into account!*

Two apparent reasons why using DIRAS instead of [LlamaIndex info-retrieval benchmarking](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/)
- Reason 1: DIRAS takes domain-specific relevance definition into consideration. It also produce calibrated relevance scores instead of binary relevance labels, addressing the painful "partially relevance" problem.
- Reason 2: DIRAS achieves superior performance using open-sourced LLMs (verified in our [paper](https://arxiv.org/abs/2406.14162), while LlamaIndex retriever evaluator need API-based LLMs to achieve best outcome.

Besides benchmark annotator, DIRAS fine-tuned LLMs can serve as efficient and effective re-rankers to boost RAG performance.

### How to use DIRAS?

