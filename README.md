# DIRAS: Efficiently and Effectively Annotate Reliable Information Retrieval Benchmark for Your RAG Appliciation
Official repository for the paper DIRAS: Efficient LLM-Assisted Annotation of Document Relevance in Retrieval Augmented Generation. [Link to Paper](https://arxiv.org/abs/2406.14162)

DIRAS solves the following pain points:
- Pain Point \#1: Information Retrieval is critical for RAG performance, but there lacks application-case-specific benchmark. -- *DIRAS leverages LLMs to do it for you!*
- Pain Point \#2: Annotating document relevance using SOTA API-based LLM is expensive, resulting in unthoroughly annotating (query, document) pairs. -- *DRIAS finetunes Open-sourced LLMs which is (1)efficient: can annotate all (query, document) combinations in minutes (2) effective: achieving GPT-4-level performance*
- Pain Point \#3: Different RAG applications have (subtly) different definitions about what is relevant/irrelevant. -- *DIRAS takes granular or domain-specific relevance definitions into account*

Reasons why using DIRAS instead of [LlamaIndex info-retrieval benchmarking]()
