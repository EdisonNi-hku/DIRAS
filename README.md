# DIRAS: Efficiently and Effectively Annotate Reliable Information Retrieval Benchmark for Your RAG Appliciation
Official repository for the paper DIRAS: Efficient LLM-Assisted Annotation of Document Relevance in Retrieval Augmented Generation. [Link to Paper](https://arxiv.org/abs/2406.14162)

DIRAS solves the following pain points:
- Pain Point \#1: Without an application-case-specific benchmark, how can we optimize RAG implementation? Especially for the info-retrieval module? -- *DIRAS leverages LLMs to annotate it for you!*
- Pain Point \#2: Annotating document relevance using SOTA API-based LLM is expensive, and can not cover all possible (query, document) pairs. -- *DRIAS finetunes Open-sourced LLMs which is (1) efficient: can annotate all (query, document) combinations in minutes (2) effective: achieving GPT-4-level performance!*
- Pain Point \#3: Different RAG applications have (subtly) different definitions about what is relevant/irrelevant. -- *DIRAS takes granular and domain-specific relevance definitions into account!*

Two apparent reasons why using DIRAS instead of [LlamaIndex info-retrieval benchmarking](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/)
- Reason 1: DIRAS takes domain-specific relevance definition into consideration. It also produce calibrated relevance scores instead of binary relevance labels, addressing the painful "partially relevance" problem.
- Reason 2: DIRAS achieves superior performance using open-sourced LLMs (verified in our [paper](https://arxiv.org/abs/2406.14162), while LlamaIndex retriever evaluator need API-based LLMs to achieve best outcome.

Besides benchmark annotator, DIRAS fine-tuned LLMs can serve as efficient and effective re-rankers to boost RAG performance.

## How to use DIRAS?
### Step 0: Configurate your OpenAI API key.
This repo relies on OpenAI models for embedding and training data distillation. Please put your API key in `code/apikey.env`.

### Step 1: Prepare your questions and documents.
Prepare questions and documents from your RAG application in `jsonl` format. See examples in [`example/question.jsonl`](./example/question.jsonl) and [`example/document.jsonl`](./example/document.jsonl).

Specifically, `question.jsonl` should initially contain one field: `question`, containing all questions you provided, and alternatively an `explanation` filed for question explanations.

`document.jsonl` should contain two fields: `report` for the report names and `document` for the text chunks from the corresponding report.

### Step 2: Generate question explanations.
If your `question.jsonl` does not contain an `explanation` field, you may want to generate the question explanations using our prompt and GPT-4o by running the following command:
```shell
python get_question_explanation.py --question_file example/question.jsonl
```
The script will output explanations for each question.

### Step 3: Distill training data from GPT-4o.
After having question explanations (i.e., relevance definitions), distill fine-tuning data using the following command:
```shell
python get_training_data.py --question_file example/question.jsonl --document_file example/document.jsonl --output_file example/distilled_train_data.xlsx --student_llm meta-llama/Meta-Llama-3-8B-Instruct --cache . --huggingface_token xxx
```
where `output_file` is the path and name of the generated data. The instruction-response pairs formatted by the student LLM's chat template will be in `example/distilled_train_data.csv`. `cache` should be the path to your huggingface storage cache. `huggingface_token` refers to the huggingface [user access token](https://huggingface.co/docs/hub/security-tokens), which is necessary for accessing some models (e.g., llama-3).

### Step 4: Fine-tune Llama-3-8b-Instruct with distilled data.
Then we use QLoRA to fine-tune llama-3-8b-Instruct:
```shell
bash run_qlora.sh meta-llama/Meta-Llama-3-8B-Instruct example/distilled_train_data.csv CACHE_DIR ./finetuned_llama3
```
- The first field should be the student model's name or path.
- The second field  is the path to the distilled training data.
- The third field is your huggingface cache directory.
- The fourth field is the output directory of fine-tuned checkpoints.

### Step 5: Annotate all (question, document) pairs using the fine-tuned llama-3.
```shell
python inference.py --model_cache_dir CACHE_DIR --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./finetuned_llama3/checkpoint-xxx --load_tokenizer --question_file example/question.jsonl --document_file example/document.jsonl --output_file example/annotated_question_document_pairs.csv
```
- Again, `model_cache_dir` stands for huggingface cache.
- `model_path` is the base model name or path.
- `lora_path` is the path to fine-tuned LoRA.
- `load_tokenizer` indicates the tokenizer will be loaded from the saved LoRA.
- `output_file` will contain all inference results.

The `output_file` will contain the final annotations in 3 fields:
- `guess`: model's guess on whether the document is helpful for answering the question or not.
- `ask_confidence`: the model's linguistic confidence value of the guess being correct.
- `tok_confidence`: the model's confidence of saying Yes or No indicated by token-level probability.

According to our observations in paper, `ask_confidence` is easier to interpret while `tok_confidence` is more accurate.

## How to reproduce the results in paper?
### ChatReport Data
Let's first go through our annotated dataset files. The `data` directory contains the following files:
- `data/chatreport_test.csv` The annotated test split of ChatReport data (details in paper Section 3). It contains following fields:
  - `Question`, `Background`, and `Paragraph`: questions, question explanations, and paragraph from [ChatReport](https://github.com/edisonni-hku/chatreport).
  - `annotation_1`, `annotation_2`, and `meta_annotation`: annotations from our three expert annotators about relevance.
  - `hard`: uncertainty label (see paper Section 2.2). 1 means uncertain; 0 means not uncertain.
  - `gold`: relevance labels. Yes means relevant; No means irrelevant.
- `data/chatreport_train.csv` Training data distilled for ChatReport experiment.
  - `Question`, `Background`, and `Paragraph`: questions, question explanations, and paragraph from [ChatReport](https://github.com/edisonni-hku/chatreport). The set of questions and reports is different from the test set.
  - `system_prompt` and `user_prompt`: prompts for data distillation.
  - `gpt4_answer`: distilled answer from GPT-4.

### Reproduce Table 1 and Table 2 First 5 Rows
In paper, Table 1 investigates how to distill high-performance ranking results from GPT-4. We follow the original [paper](https://aclanthology.org/2023.emnlp-main.923.pdf) to implement the listwise ranking algorithm (see `code/listwise_rerank.py`). 
To reproduce the numbers in table 1 and the first five rows of Table 2, run the following code:
```shell
python code/evaluation.py
```
This code will generate `results_embed.xlsx` and `results_chatgpt.xlsx`, containing the numbers in Table 1 and first five rows of Table 2.

- `results_embed.xlsx` contains all rerankers, embedding models, and GPT-based listwise ranker results.
- `results_chatgpt.xlsx` contains ChatGPT results.

Due to the reproduction problem of OpenAI models during the experiment time (March 2024), we release all OpenAI generation results in `data/chatreport_test_results.csv`.
You can also reproduce from scratch using `code/get_training_data.py` and `code/listwise_rerank.py` using `gpt-4-preview-0125`. The scores should have no significant difference from those in paper.

### Reproduce the Rest of Table 2
All inference results of fine-tuned models in paper are stored in `small_llm_results`. The rest of Table 2 can be reproduced by running:
```shell
python code/run_evaluate_small_llms.py
```
Then you will get `results_raw.xlsx` and `results_ft.xlsx`, containing results of raw LLMs and their 2-epoch fine-tuned checkpoints on `data/chatreport_train.csv`.

To reproduce the fine-tuned checkpoints, we can use the fine-tuning script `code/run_qlora.sh`.


### Reproduce ClimRetrieve Results

Relevant (query, document) pairs annotated by ClimRetrieve can be found in `data/climretrieve_relevant.xlsx`. The fields:
- `paragraph`, `report`, and `question`: paragraphs, name of the reports, and questions from ClimRetrieve
- `relevant_text`: the exact text chunks considered relevant by the experts.
- `relevance`: how relevant the paragraph is, degree from 1 to 3.
- `background_generic`, `background_specific`: the question explanation written by GPT-4 and improved by Human experts (see paper section 4.2).

To reproduce the results in Table 3, run:
```shell
python code/evaluate_climretrieve.py
```

`data/climretrieve_all.zip` contains all (query, document) pairs in ClimRetrieve. To reproduce the results in Table 4 and 5, please run:
```shell
cd data
unzip climretrieve_all.zip
cd ..
python code/evaluate_climretrieve_all.py
```

### ClimRetrieve Re-annotation
#TODO
