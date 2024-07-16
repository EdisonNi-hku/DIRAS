import subprocess

scripts_and_args = [
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_rgc_ask', '--file', 'small_llm_results/gemma_rgc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_rgc_tok', '--file', 'small_llm_results/gemma_rgc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_gc_ask', '--file', 'small_llm_results/gemma_gc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_gc_tok', '--file', 'small_llm_results/gemma_gc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_rgc_ask', '--file', 'small_llm_results/llama_rgc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_rgc_tok', '--file', 'small_llm_results/llama_rgc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_gc_ask', '--file', 'small_llm_results/llama_gc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_gc_tok', '--file', 'small_llm_results/llama_gc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_rgc_ask', '--file', 'small_llm_results/phi3_rgc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_rgc_tok', '--file', 'small_llm_results/phi3_rgc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_gc_ask', '--file', 'small_llm_results/phi3_gc.csv', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_gc_tok', '--file', 'small_llm_results/phi3_gc.pkl', '--output', 'results_raw.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_rgc_2e_ask', '--file', 'small_llm_results/rgc_gemma_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_rgc_2e_tok', '--file', 'small_llm_results/rgc_gemma_2e.pkl', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_gc_2e_ask', '--file', 'small_llm_results/gc_gemma_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'gemma_gc_2e_tok', '--file', 'small_llm_results/gc_gemma_2e.pkl', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_rgc_2e_ask', '--file', 'small_llm_results/rgc_llama_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_rgc_2e_tok', '--file', 'small_llm_results/rgc_llama_2e.pkl', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_gc_2e_ask', '--file', 'small_llm_results/gc_llama_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'llama_gc_2e_tok', '--file', 'small_llm_results/gc_llama_2e.pkl', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_rgc_2e_ask', '--file', 'small_llm_results/rgc_phi3_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_rgc_2e_tok', '--file', 'small_llm_results/rgc_phi3_2e.pkl', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_gc_2e_ask', '--file', 'small_llm_results/gc_phi3_2e.csv', '--output', 'results_ft.xlsx']),
    ('code/evaluate_small_llms.py', ['--setting_name', 'phi3_gc_2e_tok', '--file', 'small_llm_results/gc_phi3_2e.pkl', '--output', 'results_ft.xlsx']),
]

for script, args in scripts_and_args:
    try:
        # Capture the output and errors
        result = subprocess.run(['python', script] + args, text=True, capture_output=True, check=True)
        print(f"Output of {script}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error in {script}: {e}")

