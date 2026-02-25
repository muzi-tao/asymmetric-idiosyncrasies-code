import os
import json
import random
import logging
from tqdm import tqdm

# Reduce noisy logs before importing vLLM
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")

from vllm import LLM, SamplingParams

# Suppress vllm and ray logging
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.core").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)

# Model setup
model_name = "Qwen/Qwen2.5-7B-Instruct"
llm = LLM(model=model_name, dtype="float16",
          tensor_parallel_size=1, gpu_memory_utilization=0.92, trust_remote_code=True)
sampling = SamplingParams(max_tokens=256, temperature=0.0)

# Paths
src_path = "../dataset/text_prompts.json"
tgt_path = "detail/"
tgt_file = os.path.join(tgt_path, "detail_ranking.json")
print("Source path:", src_path)
print("Target path:", tgt_file)

os.makedirs(tgt_path, exist_ok=True)

# Load data
try:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Source file not found at {src_path}")
    exit()

# Load existing results if any to resume
if os.path.exists(tgt_file):
    with open(tgt_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

prompt_template = """You are an expert in evaluating text quality. Your task is to rank the following three texts based on their level of detail. "Level of detail" refers to the amount of specific, factual, and descriptive information provided. A more detailed text will be more comprehensive and specific.

Here are the three texts:

Text A:
---
{text_a}
---

Text B:
---
{text_b}
---

Text C:
---
{text_c}
---

Please rank them from most detailed to least detailed. Provide your ranking as a JSON object with a single key "ranking" which is a list of the text labels in order. For example: {{"ranking": ["Text B", "Text C", "Text A"]}}. Do not provide any other explanation or text."""

model_keys = ["gemini", "claude", "gpt4"]

items_to_process = {k: v for k, v in data.items() if k not in results}

valid_items_to_process = []
for prompt_id, item in items_to_process.items():
    answers = {key: item[key]['answer'] for key in model_keys if key in item and 'answer' in item[key]}
    if len(answers) == 3:
        valid_items_to_process.append((prompt_id, item))

item_list = valid_items_to_process
batch_size = 32

# Single global progress bar
with tqdm(total=len(item_list), desc="Generating rankings", disable=False) as pbar:
    for i in range(0, len(item_list), batch_size):
        batch_items = item_list[i:i+batch_size]
        batch_prompts = []
        batch_ids_info = []

        if not batch_items:
            continue

        for prompt_id, item in batch_items:
            answers = {key: item[key]['answer'] for key in model_keys}
            
            shuffled_keys = list(answers.keys())
            random.shuffle(shuffled_keys)
            
            mapping = {"Text A": shuffled_keys[0], "Text B": shuffled_keys[1], "Text C": shuffled_keys[2]}
            
            prompt = prompt_template.format(
                text_a=answers[mapping["Text A"]],
                text_b=answers[mapping["Text B"]],
                text_c=answers[mapping["Text C"]]
            )
            batch_prompts.append(prompt)
            batch_ids_info.append({
                "id": prompt_id,
                "mapping": mapping
            })

        if not batch_prompts:
            pbar.update(len(batch_items))
            continue
            
        # Disable vLLM internal tqdm for this call
        outs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        
        for j, out in enumerate(outs):
            raw_output = out.outputs[0].text.strip()
            current_id_info = batch_ids_info[j]
            prompt_id = current_id_info['id']
            mapping = current_id_info['mapping']
            
            ranking = []
            try:
                if "```json" in raw_output:
                    json_str = raw_output.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_output:
                    json_str = raw_output.split("```")[1].split("```")[0].strip()
                else:
                    json_str = raw_output

                parsed_output = json.loads(json_str)
                ranked_labels = parsed_output.get("ranking", [])
                ranking = [mapping[label] for label in ranked_labels if label in mapping]
            except (json.JSONDecodeError, IndexError) as e:
                ranking = ["parsing_error"]

            results[prompt_id] = {
                "ranking": ranking,
                "raw_output": raw_output,
                "mapping": mapping
            }

        # Save results periodically
        with open(tgt_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        pbar.update(len(batch_items))

if not items_to_process:
    print("No new items to process. Results are up to date.")
else:
    print(f"Finished ranking. Results saved to {tgt_file}")
            