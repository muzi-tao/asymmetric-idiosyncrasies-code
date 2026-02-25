import os
import json
import logging
import re
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
tgt_path = "texture/llm/"
tgt_file = os.path.join(tgt_path, "analysis_results.json")
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

# Create texture analysis prompt template
texture_analysis_prompt_template = """You are an expert in analyzing texture vocabulary usage in text. Your task is to read a caption and detect whether ANY texture-related words appear, and classify them as basic or nuanced.

**Basic textures** are simple, everyday texture terms that most people use commonly - basic tactile descriptions that are widely understood.

**Nuanced textures** are more sophisticated, specific, or descriptive texture terms - including:
- Specific material textures (e.g., "velvet", "corduroy", "burlap")
- Technical/artistic texture descriptions (e.g., "stippled", "embossed", "burnished")
- Complex tactile descriptions (e.g., "silky", "grainy", "porous", "fibrous")
- Surface finish terms (e.g., "matte", "glossy", "satin", "brushed")
- Detailed texture qualities (e.g., "coarse-grained", "fine-textured", "weathered")

Instructions:
- Look for texture/surface/material words in their tactile/physical sense only
- Include words describing how something feels, looks textured, or surface qualities
- Match whole words, ignore substrings
- Treat hyphens/slashes as separators
- Use your linguistic knowledge to judge what constitutes "basic" vs "nuanced" texture vocabulary

Return ONLY a valid JSON object with the count of each type. Do not include any explanatory text before or after the JSON:
{{"basic_count": <number>, "nuanced_count": <number>}}

Examples:
{{"basic_count": 0, "nuanced_count": 0}}
{{"basic_count": 2, "nuanced_count": 1}}
{{"basic_count": 1, "nuanced_count": 3}}

Caption to analyze:
{caption_text}"""

model_keys = ["gemini", "claude", "gpt4"]

# Prepare items to process
items_to_process = {k: v for k, v in data.items() if k not in results}

# Process each model's responses separately
valid_items_to_process = []
for prompt_id, item in items_to_process.items():
    for model_key in model_keys:
        if model_key in item and 'answer' in item[model_key]:
            valid_items_to_process.append((prompt_id, model_key, item))

item_list = valid_items_to_process
batch_size = 32

# Single global progress bar
with tqdm(total=len(item_list), desc="Analyzing textures", disable=False) as pbar:
    for i in range(0, len(item_list), batch_size):
        batch_items = item_list[i:i+batch_size]
        batch_prompts = []
        batch_ids_info = []

        if not batch_items:
            continue

        for prompt_id, model_key, item in batch_items:
            answer_text = item[model_key]['answer']
            
            prompt = texture_analysis_prompt_template.format(caption_text=answer_text)
            batch_prompts.append(prompt)
            batch_ids_info.append({
                "id": prompt_id,
                "model": model_key,
                "caption": answer_text
            })

        if not batch_prompts:
            pbar.update(len(batch_items))
            continue
            
        # Disable vLLM internal tqdm for this call
        outs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        
        for j, out in enumerate(outs):
            raw_output = out.outputs[0].text.strip()
            current_info = batch_ids_info[j]
            prompt_id = current_info['id']
            model_key = current_info['model']
            
            basic_count, nuanced_count = 0, 0
            try:
                # Clean the raw output first
                cleaned_output = raw_output.strip()
                
                # Remove control characters that can break JSON parsing
                cleaned_output = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_output)
                
                # Try multiple extraction strategies
                json_str = None
                
                # Strategy 1: Extract from markdown code blocks
                if "```json" in cleaned_output:
                    parts = cleaned_output.split("```json")
                    if len(parts) > 1:
                        json_str = parts[1].split("```")[0].strip()
                elif "```" in cleaned_output:
                    parts = cleaned_output.split("```")
                    if len(parts) >= 3:
                        json_str = parts[1].strip()
                
                # Strategy 2: Look for JSON-like patterns with regex
                if json_str is None:
                    json_match = re.search(r'\{[^}]*"basic_count"[^}]*"nuanced_count"[^}]*\}', cleaned_output)
                    if json_match:
                        json_str = json_match.group(0)
                
                # Strategy 3: Try the whole cleaned output
                if json_str is None:
                    json_str = cleaned_output
                
                # Strategy 4: Extract just the first complete JSON object
                if json_str and not json_str.startswith('{'):
                    brace_start = json_str.find('{')
                    if brace_start != -1:
                        json_str = json_str[brace_start:]
                        # Find matching closing brace
                        brace_count = 0
                        end_pos = 0
                        for i, char in enumerate(json_str):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                        if end_pos > 0:
                            json_str = json_str[:end_pos]
                
                if json_str:
                    parsed_output = json.loads(json_str)
                    basic_count = int(parsed_output.get("basic_count", 0))
                    nuanced_count = int(parsed_output.get("nuanced_count", 0))
                else:
                    print(f"No JSON found for {prompt_id}-{model_key}")
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Last resort: try to extract numbers with regex
                basic_match = re.search(r'"basic_count":\s*(\d+)', raw_output)
                nuanced_match = re.search(r'"nuanced_count":\s*(\d+)', raw_output)
                
                if basic_match:
                    basic_count = int(basic_match.group(1))
                if nuanced_match:
                    nuanced_count = int(nuanced_match.group(1))
                    
                # Only print detailed error if extraction also failed
                if basic_count == 0 and nuanced_count == 0:
                    print(f"Complete parsing failure for {prompt_id}-{model_key}: {e}")
                    print(f"Raw output (first 100 chars): {raw_output[:100]}...")
                    if 'json_str' in locals() and json_str:
                        print(f"Extracted JSON attempt: {json_str[:100]}...")
                else:
                    print(f"Fallback parsing succeeded for {prompt_id}-{model_key}: basic={basic_count}, nuanced={nuanced_count}")

            # Initialize prompt_id entry if not exists
            if prompt_id not in results:
                results[prompt_id] = {}
            
            results[prompt_id][model_key] = {
                "basic_count": basic_count,
                "nuanced_count": nuanced_count,
                "has_basic": basic_count > 0,
                "has_nuanced": nuanced_count > 0,
                "raw_output": raw_output,
                "caption": current_info['caption']
            }

        # Save results periodically
        with open(tgt_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        pbar.update(len(batch_items))

if not items_to_process:
    print("No new items to process. Results are up to date.")
else:
    print(f"Finished texture analysis. Results saved to {tgt_file}")

# Generate summary statistics
print("\nGenerating summary statistics...")
summary = {
    "total_entries": len(results),
    "model_summaries": {}
}

for model_key in model_keys:
    entries_with_basic = 0
    entries_with_nuanced = 0
    entries_with_both = 0
    total_basic_words = 0
    total_nuanced_words = 0
    total_processed = 0
    
    for prompt_id, entry in results.items():
        if model_key in entry:
            total_processed += 1
            basic_count = entry[model_key]['basic_count']
            nuanced_count = entry[model_key]['nuanced_count']
            has_basic = entry[model_key]['has_basic']
            has_nuanced = entry[model_key]['has_nuanced']
            
            total_basic_words += basic_count
            total_nuanced_words += nuanced_count
            
            if has_basic:
                entries_with_basic += 1
            if has_nuanced:
                entries_with_nuanced += 1
            if has_basic and has_nuanced:
                entries_with_both += 1
    
    summary["model_summaries"][model_key] = {
        "total_processed": total_processed,
        "entries_with_basic_textures": entries_with_basic,
        "entries_with_nuanced_textures": entries_with_nuanced,
        "entries_with_both_textures": entries_with_both,
        "total_basic_words": total_basic_words,
        "total_nuanced_words": total_nuanced_words,
        "avg_basic_per_entry": (total_basic_words / total_processed) if total_processed > 0 else 0,
        "avg_nuanced_per_entry": (total_nuanced_words / total_processed) if total_processed > 0 else 0,
        "basic_presence_percentage": (entries_with_basic / total_processed * 100) if total_processed > 0 else 0,
        "nuanced_presence_percentage": (entries_with_nuanced / total_processed * 100) if total_processed > 0 else 0
    }

# Save summary
summary_file = os.path.join(tgt_path, "texture_analysis_summary.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Summary saved to {summary_file}")

# Print summary
print("\n=== TEXTURE ANALYSIS SUMMARY ===")
for model_key, model_summary in summary["model_summaries"].items():
    print(f"\n{model_key.upper()}:")
    print(f"  Total processed: {model_summary['total_processed']}")
    print(f"  Entries with basic textures: {model_summary['entries_with_basic_textures']} ({model_summary['basic_presence_percentage']:.1f}%)")
    print(f"  Entries with nuanced textures: {model_summary['entries_with_nuanced_textures']} ({model_summary['nuanced_presence_percentage']:.1f}%)")
    print(f"  Entries with both: {model_summary['entries_with_both_textures']}")
    print(f"  Total basic words: {model_summary['total_basic_words']} (avg: {model_summary['avg_basic_per_entry']:.2f} per entry)")
    print(f"  Total nuanced words: {model_summary['total_nuanced_words']} (avg: {model_summary['avg_nuanced_per_entry']:.2f} per entry)")
