import json
from collections import defaultdict
import os

def analyze_ranking(json_file_path):
    """
    Analyzes the ranking data from a JSON file.

    Args:
        json_file_path (str): The path to the JSON file with ranking data.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    # To store rank counts: {model: {1: count, 2: count, ...}}
    rank_counts = defaultdict(lambda: defaultdict(int))
    
    # To store scores: {model: score}
    scores = defaultdict(int)
    
    models = set()
    
    # Track parsing errors
    parsing_error_prompts = []
    total_prompts = len(data)
    successful_prompts = 0

    # Process each prompt's ranking
    for prompt_id, result in data.items():
        ranking = result.get("ranking")
        if ranking:
            # Check if any of the models in ranking is "parsing_error"
            if "parsing_error" in ranking:
                parsing_error_prompts.append(prompt_id)
            else:
                successful_prompts += 1
            
            for i, model in enumerate(ranking):
                models.add(model)
                rank = i + 1
                rank_counts[model][rank] += 1
                
                # Assign scores (e.g., 3 for rank 1, 2 for rank 2, 1 for rank 3)
                score = len(ranking) - i
                scores[model] += score

    sorted_models = sorted(list(models))

    # --- Print Rank Frequency Table ---
    print("--- Rank Frequencies ---")
    header = f"{'Model':<15}" + "".join([f"{f'Rank {i+1}':<10}" for i in range(len(sorted_models))])
    print(header)
    print("-" * len(header))

    for model in sorted_models:
        row = f"{model:<15}"
        for i in range(len(sorted_models)):
            rank = i + 1
            count = rank_counts[model].get(rank, 0)
            row += f"{count:<10}"
        print(row)
    
    print("\n" + "="*40 + "\n")

    # --- Print Scores ---
    print("--- Overall Scores (3 points for 1st, 2 for 2nd, 1 for 3rd) ---")
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    for model, score in sorted_scores:
        print(f"{model:<15}: {score} points")
    
    print("\n" + "="*40 + "\n")
    
    # --- Print Parsing Error Information ---
    print("--- Parsing Error Analysis ---")
    parsing_error_count = len(parsing_error_prompts)
    success_rate = (successful_prompts / total_prompts) * 100 if total_prompts > 0 else 0
    error_rate = (parsing_error_count / total_prompts) * 100 if total_prompts > 0 else 0
    
    print(f"Total prompts processed: {total_prompts}")
    print(f"Successfully parsed: {successful_prompts} ({success_rate:.2f}%)")
    print(f"Parsing errors: {parsing_error_count} ({error_rate:.2f}%)")
    
    if parsing_error_prompts:
        print(f"\nPrompt IDs with parsing errors:")
        for i, prompt_id in enumerate(parsing_error_prompts, 1):
            print(f"  {i}. {prompt_id}")
        
        # Show first few parsing error examples
        print(f"\nFirst {min(3, len(parsing_error_prompts))} parsing error examples:")
        for i, prompt_id in enumerate(parsing_error_prompts[:3], 1):
            result = data[prompt_id]
            print(f"  {i}. {prompt_id}:")
            print(f"     Raw output: {result.get('raw_output', 'N/A')[:100]}{'...' if len(result.get('raw_output', '')) > 100 else ''}")
            print(f"     Ranking: {result.get('ranking', 'N/A')}")
            print()
    else:
        print("\nNo parsing errors found!")

if __name__ == "__main__":
    # Assuming the script is in 'analysis' and the data in 'detail'
    # under the same parent directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The user mentioned 'detail/detail_ranking.json'. Let's assume it's relative
    # to the project root, and the script is in cap-to-img/analysis
    project_root = os.path.join(script_dir, '..', '..') # up two levels from cap-to-img/analysis
    json_path = os.path.join(project_root, 'detail/detail_ranking.json')
    
    # Let's try to find the file from a few common locations
    possible_paths = [
        'detail/detail_ranking.json', # relative to project root
        os.path.join(os.path.dirname(__file__), '..', 'detail', 'detail_ranking.json'), # cap-to-img/detail/
        '../detail/detail_ranking.json', # relative to script
    ]
    
    final_path = None
    for path in possible_paths:
        # Normalize path for comparison
        abs_path = os.path.abspath(path)
        if os.path.exists(path):
            final_path = path
            break

    if final_path:
        print(f"Found ranking file at: {os.path.abspath(final_path)}")
        analyze_ranking(final_path)
    else:
        print("Error: Could not find 'detail/detail_ranking.json'.")
        print("Please check the file path.")
