import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
DATA_DIR = 'dataset/text'  # Directory containing the JSON files
OUTPUT_DIR = 'analysis'  # Directory to save the analysis results
JSON_FILES = [
    'claude-3-5-sonnet_prompts.json',
    'gemini-1.5-pro_prompts.json',
    'gpt-4o_prompts.json',
    # "internvl3.5_prompts.json",  
    "qwen3-vl_prompts.json"
]
TOP_N = 30 # Number of top phrases to display per file
NGRAM_RANGE = (2, 3) # Consider phrases of 2 and 3 words (bigrams and trigrams)
# --- End Configuration ---

def analyze_single_file(filepath, top_n=20, ngram_range=(2, 3)):
    """
    Loads a single JSON file, performs TF-IDF analysis on the 'answer' field,
    and returns the top N phrases.

    Args:
        filepath (str): The full path to the JSON file.
        top_n (int): The number of top phrases to return.
        ngram_range (tuple): The range of n-grams to consider.

    Returns:
        list: A list of tuples (phrase, score), sorted by score, or None on failure.
    """
    all_answers = []
    print(f"--- Analyzing File: {os.path.basename(filepath)} ---")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'answer' in item and isinstance(item['answer'], str):
                    all_answers.append(item['answer'])
                else:
                    print(f"  Warning: Skipping item - missing or invalid 'answer' field: {item.get('index', 'N/A')}")
    except FileNotFoundError:
        print(f"  Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from file - {filepath}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while loading {filepath}: {e}")
        return None

    if not all_answers:
        print("  Error: No 'answer' text could be extracted from this file.")
        return None

    print(f"  Extracted text from {len(all_answers)} answers.")

    # TF-IDF requires more than one document for IDF calculation to be meaningful
    if len(all_answers) < 2:
        print("  Warning: Only one 'answer' found. TF-IDF results (especially IDF) may not be meaningful.")
        # Optionally, you could switch to simple term frequency here if needed
        # For now, we'll proceed but the scores might just reflect term frequency.

    print(f"  Calculating TF-IDF for phrases (n-grams: {ngram_range})...")

    # Initialize TF-IDF Vectorizer for this file
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        use_idf=True, # Keep IDF, but be aware of the warning above if only 1 doc
        smooth_idf=True
    )

    # Fit and transform for this file's data
    try:
        tfidf_matrix = vectorizer.fit_transform(all_answers)
    except ValueError as e:
        print(f"  Error during TF-IDF calculation: {e}")
        print("  This might happen if documents become empty after preprocessing.")
        return None

    # Get feature names (phrases) for this file
    feature_names = vectorizer.get_feature_names_out()

    if not feature_names.size:
         print("  Warning: No features (phrases) found after vectorization. Check stop words or text content.")
         return [] # Return empty list if no features

    # Sum TF-IDF scores for each phrase across all answers *within this file*
    sum_tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Pair scores with feature names
    scored_features = []
    for i, feature in enumerate(feature_names):
         # Ensure index is within bounds (should be, but good practice)
         if i < len(sum_tfidf_scores):
              scored_features.append((sum_tfidf_scores[i], feature))
         else:
              print(f"Warning: Index mismatch between features and scores at index {i}")


    # Sort by score (descending)
    scored_features.sort(key=lambda x: x[0], reverse=True)

    print(f"  TF-IDF calculation complete. Found {len(feature_names)} unique phrases in this file.")

    # Return top N
    return scored_features[:top_n]

# --- Main Execution ---
if __name__ == "__main__":
    # Check if base directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Base directory not found - {DATA_DIR}")
        print("Please ensure the directory exists and contains the JSON files.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, 'tfidf_analysis_results.txt')
        out = open(output_file, 'w', encoding='utf-8')
        
        # Loop through each specified JSON file
        for filename in JSON_FILES:
            filepath = os.path.join(DATA_DIR, filename)

            # Analyze the current file
            top_phrases = analyze_single_file(filepath, top_n=TOP_N, ngram_range=NGRAM_RANGE)

            # Print results for the current file
            if top_phrases is not None: # Check if analysis was successful
                if top_phrases: # Check if any phrases were returned
                    print(f"\n  --- Top {len(top_phrases)} Phrases for {os.path.basename(filepath)} ---")
                    out.write(f"\n  --- Top {len(top_phrases)} Phrases for {os.path.basename(filepath)} ---\n")
                    print(f"  {'Rank':<5} {'Score':<10} {'Phrase'}")
                    print("  " + "-" * (5 + 1 + 10 + 1 + 40)) # Adjust width as needed
                    for i, (score, phrase) in enumerate(top_phrases):
                        print(f"  {i+1:<5} {score:<10.4f} {phrase}")
                        out.write(f"  {i+1:<5} {score:<10.4f} {phrase}\n")
                else:
                     print("  No top phrases found or returned for this file.")
            else:
                # Error message was already printed inside analyze_single_file
                print(f"  Skipping results for {os.path.basename(filepath)} due to errors.")

            print("\n" + "="*70 + "\n") # Separator between file results

        print("Analysis complete for all specified files.")
        out.close()
        print(f"Results saved to: {output_file}")