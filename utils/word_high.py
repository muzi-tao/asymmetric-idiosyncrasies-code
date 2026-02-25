# Full Python script to analyze word frequency SEPARATELY for each JSON file
# Corrected to include 'punkt_tab' NLTK resource
# Current Date: Sunday, April 20, 2025 at 9:15 PM PDT

import json
import os
import string
from collections import Counter
import nltk
import sys # Import sys for potentially exiting on critical errors
import traceback # Import traceback for better error printing

# --- Configuration ---
# !!! IMPORTANT: Adjust BASE_DIR if your script is not in the directory
#     immediately outside 'cap-to-img/' !!!
BASE_DIR = "dataset/text/"  # Directory containing the JSON files
FILE_NAMES = [
    "claude-3-5-sonnet_prompts.json",
    "gemini-1.5-pro_prompts.json",
    "gpt-4o_prompts.json",
    "internvl3.5_prompts.json",  
    "qwen3-vl_prompts.json"
]
NUM_TOP_WORDS = 20  # How many of the most frequent words to display per file

# --- NLTK Setup ---
# (This part remains the same - ensures necessary data is available)
required_nltk_resources = {
    'corpora/stopwords': 'stopwords',
    'tokenizers/punkt': 'punkt',
    'tokenizers/punkt_tab': 'punkt_tab'
}

print("--- Initializing NLTK ---")
nltk_ready = True
for resource_path, resource_id in required_nltk_resources.items():
    try:
        print(f"Checking for NLTK resource: '{resource_id}' (Path: {resource_path})...")
        nltk.data.find(resource_path)
        print(f"Resource '{resource_id}' found.")
    except LookupError:
        print(f"NLTK resource '{resource_id}' not found. Attempting download...")
        try:
            nltk.download(resource_id, quiet=True)
            print(f"Successfully downloaded NLTK resource '{resource_id}'.")
            nltk.data.find(resource_path)
            print(f"Verified resource '{resource_id}' is now available.")
        except Exception as download_exc:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Failed to download required NLTK resource '{resource_id}'.")
            print(f"Reason: {download_exc}")
            print("Script cannot proceed without this resource. Exiting.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            nltk_ready = False
            break # Stop checking if one download fails

if not nltk_ready:
     sys.exit(1) # Exit if NLTK setup failed

# Import necessary NLTK components AFTER attempting downloads
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    print("Successfully imported NLTK components (stopwords, word_tokenize).")
    STOP_WORDS_SET = set(stopwords.words('english'))
except Exception as e:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: Failed to import or setup NLTK components after download checks: {e}")
    print("Ensure NLTK is installed and data downloaded correctly. Exiting.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    sys.exit(1)

print("--- NLTK Initialization Complete ---")


# --- Helper Functions ---
# (This function remains the same)
def clean_and_tokenize(text):
    """
    Converts text to lowercase, removes punctuation, tokenizes,
    and removes stop words and non-alphabetic tokens.
    """
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text_lower.translate(translator)
    tokens = word_tokenize(text_no_punct)
    filtered_tokens = [
        word for word in tokens
        if word.isalpha() and word not in STOP_WORDS_SET
    ]
    return filtered_tokens

# --- Main Processing Loop (Separate Analysis per File) ---
print(f"\n--- Starting Separate Word Frequency Analysis ---")
print(f"Looking for JSON files in directory: '{os.path.abspath(BASE_DIR)}'")

analysis_results = {} # To store results per file if needed later
missing_files = []
encountered_errors = False

# Loop through each specified file name
for file_name in FILE_NAMES:
    file_path = os.path.join(BASE_DIR, file_name)
    print(f"\n==========================================================")
    print(f"Analyzing File: {file_name}")
    print(f"Full Path: {file_path}")
    print(f"==========================================================")

    current_file_words = [] # Reset word list for each new file
    file_processed_successfully = False
    file_entries_checked = 0
    file_answers_extracted = 0
    file_had_processing_errors = False

    try:
        # Open and load the JSON data for the current file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_processed_successfully = True # Mark as successfully opened/loaded

            # Check if the data is a list
            if not isinstance(data, list):
                print(f"  Warning: Expected a list of objects, but got {type(data)}. Skipping analysis for this file.")
                file_processed_successfully = False # Cannot process if not a list
                continue # Go to the next file in the outer loop

            print(f"  Successfully loaded JSON data ({len(data)} entries found). Processing entries...")
            # Iterate through each item in the current file's data
            for item_index, item in enumerate(data):
                file_entries_checked += 1
                try:
                    # Process each item within its own try block
                    if isinstance(item, dict) and 'answer' in item and isinstance(item['answer'], str):
                        answer_text = item['answer']
                        words = clean_and_tokenize(answer_text)
                        current_file_words.extend(words)
                        file_answers_extracted += 1
                    elif isinstance(item, dict):
                         # Optional: Add warnings for malformed entries if needed
                         pass # Keep output cleaner for now
                    elif not isinstance(item, dict):
                         print(f"  Warning: Found non-dictionary item at list index {item_index}.")

                except Exception as item_exc:
                    # Catch errors during tokenization/processing of a single item
                    print(f"  ------------------------------------------------------")
                    print(f"  ERROR processing item at index {item_index} (ID: {item.get('index', 'N/A')}).")
                    print(f"  Error Type: {type(item_exc)}")
                    print(f"  Error Details: {item_exc}")
                    # Optionally uncomment traceback for debugging specific item errors
                    # print(f"  Traceback:")
                    # traceback.print_exc()
                    print(f"  Skipping this item and continuing...")
                    print(f"  ------------------------------------------------------")
                    file_had_processing_errors = True
                    encountered_errors = True # Mark that some error occurred globally

            print(f"  Finished processing entries for {file_name}.")
            print(f"  Checked: {file_entries_checked} entries.")
            print(f"  Extracted & Cleaned Answers: {file_answers_extracted} entries.")

    except FileNotFoundError:
        print(f"  ERROR: File not found at {file_path}")
        missing_files.append(file_name)
        encountered_errors = True
        continue # Skip to the next file
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON format in {file_name} near line {e.lineno}, col {e.colno}.")
        print(f"  Details: {e}")
        encountered_errors = True
        continue # Skip to the next file
    except Exception as e:
        # Catch unexpected errors during file open/load
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  An unexpected error occurred while opening or loading {file_path}: {e}")
        print(f"  Error Type: {type(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        encountered_errors = True
        continue # Skip to the next file

    # --- Frequency Calculation for the CURRENT file ---
    if file_processed_successfully and current_file_words:
        print(f"\n  --- Top {NUM_TOP_WORDS} Words for {file_name} ---")
        word_counts = Counter(current_file_words)
        most_common_words = word_counts.most_common(NUM_TOP_WORDS)
        if not most_common_words:
             print("    (No frequent words found based on criteria)")
        else:
            for rank, (word, count) in enumerate(most_common_words, 1):
                print(f"    {rank}. {word}: {count}")
        analysis_results[file_name] = most_common_words # Store results

    elif file_processed_successfully and not current_file_words:
        print(f"\n  --- Word Frequency for {file_name} ---")
        print(f"    No valid words extracted after cleaning.")
        if file_had_processing_errors:
            print(f"    Note: Errors occurred during item processing for this file.")
        analysis_results[file_name] = [] # Store empty result

    elif not file_processed_successfully:
         print(f"\n  --- Word Frequency for {file_name} ---")
         print(f"    Analysis skipped due to errors loading or reading the file.")
         analysis_results[file_name] = None # Indicate failure

# --- Final Summary ---
print(f"\n==========================================================")
print(f"--- Overall Analysis Complete ---")
if missing_files:
    print(f"Warning: The following files were specified but not found: {', '.join(missing_files)}")
if encountered_errors:
    print(f"Warning: One or more errors occurred during processing. Review output above for details.")
print(f"Displayed top {NUM_TOP_WORDS} words for each analyzed file.")
print(f"==========================================================")