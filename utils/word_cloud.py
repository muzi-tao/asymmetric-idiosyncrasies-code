# Simplified Python script to generate separate word clouds from JSON 'answer' fields
# Current Date: Sunday, April 20, 2025 at 9:39 PM PDT

import json
import os
import string
import sys
import traceback
import nltk

# --- Essential Dependencies Check ---
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing required library: {e}\nRun: pip install nltk wordcloud matplotlib")
    sys.exit(1)

# --- Configuration ---
BASE_DIR = "dataset/text/"
FILE_NAMES = [
    # "claude-3-5-sonnet_prompts.json",
    # "gemini-1.5-pro_prompts.json",
    # "gpt-4o_prompts.json",
    "internvl3.5_prompts.json",  
    "qwen3-vl_prompts.json"
]
OUTPUT_DIR = "word_clouds_output" # Directory where PDF files will be saved
WC_WIDTH = 1000
WC_HEIGHT = 500
WC_BACKGROUND = "white"
WC_MAX_WORDS = 150

# --- NLTK Setup (Fixed for newer NLTK versions) ---
print("--- Initializing NLTK ---")
nltk_resources_needed = ['stopwords', 'punkt', 'punkt_tab']
try:
    # Try to check if resources exist
    for resource in ['corpora/stopwords', 'tokenizers/punkt', 'tokenizers/punkt_tab']:
        try:
            nltk.data.find(resource)
        except LookupError:
            pass  # Will handle downloads below
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOP_WORDS_SET = set(stopwords.words('english'))
    # Add common, less informative words often found in image descriptions
    STOP_WORDS_SET.update(['image', 'photo', 'picture', 'background', 'shows', 'view', 'scene', 'looks', 'like', 'appears'])
    print("NLTK components ready.")
except (LookupError, ImportError) as e:
    print(f"\nNLTK resources missing. Downloading required components...")
    try:
        # Download all required NLTK packages for newer versions
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # New requirement for newer NLTK
        nltk.download('stopwords', quiet=True)
        
        # Re-import after download
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        STOP_WORDS_SET = set(stopwords.words('english'))
        STOP_WORDS_SET.update(['image', 'photo', 'picture', 'background', 'shows', 'view', 'scene', 'looks', 'like', 'appears'])
        print("NLTK download successful, components ready.")
    except Exception as download_exc:
        print(f"ERROR: Failed to download NLTK data: {download_exc}")
        print("Please try manual download:")
        print(">>> import nltk")
        print(">>> nltk.download('punkt')")
        print(">>> nltk.download('punkt_tab')")
        print(">>> nltk.download('stopwords')")
        sys.exit(1)


# --- Helper Function ---
def preprocess_text(text):
    """Cleans text, removes stopwords, returns space-separated valid words."""
    if not isinstance(text, str): return ""
    text = text.lower()
    
    # Expand common contractions before removing punctuation
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not", "wouldn't": "would not",
        "shouldn't": "should not", "won't": "will not", "aren't": "are not",
        "isn't": "is not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would"
    }
    
    # Replace contractions
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove possessive 's (e.g., "cat's" → "cat")
    text = text.replace("'s ", " ")
    text = text.replace("'s", "")
    
    # Now remove all remaining punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha() and len(word) > 1 and word not in STOP_WORDS_SET]
    return " ".join(words)

# --- Main Processing ---
print(f"\n--- Starting Word Cloud Generation ---")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output dir silently if needed
print(f"Output directory: '{os.path.abspath(OUTPUT_DIR)}'")

for file_name in FILE_NAMES:
    file_path = os.path.join(BASE_DIR, file_name)
    print(f"\nProcessing: {file_name}...")

    combined_text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                 print(f"  Warning: Expected list in JSON, got {type(data)}. Skipping.")
                 continue
            # Combine 'answer' fields directly
            combined_text = " ".join(item['answer'] for item in data if isinstance(item, dict) and 'answer' in item and isinstance(item['answer'], str))

    except FileNotFoundError:
        print(f"  ERROR: File not found.")
        continue
    except Exception as e:
        print(f"  ERROR reading or parsing file: {e}")
        traceback.print_exc()
        continue

    # Preprocess and check if any text remains
    processed_text = preprocess_text(combined_text)
    if not processed_text:
        print("  No valid words found after preprocessing. Skipping cloud.")
        continue

    # Generate and Save Word Cloud
    print(f"  Generating word cloud...")
    try:
        wordcloud = WordCloud(
            width=WC_WIDTH,
            height=WC_HEIGHT,
            background_color=WC_BACKGROUND,
            max_words=WC_MAX_WORDS,
            stopwords=None, # Already handled
            collocations=True
        ).generate(processed_text)

        output_filename = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}_wordcloud.pdf")

        # Save as PDF using matplotlib
        plt.figure(figsize=(WC_WIDTH/100, WC_HEIGHT/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  Word cloud saved to: {output_filename}")

    except Exception as e:
        print(f"  ERROR generating/saving word cloud: {e}")
        traceback.print_exc()


print("\n--- Word Cloud Generation Process Complete ---")