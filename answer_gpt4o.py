from openai import OpenAI
import os
import time
import csv
from tqdm import tqdm
import base64
from datetime import datetime
import json
from PIL import Image
from io import BytesIO

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please export it before running, e.g. "
        "export OPENAI_API_KEY='your_key'"
    )
client = OpenAI(api_key=openai_api_key)

MODEL="gpt-4o"

def create_output_directory():
    """Create and return path to output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"answers/{MODEL}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_image_media_type(image_path):
    """Determine the media type of the image file by actually reading it"""
    try:
        with Image.open(image_path) as img:
            # Get actual format from the image
            actual_format = img.format.lower()
            
            # Map PIL formats to MIME types
            mime_types = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'bmp': 'image/bmp'
            }
            
            return mime_types.get(actual_format) or f'image/{actual_format}'
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")

def encode_image(image_path):
    """Encode image file to base64 and return media type after validating"""
    try:
        # Verify and get format using PIL
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles PNG with alpha channel)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Save to bytes with proper format
            buffer = BytesIO()
            img.save(buffer, format=img.format)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            media_type = get_image_media_type(image_path)
            return base64_image, media_type
            
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def get_completion_with_image(prompt, image_path, model=MODEL, max_tokens=1024, temperature=1.0):
    base64_image, media_type = encode_image(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}"
                    }
                },
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def read_image_files_from_tsv(tsv_path):
    """Read image filenames from the data info TSV file"""
    image_files = []
    with open(tsv_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            image_files.append(row['Filename'])
    return image_files

def load_progress(output_file):
    """Load previously processed images from an existing TSV file"""
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    processed_images.add(row[0])  # First column is image filename
    return processed_images

def save_progress_file(output_dir, progress_data):
    """Save progress information to a JSON file"""
    progress_file = os.path.join(output_dir, "progress.json")
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress_file(output_dir):
    """Load progress information from JSON file"""
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def verify_image(image_path):
    """Verify if the image is valid and return its format"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image
            return True, img.format
    except Exception as e:
        return False, str(e)

def save_error_log(output_dir, error_data):
    """Save error information to a JSON file"""
    error_file = os.path.join(output_dir, "error_log.json")
    
    # Load existing errors if file exists
    existing_errors = {}
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            existing_errors = json.load(f)
    
    # Update with new errors
    existing_errors.update(error_data)
    
    # Save updated errors
    with open(error_file, 'w') as f:
        json.dump(existing_errors, f, indent=2)

def process_prompt(prompt, image_files, image_directory, output_dir, prompt_num, max_tokens=1024, temperature=0.0):
    """Process a single prompt for all images with progress and error tracking"""
    output_file = os.path.join(output_dir, f"prompt_{prompt_num}.tsv")
    processed_images = load_progress(output_file)
    
    # Initialize error tracking for this prompt
    error_data = {
        f"prompt_{prompt_num}": {
            "prompt_text": prompt,
            "errors": {}
        }
    }
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(["Image", "Prompt", "Answer"])
    
    print(f"\nProcessing prompt {prompt_num}: {prompt}")
    print(f"Resuming from {len(processed_images)} previously processed images")
    
    # Process images that haven't been processed yet
    remaining_images = [img for img in image_files if img not in processed_images]
    
    for image_file in tqdm(remaining_images):
        image_path = os.path.join(image_directory, image_file)
        error_info = {}
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Verify image
            is_valid, format_or_error = verify_image(image_path)
            if not is_valid:
                raise ValueError(f"Invalid image: {format_or_error}")
            
            # Try to process image
            answer = get_completion_with_image(prompt, image_path, 
                                             max_tokens=max_tokens, 
                                             temperature=temperature)
            
            # Write successful result
            with open(output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                writer.writerow([image_file, prompt, answer])
            
        except Exception as e:
            # Collect error information
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "image_path": image_path,
                "prompt_num": prompt_num,
                "prompt_text": prompt
            }
            
            # Add error to error tracking
            error_data[f"prompt_{prompt_num}"]["errors"][image_file] = error_info
            
            # Write error to TSV
            with open(output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                writer.writerow([image_file, prompt, f"Error: {str(e)}"])
            
            print(f"Error processing image {image_file}: {str(e)}")
        
        # Save progress
        processed_images.add(image_file)
        progress_data = load_progress_file(output_dir)
        progress_data[f"prompt_{prompt_num}"] = list(processed_images)
        save_progress_file(output_dir, progress_data)
        
        # If there were errors, save them
        if error_data[f"prompt_{prompt_num}"]["errors"]:
            save_error_log(output_dir, error_data)
        
        time.sleep(2)  # To avoid rate limiting
    
    return output_file

def main():
    # Set the directories and files
    image_directory = "data/mixed"
    data_info_tsv = "data/image_info.tsv"
    output_dir = create_output_directory()
    
    # Read image files from TSV
    image_files = read_image_files_from_tsv(data_info_tsv)
    print(f"Found {len(image_files)} images in the data info file")
    
    # Process first prompt with default parameters
    process_prompt(
        prompt="Describe the image.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=1
    )

    # Process second prompt with custom parameters
    process_prompt(
        prompt="Write a detailed caption for the image.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=2,
        max_tokens=1024
    )

    # Process third prompt with different parameters
    process_prompt(
        prompt="Tell me everything you can see in the image, including as many visible elements as possible.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=3,
        max_tokens=4096
    )

if __name__ == "__main__":
    main()