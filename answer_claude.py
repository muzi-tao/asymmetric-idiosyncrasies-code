from anthropic import Anthropic
import os
import time
import csv
from tqdm import tqdm
import base64
from datetime import datetime
import json
from PIL import Image
from io import BytesIO
import random

# Set up Anthropic API client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Please export it before running, e.g. "
        "export ANTHROPIC_API_KEY='your_key'"
    )
client = Anthropic(api_key=anthropic_api_key)

MODEL = "claude-3-5-sonnet-20241022"

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
            
            # Handle MPO files as JPEG since they're essentially multi-picture JPEG files
            if actual_format == 'mpo':
                return 'image/jpeg'
            
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
            # Convert MPO to JPEG
            if img.format.upper() == 'MPO':
                # Create a new buffer for JPEG conversion
                buffer = BytesIO()
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                # Save as JPEG
                img.save(buffer, format='JPEG', quality=95)
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return base64_image, 'image/jpeg'
            
            # Handle other formats as before
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            buffer = BytesIO()
            img.save(buffer, format=img.format)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            media_type = get_image_media_type(image_path)
            return base64_image, media_type
            
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def get_completion_with_image(prompt, image_path, model=MODEL, max_tokens=1024, temperature=1.0):
    """Get completion with automatic retry for overloaded and rate limit errors"""
    base64_image, media_type = encode_image(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    while True:
        try:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.content[0].text
            
        except Exception as e:
            error_message = str(e)
            
            if 'rate_limit_error' in error_message:
                print(f"\nRate limit reached. Waiting for 3600 seconds before retrying...")
                time.sleep(3600)
                continue
                
            elif 'overloaded_error' in error_message:
                print(f"\nServer overloaded. Waiting for 30 seconds before retrying...")
                time.sleep(30)
                continue
                
            else:
                raise

def read_image_files_from_tsv(tsv_path):
    """Read image filenames from the data info TSV file"""
    image_files = []
    with open(tsv_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            image_files.append(row['Filename'])
    return image_files

def get_errored_images(output_dir, prompt_num):
    """Get list of images that had errors in previous runs"""
    error_file = os.path.join(output_dir, "error_log.json")
    errored_images = set()
    
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_data = json.load(f)
            # Look for errors in this specific prompt
            prompt_key = f"prompt_{prompt_num}"
            if prompt_key in error_data:
                errored_images = set(error_data[prompt_key]["errors"].keys())
    
    return errored_images

def get_successful_images(output_file):
    """Get list of successfully processed images (no errors in output)"""
    successful_images = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)  # Skip header
            for row in reader:
                if row and not row[2].startswith("Error:"):  # Check if the answer doesn't start with "Error:"
                    successful_images.add(row[0])
    
    return successful_images

def find_latest_output_dir(base_dir="answers", model=MODEL):
    """Find the most recent output directory"""
    if not os.path.exists(base_dir):
        return None
        
    dirs = [d for d in os.listdir(base_dir) if d.startswith(model)]
    if not dirs:
        return None
    
    latest_dir = max(dirs, key=lambda x: os.path.getctime(os.path.join(base_dir, x)))
    return os.path.join(base_dir, latest_dir)

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
            # If it's an MPO file, report it as JPEG
            if img.format.upper() == 'MPO':
                return True, 'JPEG'
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

def process_prompt(prompt, image_files, image_directory, output_dir, prompt_num, max_tokens=1024, temperature=1.0):
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
        
        while True:
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Verify image
                is_valid, format_or_error = verify_image(image_path)
                if not is_valid:
                    raise ValueError(f"Invalid image: {format_or_error}")
                
                # Try to process image
                answer = get_completion_with_image(
                    prompt, 
                    image_path, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
                
                # Write successful result
                with open(output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t')
                    writer.writerow([image_file, prompt, answer])
                
                break
                
            except Exception as e:
                error_message = str(e)
                
                if 'rate_limit_error' in error_message or 'overloaded_error' in error_message:
                    print(f"\nAPI error while processing {image_file}. Will retry automatically...")
                    continue
                    
                # For other errors, log them and move to next image
                error_info = {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "image_path": image_path,
                    "prompt_text": prompt
                }
                
                error_data[f"prompt_{prompt_num}"]["errors"][image_file] = error_info
                
                # Write error to TSV
                with open(output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t')
                    writer.writerow([image_file, prompt, f"Error: {str(e)}"])
                
                print(f"\nError processing image {image_file}: {str(e)}")
                break
        
        # Save progress
        processed_images.add(image_file)
        progress_data = load_progress_file(output_dir)
        progress_data[f"prompt_{prompt_num}"] = list(processed_images)
        save_progress_file(output_dir, progress_data)
        
        # If there were errors, save them
        if error_data[f"prompt_{prompt_num}"]["errors"]:
            save_error_log(output_dir, error_data)
        
        # Add random delay between 2-4 seconds to avoid consistent load patterns
        delay = 2 + random.random() * 2
        time.sleep(delay)
    
    return output_file

def main():
    # Set the directories and files
    image_directory = "data/mixed"
    data_info_tsv = "data/image_info.tsv"
    
    # Find the latest output directory from previous runs
    prev_output_dir = find_latest_output_dir()
    
    # Create new output directory for this run
    output_dir = create_output_directory()
    
    try:
        # Read image files from TSV
        image_files = read_image_files_from_tsv(data_info_tsv)
        print(f"Found {len(image_files)} total images in the data info file")
        
        # Process prompts with different parameters
        prompts = [
            {
                "text": "Describe the image.",
                "max_tokens": 1024,
                "temperature": 1.0
            },
            {
                "text": "Write a detailed caption for the image.",
                "max_tokens": 1024,
                "temperature": 1.0
            },
            {
                "text": "Tell me everything you can see in the image, including as many visible elements as possible.",
                "max_tokens": 4096,
                "temperature": 1.0
            }
        ]
        
        # Process each prompt
        for i, prompt_config in enumerate(prompts, 1):
            try:
                prompt_output_file = f"prompt_{i}.tsv"
                
                # Get previously errored images if previous output directory exists
                errored_images = set()
                if prev_output_dir:
                    prev_output_file = os.path.join(prev_output_dir, prompt_output_file)
                    errored_images = get_errored_images(prev_output_dir, i)
                    successful_images = get_successful_images(prev_output_file)
                    
                    print(f"\nPrompt {i}:")
                    print(f"Found {len(successful_images)} successfully processed images")
                    print(f"Found {len(errored_images)} images with errors")
                    
                    # Determine images that need processing (unprocessed + errored)
                    images_to_process = set(image_files) - successful_images | errored_images
                else:
                    images_to_process = set(image_files)
                
                if not images_to_process:
                    print(f"\nSkipping prompt {i} - all images already processed successfully")
                    continue
                
                print(f"\nProcessing {len(images_to_process)} images for prompt {i}")
                
                # Copy successful results from previous run if they exist
                if prev_output_dir:
                    prev_output_file = os.path.join(prev_output_dir, prompt_output_file)
                    new_output_file = os.path.join(output_dir, prompt_output_file)
                    
                    if os.path.exists(prev_output_file):
                        with open(prev_output_file, 'r', newline='', encoding='utf-8') as prev_file:
                            reader = csv.reader(prev_file, delimiter='\t')
                            
                            # Create new file with headers
                            with open(new_output_file, 'w', newline='', encoding='utf-8') as new_file:
                                writer = csv.writer(new_file, delimiter='\t')
                                
                                # Copy header
                                header = next(reader)
                                writer.writerow(header)
                                
                                # Copy successful results
                                for row in reader:
                                    if row and row[0] not in images_to_process:
                                        writer.writerow(row)
                
                # Process remaining images
                process_prompt(
                    prompt=prompt_config["text"],
                    image_files=list(images_to_process),
                    image_directory=image_directory,
                    output_dir=output_dir,
                    prompt_num=i,
                    max_tokens=prompt_config["max_tokens"],
                    temperature=prompt_config["temperature"]
                )
                
            except Exception as e:
                print(f"\nError processing prompt {i}: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nScript finished. Check the output directory for results and error logs.")

if __name__ == "__main__":
    main()