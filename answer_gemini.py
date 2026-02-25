import google.generativeai as genai
import os
import time
import csv
from tqdm import tqdm
import base64
from datetime import datetime
import json
from PIL import Image
from io import BytesIO

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Please export it before running, e.g. "
        "export GEMINI_API_KEY='your_key'"
    )
genai.configure(api_key=gemini_api_key)

MODEL = "gemini-2.5-flash"
GEMINI = genai.GenerativeModel(model_name="gemini-2.5-flash")

def extract_text_from_response(response):
    """Extract textual content from a Gemini response object."""
    if response is None:
        return ""

    # Try the standard .text accessor first (works when finish_reason is STOP)
    try:
        text = response.text
        if text:
            return text
    except (ValueError, AttributeError):
        # Falls through to manual extraction if .text fails
        pass

    # Manual extraction from response structure
    texts = []
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        for part in parts:
            # Try different ways to get text
            part_text = None
            if hasattr(part, 'text'):
                part_text = part.text
            elif hasattr(part, '_raw_part') and hasattr(part._raw_part, 'text'):
                part_text = part._raw_part.text
            elif isinstance(part, dict) and 'text' in part:
                part_text = part['text']
            
            if part_text:
                texts.append(part_text)

    return "\n\n".join(texts).strip()

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

def get_completion_with_image(prompt, image_data, model=GEMINI, max_tokens=1024, temperature=1.0):
    """Generate completion using an already-uploaded image file handle.
    
    Args:
        prompt: The text prompt
        image_data: Already-uploaded Gemini file object (from genai.upload_file)
        model: The Gemini model to use
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
    """
    config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    response = model.generate_content(
        [
            image_data,
            prompt
        ],
        generation_config=config,
    )

    # Try to extract text from response
    response_text = extract_text_from_response(response)
    
    # If we got text, return it even if it was truncated
    if response_text:
        return response_text
    
    # Check finish reason to understand why there's no text
    finish_reasons = [
        getattr(candidate, "finish_reason", "UNKNOWN")
        for candidate in (getattr(response, "candidates", None) or [])
    ]
    
    # If MAX_TOKENS is hit but there's no text, it means the model started generating
    # but the output was so long it got completely truncated. Return a placeholder.
    if any("MAX_TOKENS" in str(reason) for reason in finish_reasons):
        return "[Response truncated - exceeded token limit]"
    
    # For other errors, raise with details
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
    metadata = {
        "finish_reasons": [str(reason) for reason in finish_reasons],
        "prompt_feedback": str(block_reason) if block_reason else None
    }
    raise ValueError(f"Model returned no text content. Details: {metadata}")

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

def process_all_prompts_optimized(prompts_config, image_files, image_directory, output_dir):
    """Process all prompts for all images - upload each image once, reuse for all prompts.
    
    Args:
        prompts_config: List of dicts with keys: 'prompt', 'prompt_num', 'max_tokens', 'temperature'
        image_files: List of image filenames
        image_directory: Directory containing images
        output_dir: Output directory for results
    """
    # Initialize output files and load progress
    output_files = {}
    processed_status = {}  # Track which images have been processed for which prompts
    
    for config in prompts_config:
        prompt_num = config['prompt_num']
        output_file = os.path.join(output_dir, f"prompt_{prompt_num}.tsv")
        output_files[prompt_num] = output_file
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                writer.writerow(["Image", "Prompt", "Answer"])
        
        # Load already processed images for this prompt
        processed_status[prompt_num] = load_progress(output_file)
        print(f"Prompt {prompt_num}: {len(processed_status[prompt_num])} already processed")
    
    # Initialize error tracking
    error_data = {f"prompt_{config['prompt_num']}": {"prompt_text": config['prompt'], "errors": {}} 
                  for config in prompts_config}
    
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)} images with {len(prompts_config)} prompts")
    print(f"Upload optimization: Each image uploaded ONCE, reused for all prompts")
    print(f"{'='*60}\n")
    
    # Process each image: upload once, apply all prompts
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_directory, image_file)
        
        # Determine which prompts still need to be processed for this image
        prompts_to_process = [
            config for config in prompts_config 
            if image_file not in processed_status[config['prompt_num']]
        ]
        
        if not prompts_to_process:
            continue  # Skip if all prompts already processed for this image
        
        # Upload image once
        uploaded_image = None
        upload_error = None
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Verify image
            is_valid, format_or_error = verify_image(image_path)
            if not is_valid:
                raise ValueError(f"Invalid image: {format_or_error}")
            
            # Upload image ONCE
            uploaded_image = genai.upload_file(path=image_path)
            
        except Exception as e:
            upload_error = e
            print(f"\n✗ Failed to upload {image_file}: {str(e)}")
        
        # Process all prompts for this uploaded image
        for config in prompts_to_process:
            prompt_num = config['prompt_num']
            prompt = config['prompt']
            max_tokens = config.get('max_tokens', 1024)
            temperature = config.get('temperature', 0.0)
            output_file = output_files[prompt_num]
            
            try:
                if upload_error:
                    raise upload_error
                
                # Use the already-uploaded image
                answer = get_completion_with_image(
                    prompt, 
                    uploaded_image,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
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
            
            # Mark as processed
            processed_status[prompt_num].add(image_file)
        
        # Save progress after each image (all prompts)
        progress_data = {f"prompt_{pnum}": list(pstatus) 
                        for pnum, pstatus in processed_status.items()}
        save_progress_file(output_dir, progress_data)
        
        # Save errors if any
        if any(error_data[key]["errors"] for key in error_data):
            save_error_log(output_dir, error_data)
        
        time.sleep(2)  # To avoid rate limiting
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    for prompt_num in sorted(output_files.keys()):
        total = len(image_files)
        processed = len(processed_status[prompt_num])
        print(f"Prompt {prompt_num}: {processed}/{total} images processed")
    print(f"{'='*60}\n")

def main():
    # Set the directories and files
    image_directory = "data/mixed"
    data_info_tsv = "data/image_info.tsv"
    output_dir = create_output_directory()
    
    # Read image files from TSV
    image_files = read_image_files_from_tsv(data_info_tsv)
    print(f"Found {len(image_files)} images in the data info file")
    
    # Define all prompts upfront
    prompts_config = [
        {
            'prompt': "Describe the image.",
            'prompt_num': 1,
            'max_tokens': 1024,
            'temperature': 0.0
        },
        {
            'prompt': "Write a detailed caption for the image.",
            'prompt_num': 2,
            'max_tokens': 1024,
            'temperature': 0.0
        },
        {
            'prompt': "Tell me everything you can see in the image, including as many visible elements as possible.",
            'prompt_num': 3,
            'max_tokens': 4096,
            'temperature': 0.0
        }
    ]
    
    # Process all prompts with optimized image uploading (upload once per image)
    process_all_prompts_optimized(
        prompts_config=prompts_config,
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()