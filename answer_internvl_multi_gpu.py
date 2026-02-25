#!/usr/bin/env python3
"""
Multi-GPU accelerated version for InternVL3.5-8B using multiple processes
Each GPU runs its own model instance processing a subset of images
"""
from vllm import LLM, SamplingParams
import os
import time
import csv
from tqdm import tqdm
from datetime import datetime
import json
from PIL import Image
from io import BytesIO
import base64
import logging
import warnings
import multiprocessing as mp

# Suppress vLLM verbose logging and warnings
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Set environment variables to reduce vLLM verbosity
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Initialize the InternVL3.5-8B model using vLLM
MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
MODEL = "internvl3.5-8b"

def create_output_directory():
    """Create and return path to output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"answers/{MODEL}_multigpu_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def encode_image_to_base64(image_path, max_size=1568):
    """Encode image to base64 for vLLM multimodal inference with size limiting
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) to prevent token overflow
    """
    try:
        with Image.open(image_path) as img:
            # Resize if image is too large to prevent token overflow
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            image_bytes = buffered.getvalue()
            return base64.b64encode(image_bytes).decode("utf-8")
            
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def verify_image(image_path):
    """Verify if the image is valid and return its format"""
    try:
        with Image.open(image_path) as img:
            img.verify()
            return True, img.format
    except Exception as e:
        return False, str(e)

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
            next(reader, None)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    processed_images.add(row[0])  # First column is image filename
    return processed_images

def save_progress_file(output_dir, progress_data):
    """Save progress information to a JSON file"""
    progress_file = os.path.join(output_dir, "progress.json")
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress_file(output_dir):
    """Load progress information from JSON file"""
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_error_log(output_dir, error_data):
    """Save error information to a JSON file"""
    error_file = os.path.join(output_dir, "error_log.json")
    
    # Load existing errors if file exists
    existing_errors = {}
    if os.path.exists(error_file):
        try:
            with open(error_file, 'r') as f:
                existing_errors = json.load(f)
        except:
            pass
    
    # Update with new errors
    existing_errors.update(error_data)
    
    # Save updated errors
    with open(error_file, 'w') as f:
        json.dump(existing_errors, f, indent=2)

def get_completion_with_image_batch(llm, prompt, image_paths, max_tokens=1024, temperature=0.7):
    """Get completions from vLLM model for multiple images in batch"""
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|endoftext|>"]
        )
        
        # Create messages for all images in batch
        messages_list = []
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            messages_list.append(messages)
        
        # Process batch
        outputs = llm.chat(messages_list, sampling_params)
        results = [output.outputs[0].text.strip() for output in outputs]
        return results
        
    except Exception as e:
        raise RuntimeError(f"Batch inference failed: {str(e)}")

def process_gpu_worker(gpu_id, image_subset, prompt, image_directory, output_file, 
                       max_tokens, temperature, batch_size, prompt_num, prompt_text):
    """Worker function that runs on a specific GPU - writes to separate file per GPU"""
    # Set which GPU this process should use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Suppress vLLM output in workers
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # Import after setting CUDA_VISIBLE_DEVICES
    from io import StringIO
    import sys
    
    # Create a GPU-specific output file
    gpu_output_file = output_file.replace('.tsv', f'_gpu{gpu_id}.tsv')
    
    # Create JSON error log for this GPU
    gpu_error_json = output_file.replace('.tsv', f'_gpu{gpu_id}_errors.json')
    
    # Track errors for this GPU
    gpu_errors = {}
    
    try:
        # Initialize model on this GPU
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.88,
            tensor_parallel_size=1,
            max_num_batched_tokens=32768,
            max_num_seqs=256,
            enforce_eager=False,
            enable_chunked_prefill=True
        )
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Create GPU-specific file with header
        with open(gpu_output_file, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(["Image", "Prompt", "Answer"])
        
        # Process images in batches with tqdm progress bar
        desc = f"GPU {gpu_id}"
        pbar = tqdm(total=len(image_subset), desc=desc, position=gpu_id, leave=True)
        
        for batch_idx in range(0, len(image_subset), batch_size):
            batch_images = image_subset[batch_idx:batch_idx+batch_size]
            batch_paths = [os.path.join(image_directory, img) for img in batch_images]
            
            # Validate batch images
            valid_images = []
            valid_paths = []
            for image_file, image_path in zip(batch_images, batch_paths):
                try:
                    if not os.path.exists(image_path):
                        gpu_errors[image_file] = {
                            "timestamp": datetime.now().isoformat(),
                            "error_type": "FileNotFoundError",
                            "error_message": f"Image file not found: {image_path}",
                            "image_path": image_path,
                            "prompt_num": prompt_num,
                            "prompt_text": prompt_text,
                            "gpu_id": gpu_id
                        }
                        continue
                    is_valid, format_or_error = verify_image(image_path)
                    if is_valid:
                        valid_images.append(image_file)
                        valid_paths.append(image_path)
                    else:
                        gpu_errors[image_file] = {
                            "timestamp": datetime.now().isoformat(),
                            "error_type": "InvalidImageError",
                            "error_message": f"Invalid image: {format_or_error}",
                            "image_path": image_path,
                            "prompt_num": prompt_num,
                            "prompt_text": prompt_text,
                            "gpu_id": gpu_id
                        }
                except Exception as e:
                    gpu_errors[image_file] = {
                        "timestamp": datetime.now().isoformat(),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "image_path": image_path,
                        "prompt_num": prompt_num,
                        "prompt_text": prompt_text,
                        "gpu_id": gpu_id
                    }
                    continue
            
            if not valid_paths:
                pbar.update(len(batch_images))
                continue
            
            try:
                # Suppress vLLM output during inference
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                # Batch inference
                answers = get_completion_with_image_batch(llm, prompt, valid_paths, 
                                                         max_tokens=max_tokens, 
                                                         temperature=temperature)
                
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Write results immediately to GPU-specific file
                with open(gpu_output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t')
                    for image_file, answer in zip(valid_images, answers):
                        writer.writerow([image_file, prompt, answer])
                
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                error_msg = str(e)
                
                # Log errors for each image in the batch
                for image_file, image_path in zip(valid_images, valid_paths):
                    gpu_errors[image_file] = {
                        "timestamp": datetime.now().isoformat(),
                        "error_type": type(e).__name__,
                        "error_message": error_msg,
                        "image_path": image_path,
                        "prompt_num": prompt_num,
                        "prompt_text": prompt_text,
                        "gpu_id": gpu_id
                    }
                
                # Write errors immediately
                with open(gpu_output_file, 'a', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t')
                    for image_file in valid_images:
                        writer.writerow([image_file, prompt, f"Error: {error_msg}"])
            
            # Update progress bar
            pbar.update(len(batch_images))
        
        pbar.close()
        
        # Save errors to JSON file
        if gpu_errors:
            with open(gpu_error_json, 'w') as f:
                json.dump(gpu_errors, f, indent=2)
        
        # Count results
        with open(gpu_output_file, 'r') as f:
            result_count = sum(1 for _ in f) - 1  # Subtract header
        
        return result_count
        
    except Exception as e:
        print(f"\nGPU {gpu_id}: ❌ Failed - {str(e)}")
        
        # Save any collected errors before exiting
        if gpu_errors:
            try:
                with open(gpu_error_json, 'w') as f:
                    json.dump(gpu_errors, f, indent=2)
            except:
                pass
        
        return 0

def combine_gpu_results(output_file, num_gpus, output_dir, prompt_num, prompt_text):
    """Combine results from all GPU-specific files into final output file and consolidate errors"""
    # Create final file with header
    with open(output_file, 'w', newline='', encoding='utf-8') as final_file:
        writer = csv.writer(final_file, delimiter='\t')
        writer.writerow(["Image", "Prompt", "Answer"])
        
        # Append results from each GPU file
        for gpu_id in range(num_gpus):
            gpu_file = output_file.replace('.tsv', f'_gpu{gpu_id}.tsv')
            if os.path.exists(gpu_file):
                with open(gpu_file, 'r', encoding='utf-8') as gf:
                    next(gf)  # Skip header
                    for line in gf:
                        final_file.write(line)
                
                # Delete GPU-specific file after combining
                os.remove(gpu_file)
    
    # Combine GPU error JSON files
    all_errors = {}
    for gpu_id in range(num_gpus):
        gpu_error_file = output_file.replace('.tsv', f'_gpu{gpu_id}_errors.json')
        if os.path.exists(gpu_error_file):
            try:
                with open(gpu_error_file, 'r') as f:
                    gpu_errors = json.load(f)
                    all_errors.update(gpu_errors)
                # Delete GPU-specific error file after combining
                os.remove(gpu_error_file)
            except:
                pass
    
    # Save combined errors if any
    if all_errors:
        error_data = {
            f"prompt_{prompt_num}": {
                "prompt_text": prompt_text,
                "errors": all_errors
            }
        }
        save_error_log(output_dir, error_data)
        return len(all_errors)
    
    return 0

def process_prompt_multi_gpu(prompt, image_files, image_directory, output_dir, 
                             prompt_num, max_tokens=1024, temperature=0.0, 
                             batch_size=16, num_gpus=8):
    """Process a single prompt across multiple GPUs with individual progress bars"""
    output_file = os.path.join(output_dir, f"prompt_{prompt_num}.tsv")
    
    print(f"\n{'='*70}")
    print(f"📝 PROMPT {prompt_num}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"{'='*70}")
    print(f"🔧 Configuration:")
    print(f"   - GPUs: {num_gpus}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Total images: {len(image_files):,}")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Temperature: {temperature}")
    print()
    
    # Split images across GPUs
    chunk_size = len(image_files) // num_gpus
    image_subsets = []
    print("📊 Image distribution:")
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(image_files)
        image_subsets.append(image_files[start_idx:end_idx])
        batches = (len(image_subsets[i]) + batch_size - 1) // batch_size
        print(f"   GPU {i}: {len(image_subsets[i]):>5} images ({batches:>3} batches)")
    print()
    
    print("🚀 Starting parallel processing...")
    print(f"   Each GPU writes to separate file, combined at end")
    print()
    
    # Create processes for each GPU
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_gpu_worker,
            args=(gpu_id, image_subsets[gpu_id], prompt, image_directory, 
                  output_file, max_tokens, temperature, batch_size, prompt_num, prompt)
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    
    print()
    print("📦 Combining results from all GPUs...")
    
    # Combine all GPU results into final file and consolidate errors
    error_count = combine_gpu_results(output_file, num_gpus, output_dir, prompt_num, prompt)
    
    # Count results
    with open(output_file, 'r') as f:
        result_count = sum(1 for _ in f) - 1  # Subtract header
    
    # Save progress
    processed_images = load_progress(output_file)
    progress_data = load_progress_file(output_dir)
    progress_data[f"prompt_{prompt_num}"] = list(processed_images)
    save_progress_file(output_dir, progress_data)
    
    print()
    print(f"{'='*70}")
    print(f"✅ Prompt {prompt_num} completed!")
    print(f"   - Processed: {result_count:,} images")
    if error_count > 0:
        print(f"   - Errors: {error_count:,} images (see error_log.json)")
    print(f"   - Time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    if result_count > 0:
        print(f"   - Speed: {result_count/elapsed:.2f} images/sec")
    print(f"   - Output: {output_file}")
    print(f"{'='*70}")
    
    return output_file

def main():
    print("🚀 Starting Multi-GPU InternVL3.5-8B Image Captioning")
    print("=" * 70)
    
    # Detect available GPUs
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"✅ Detected {num_gpus} GPU(s)")
    except:
        num_gpus = 1
        print("⚠️  Could not detect GPUs, defaulting to 1")
    
    # Set the directories and files
    image_directory = "data/mixed"
    data_info_tsv = "data/image_info.tsv"
    output_dir = create_output_directory()
    
    print(f"📂 Output directory: {output_dir}")
    
    # Read image files from TSV
    image_files = read_image_files_from_tsv(data_info_tsv)
    print(f"📸 Total images: {len(image_files):,}")
    print(f"⚡ Expected speedup: ~{num_gpus}x with multi-GPU + batching")
    print(f"🔧 Model: InternVL3.5-8B (8B parameters, high quality)")
    print(f"📏 Image processing: max_model_len=8192, auto-resize to 1568px")
    print()
    print("=" * 70)
    
    total_start = time.time()
    
    # Process first prompt
    process_prompt_multi_gpu(
        prompt="Describe the image.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=1,
        batch_size=16,
        num_gpus=num_gpus
    )

    # Process second prompt
    process_prompt_multi_gpu(
        prompt="Write a detailed caption for the image.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=2,
        max_tokens=1024,
        temperature=0.3,
        batch_size=16,
        num_gpus=num_gpus
    )

    # Process third prompt
    process_prompt_multi_gpu(
        prompt="Tell me everything you can see in the image, including as many visible elements as possible.",
        image_files=image_files,
        image_directory=image_directory,
        output_dir=output_dir,
        prompt_num=3,
        max_tokens=2048,
        temperature=0.0,
        batch_size=8,
        num_gpus=num_gpus
    )

    total_elapsed = time.time() - total_start
    
    print()
    print("=" * 70)
    print("🎉 ALL PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📂 Results saved in: {output_dir}")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"📊 Total images × prompts: {len(image_files) * 3:,}")
    if total_elapsed > 0:
        print(f"⚡ Overall speed: {(len(image_files) * 3)/total_elapsed:.2f} images/sec")
    
    # Check for error log and display summary
    error_log_file = os.path.join(output_dir, "error_log.json")
    if os.path.exists(error_log_file):
        try:
            with open(error_log_file, 'r') as f:
                error_log = json.load(f)
            
            total_errors = sum(len(prompt_data.get("errors", {})) for prompt_data in error_log.values())
            
            print()
            print(f"⚠️  Errors encountered: {total_errors} images")
            print(f"   Error details: {error_log_file}")
            print()
            print("   Error breakdown by prompt:")
            for prompt_key, prompt_data in error_log.items():
                error_count = len(prompt_data.get("errors", {}))
                print(f"   - {prompt_key}: {error_count} errors")
        except:
            pass
    else:
        print(f"🏆 InternVL3.5-8B multi-GPU processing complete with no errors!")
    
    # Show progress file location
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        print(f"\n📊 Progress tracking: {progress_file}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()

