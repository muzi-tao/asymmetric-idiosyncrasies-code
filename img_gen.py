#!/usr/bin/env python3
import os
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import logging
import re
import sentencepiece

from diffusers.utils import logging as dlogging
dlogging.set_verbosity_error()
from transformers.utils import logging as tlogging
tlogging.set_verbosity_error()
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.backends.cuda.matmul.allow_tf32 = True

from diffusers import StableDiffusion3Pipeline
from diffusers import AutoPipelineForText2Image
from diffusers import FluxPipeline
from diffusers import DiffusionPipeline

def setup_distributed():
    """Initialize distributed training setup."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0

def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

def gather_object(obj):
    """Gather objects from all processes to rank 0."""
    if not dist.is_initialized():
        return [obj]
    
    world_size = dist.get_world_size()
    gathered_objs = [None] * world_size
    dist.all_gather_object(gathered_objs, obj)
    return gathered_objs

def distribute_data(data_list: List, rank: int, world_size: int) -> List:
    """Distribute data across ranks for DDP processing."""
    total_items = len(data_list)
    items_per_rank = total_items // world_size
    remainder = total_items % world_size
    
    start_idx = rank * items_per_rank + min(rank, remainder)
    end_idx = start_idx + items_per_rank + (1 if rank < remainder else 0)
    
    rank_data = data_list[start_idx:end_idx]
    
    if is_main_process():
        print(f"Distributing {total_items} items across {world_size} ranks:")
        for r in range(world_size):
            r_start = r * items_per_rank + min(r, remainder)
            r_end = r_start + items_per_rank + (1 if r < remainder else 0)
            print(f"  Rank {r}: items {r_start}-{r_end-1} ({r_end - r_start} items)")
    
    return rank_data

def check_gpu():
    """Checks if a GPU is available and prints GPU details."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available! Make sure you're running on a GPU node.")
    num_gpus = torch.cuda.device_count() 
    print(f"Found {num_gpus} GPU(s).") 
    for i in range(num_gpus): 
        device_name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {device_name} ({mem_gb:.2f} GB)") 

def load_prompts(json_path: str) -> List[Dict]:
    """Load prompts from a single JSON file."""
    with open(json_path, 'r') as f:
        prompts = json.load(f)
    return prompts

def save_checkpoint(checkpoint_path: str, model_name: str, completed_indices: set):
    """Save progress checkpoint to a JSON file (only on rank 0)."""
    if not is_main_process():
        return
    
    checkpoint_data = {
        'model_name': model_name,
        'completed_indices': sorted(completed_indices)
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved to: {checkpoint_path}")

def load_checkpoint(checkpoint_path: str) -> tuple[str, set]:
    """Load progress checkpoint from a JSON file."""
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data['model_name'], set(checkpoint_data['completed_indices'])
    except FileNotFoundError:
        return None, set()

def scan_existing_outputs(model_name: str, sd_model_shortcut: str) -> set: 
    """Scan existing output directories for generated images and return their prompt IDs (only on rank 0)."""
    existing_indices = set()
    
    if is_main_process():
        dir_pattern = f"gen_{sd_model_shortcut}_{model_name}_*"
        prefix = f"{sd_model_shortcut}_"
        prefix_len = len(prefix)

        for dir_path in glob.glob(dir_pattern):
            print(f"Scanning directory: {dir_path}")
            if os.path.isdir(dir_path):
                for file_path in glob.glob(os.path.join(dir_path, "*.png")):
                    filename = os.path.basename(file_path)
                    if filename.startswith(prefix) and filename.endswith(".png"):
                        prompt_id = filename[prefix_len:-4]
                        existing_indices.add(prompt_id)
                print(f"Found {len(existing_indices)} existing images matching prefix '{prefix}'")
    
    if dist.is_initialized():
        existing_list = sorted(existing_indices) if is_main_process() else []
        existing_list = [existing_list]
        dist.broadcast_object_list(existing_list, src=0)
        existing_indices = set(existing_list[0])
    
    return existing_indices

def gather_completed_indices(local_completed: set) -> set:
    """Gather completed indices from all ranks and return the union."""
    if not dist.is_initialized():
        return local_completed
    
    all_completed = gather_object(local_completed)
    
    global_completed = set()
    for completed_set in all_completed:
        global_completed.update(completed_set)
    
    return global_completed

def generate_image_batch(
    prompts: List[str],
    pipe,
    img_width: int,
    img_height: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    is_flux: bool = False,
    is_qwen: bool = False,
    sd_model_shortcut: str = None,
) -> tuple[List, float]:
    """Generate a batch of images using Stable Diffusion, Flux, or Qwen models and return images + time taken."""

    start_time = time.time()
    with torch.inference_mode():
        if is_flux:
            results = pipe(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=img_width,
                height=img_height,
            )
        elif is_qwen:
            results = pipe(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=img_width,
                height=img_height,
            )
        else:
            negative_prompt = (
                "blurry, bad quality, distorted, disfigured, deformed, "
                "low resolution, oversaturated, undersaturated"
            )
            results = pipe(
                prompt=prompts,
                negative_prompt=[negative_prompt] * len(prompts),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=img_width,
                height=img_height,
            )
        images = results.images
    end_time = time.time()
    generation_time = end_time - start_time

    return images, generation_time

def process_file(
    json_path: str,
    model_name: str,
    sd_model_shortcut: str,
    img_width: int,
    img_height: int,
    batch_size: int,
    timestamp: str = None,
    checkpoint_dir: str = "checkpoints",
    pipe=None,
    start_index: str = None,
    end_index: str = None,
    is_flux: bool = False,
    is_qwen: bool = False,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    resume: bool = False,
    no_wandb: bool = False,
    rank: int = 0,
    world_size: int = 1,
    outdir: str = None,
):
    """
    Process a single JSON file of prompts and generate images with checkpoint support and DDP.

    Args:
        json_path: Path to the JSON file containing prompts
        model_name: Name of the model (used in output filename)
        timestamp: Optional timestamp to use (if resuming from checkpoint)
        checkpoint_dir: Directory to store checkpoints
        pipe: Optional pipeline instance to reuse
        start_index: Index of the prompt to start from (e.g., 'claude-3-5-sonnet_prompt_1_1451')
        end_index: Index of the prompt to end at (inclusive). If not provided, processes to the end.
        rank: Current process rank for DDP
        world_size: Total number of processes for DDP
    """
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
    barrier()
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{sd_model_shortcut}_{model_name}_checkpoint.json")
    completed_indices = set()

    if resume:
        if is_main_process():
            saved_model, completed_indices = load_checkpoint(checkpoint_path)

            if saved_model and saved_model != model_name:
                raise ValueError(f"Checkpoint exists for different model: {saved_model}")

            existing_indices = scan_existing_outputs(model_name, sd_model_shortcut)
            original_count = len(completed_indices)
            completed_indices.update(existing_indices)
            new_count = len(completed_indices)

            if new_count > original_count:
                print(f"Found {new_count - original_count} existing images. Updating checkpoint.")
                save_checkpoint(checkpoint_path, model_name, completed_indices)
        
        if dist.is_initialized():
            completed_list = sorted(completed_indices) if is_main_process() else []
            completed_list = [completed_list]
            dist.broadcast_object_list(completed_list, src=0)
            completed_indices = set(completed_list[0])
        
        barrier()

    prompts_list = []
    if is_main_process():
        prompts_list = load_prompts(json_path)
        print(f"Loaded {len(prompts_list)} prompts from {json_path}")
        print(f"Already completed: {len(completed_indices)} images")
    
    if dist.is_initialized():
        prompts_list = [prompts_list]
        dist.broadcast_object_list(prompts_list, src=0)
        prompts_list = prompts_list[0]
    elif not prompts_list:
        prompts_list = load_prompts(json_path)
        if rank == 0:
            print(f"Loaded {len(prompts_list)} prompts from {json_path}")
            print(f"Already completed: {len(completed_indices)} images")

    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = outdir if outdir else f"gen_{sd_model_shortcut}_{model_name}_{timestamp}"

    start_pos = 0
    if start_index:
        for i, prompt_data in enumerate(prompts_list):
            if prompt_data['index'] == start_index:
                start_pos = i
                if is_main_process():
                    print(f"Starting from index: {start_index} (position {start_pos + 1})")
                break
        else:
            raise ValueError(f"Start index '{start_index}' not found in prompts.")

    end_pos = len(prompts_list)
    if end_index:
        for j, prompt_data in enumerate(prompts_list):
            if prompt_data['index'] == end_index:
                end_pos = j + 1
                if is_main_process():
                    print(f"Ending at index: {end_index} (position {j + 1})")
                break
        else:
            raise ValueError(f"End index '{end_index}' not found in prompts.")
        
    filtered_prompts = []
    if is_main_process():
        print("Filtering prompts based on start/end indices and completed list...")
    
    for prompt_data in prompts_list[start_pos:end_pos]:
         prompt_id = prompt_data['index']

         if prompt_id in completed_indices:
             continue

         filtered_prompts.append(prompt_data)

    if is_main_process():
        print(f"Total prompts to process: {len(filtered_prompts)}")
    
    prompts_to_process = distribute_data(filtered_prompts, rank, world_size)
    
    print(f"Rank {rank}: Processing {len(prompts_to_process)} prompts")

    has_work = len(prompts_to_process) > 0
    if dist.is_initialized():
        work_status = gather_object(has_work)
        any_rank_has_work = any(work_status)
    else:
        any_rank_has_work = has_work
    
    if not any_rank_has_work:
        if is_main_process():
            print("No prompts to process after filtering across all ranks. Exiting.")
            all_prompts_indices = {p['index'] for p in prompts_list}
            if all_prompts_indices.issubset(completed_indices):
                checkpoint_path = os.path.join(checkpoint_dir, f"{sd_model_shortcut}_{model_name}_checkpoint.json")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    print("All prompts in the original file are completed. Checkpoint cleared.")
            else:
                print(f"Not all prompts completed in the original file. {len(all_prompts_indices - completed_indices)} indices remaining. Checkpoint kept.")
        
        barrier()
        return timestamp
        
    current_batch_prompts_text: List[str] = []
    current_batch_indices: List[str] = []
    current_batch_data: List[Dict] = []
    total_processed_in_run = 0

    local_completed_indices = set()
    
    try:
        if is_main_process():
            print(f"Starting image generation in batches of {batch_size}...")
        
        progress_desc = f"Rank {rank} Processing Prompts"
        for i, prompt_data in enumerate(tqdm(prompts_to_process, desc=progress_desc, disable=not is_main_process())):
            prompt_id = prompt_data['index']
            
            raw_prompt = prompt_data['answer']
            
            cleaned_prompt = re.sub(r'\s+', ' ', raw_prompt).strip()
            
            cleaned_prompt = cleaned_prompt.replace('<|endoftext|>', '')
            
            processed_prompt = cleaned_prompt

            current_batch_prompts_text.append(processed_prompt)
            current_batch_indices.append(prompt_id)
            current_batch_data.append(prompt_data)

            is_last_prompt = (i == len(prompts_to_process) - 1)
            if len(current_batch_prompts_text) == batch_size or (is_last_prompt and current_batch_prompts_text):

                batch_start_time = time.time()
                print(f"\nRank {rank} --- Processing Batch --- (Prompt IDs: {', '.join(current_batch_indices)})")
                
                try:
                    generated_images, batch_generation_time = generate_image_batch(
                        prompts=current_batch_prompts_text,
                        pipe=pipe,
                        img_width=img_width,
                        img_height=img_height,
                        is_flux=is_flux,
                        is_qwen=is_qwen,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        sd_model_shortcut=sd_model_shortcut,
                    )

                    print(f"Rank {rank}: Batch generated in {batch_generation_time:.2f} seconds. Saving {len(generated_images)} images...")

                    for img, p_id, p_data in zip(generated_images, current_batch_indices, current_batch_data):
                        output_path = os.path.join(output_root, f"{sd_model_shortcut}_{p_id}.png")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        try:
                            img.save(output_path)

                            if not no_wandb and is_main_process():
                                try:
                                    wandb.log({
                                        "prompt_id": p_id,
                                        "prompt_text": p_data['answer'],
                                        "output_path": output_path,
                                        "image_dimensions": f"{img_width}x{img_height}",
                                        "batch_size_processed": len(current_batch_prompts_text),
                                        "batch_generation_time_sec": batch_generation_time,
                                        "time_per_image_in_batch_sec": batch_generation_time / len(current_batch_prompts_text) if current_batch_prompts_text else 0,
                                        "rank": rank,
                                    })
                                except Exception as e:
                                     print(f"Warning: Could not log metrics {p_id} to Wandb: {e}")

                            local_completed_indices.add(p_id)
                            total_processed_in_run += 1

                        except Exception as save_e:
                            print(f"ERROR: Rank {rank} failed to save image for prompt ID {p_id}: {save_e}")

                    global_completed = gather_completed_indices(local_completed_indices)
                    completed_indices.update(global_completed)
                    save_checkpoint(checkpoint_path, model_name, completed_indices)
                    
                    if is_main_process():
                        print(f"Batch complete. This run: {total_processed_in_run}, Total across all ranks: {len(completed_indices)}")

                    current_batch_prompts_text = []
                    current_batch_indices = []
                    current_batch_data = []

                except Exception as batch_e:
                    print(f"\nERROR: Rank {rank} failed to generate batch {current_batch_indices[0] if current_batch_indices else 'unknown'}: {batch_e}")
                    current_batch_prompts_text = []
                    current_batch_indices = []
                    current_batch_data = []


    except KeyboardInterrupt:
        print(f"\nRank {rank}: Process interrupted by user! Gathering progress...")
        global_completed = gather_completed_indices(local_completed_indices)
        completed_indices.update(global_completed)
        save_checkpoint(checkpoint_path, model_name, completed_indices)
        barrier()
        raise

    except Exception as overall_e:
        print(f"\nRank {rank}: An unexpected error occurred during processing: {overall_e}")
        global_completed = gather_completed_indices(local_completed_indices)
        completed_indices.update(global_completed)
        save_checkpoint(checkpoint_path, model_name, completed_indices)
        barrier()
        raise

    global_completed = gather_completed_indices(local_completed_indices)
    completed_indices.update(global_completed)
    
    if is_main_process():
        print("\nFinished processing prompts.")
        all_prompts_indices = {p['index'] for p in prompts_list}
        if all_prompts_indices.issubset(completed_indices):
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print("\nAll images generated successfully! Checkpoint cleared.")
            else:
                print("\nAll images completed, checkpoint file already removed or never created.")
        else:
            print(f"\nNot all prompts completed. {len(all_prompts_indices - completed_indices)} indices remaining. Checkpoint kept.")
    
    barrier()
    return timestamp

def main():
    rank, world_size, local_rank = setup_distributed()
    
    MODEL_MAP = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd3m": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd35l": "stabilityai/stable-diffusion-3.5-large",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "qwen-image": "Qwen/Qwen-Image",
    }
    
    FLUX_MODELS = {"flux-schnell", "flux-dev"}
    QWEN_MODELS = {"qwen-image"}
    model_choices_help = ', '.join([f'{k} = "{v}"' for k, v in MODEL_MAP.items()])

    parser = argparse.ArgumentParser(description='Generate images from prompt files with checkpoint support')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the JSON file containing prompts')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the prompt source model (e.g., claude, gemini, gpt4) - used for checkpointing and output directories.')
    parser.add_argument('--sd-model-id', type=str, default="sd35l",
                        choices=MODEL_MAP.keys(),
                        help=f'Shortcut for the text-to-image model ID. Choices: {list(MODEL_MAP.keys())}. '
                            f'Defaults to "sd35l". Includes Stable Diffusion, Flux, and Qwen models. '
                            f'Mapping: [{model_choices_help}]')
    parser.add_argument('--image-width', type=int, default=256)
    parser.add_argument('--image-height', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of images to generate in a single batch. Adjust based on GPU memory.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if it exists')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to store checkpoints')
    parser.add_argument('--start-index', type=str,
                        help='Index of the prompt to start from (e.g., claude-3-5-sonnet_prompt_1_1451)')
    parser.add_argument('--end-index', type=str,
                        help='Index of the prompt to end at (inclusive). If not provided, processes to the end.')
    parser.add_argument('--scan-only', action='store_true',
                        help='Only scan existing directories and update checkpoint, no generation')
    parser.add_argument('--num-inference-steps', type=int, default=30,
                        help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable logging to Weights & Biases')
    parser.add_argument('--outdir', type=str,
                        help='Custom output directory for generated images. If not provided, uses default format: gen_{sd_model_id}_{model_name}_{timestamp}')

    args = parser.parse_args()

    selected_sd_model_id = MODEL_MAP[args.sd_model_id]
    is_flux_model = args.sd_model_id in FLUX_MODELS
    is_qwen_model = args.sd_model_id in QWEN_MODELS
    
    if is_flux_model:
        if args.num_inference_steps == 30:
            args.num_inference_steps = 4 if args.sd_model_id == "flux-schnell" else 30
        if args.guidance_scale == 7.5:
            args.guidance_scale = 0.0

    if args.scan_only: 
        print("Running in scan-only mode...")
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.sd_model_id}_{args.model_name}_checkpoint.json")

        saved_model, completed_indices = load_checkpoint(checkpoint_path)

        if saved_model and saved_model != args.model_name:
            raise ValueError(f"Checkpoint exists for different model: {saved_model}")

        prompts = load_prompts(args.file)
        valid_indices = {p['index'] for p in prompts}

        existing_indices = scan_existing_outputs(args.model_name, args.sd_model_id)

        valid_existing = existing_indices & valid_indices
        new_indices = valid_existing - completed_indices

        print(f"Existing: {len(existing_indices)}, Valid: {len(valid_existing)}, New: {len(new_indices)}")

        if new_indices:
            completed_indices.update(new_indices)
            print("Number of completed indices:", len(completed_indices))
            save_checkpoint(checkpoint_path, args.model_name, completed_indices)
            print(f"Added {len(new_indices)} new indices to checkpoint.")
        else:
            print("No new valid indices found in existing directories.")

        print(f"Checkpoint now contains {len(completed_indices)} completed indices.")      
        return

    if rank == 0:
        check_gpu()

    if is_flux_model:
        model_type = "Flux"
    elif is_qwen_model:
        model_type = "Qwen"
    else:
        model_type = "Stable Diffusion"
    if is_main_process():
        print(f"Loading {model_type} model: {args.sd_model_id} on {world_size} GPU(s)")

    try:
        if is_flux_model:
            if world_size > 1:
                pipe = FluxPipeline.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                )
                pipe = pipe.to(f"cuda:{local_rank}")
            else:
                pipe = FluxPipeline.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="balanced"
                )
        elif is_qwen_model:
            if world_size > 1:
                pipe = DiffusionPipeline.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                )
                pipe = pipe.to(f"cuda:{local_rank}")
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
        else:
            if world_size > 1:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                    variant="fp16" if args.sd_model_id in ["sd15", "sd21"] else None,
                )
                pipe = pipe.to(f"cuda:{local_rank}")
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    selected_sd_model_id,
                    torch_dtype=torch.bfloat16,
                    variant="fp16" if args.sd_model_id in ["sd15", "sd21"] else None,
                    device_map="auto"
                )
        
        print(f"Rank {rank}: Pipeline loaded successfully on cuda:{local_rank}")
        
    except Exception as e:
        print(f"Rank {rank}: Error loading pipeline: {e}")
        raise RuntimeError(f"Failed to load pipeline on rank {rank}") from e    

    timestamp = None
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.sd_model_id}_{args.model_name}_checkpoint.json")
        if os.path.exists(checkpoint_path):
            output_dirs = [d for d in os.listdir() if d.startswith(f"gen_{args.sd_model_id}_{args.model_name}_")]
            if output_dirs:
                try:
                    output_dirs.sort(key=lambda d: d.split('_')[-1] + d.split('_')[-2], reverse=True)
                    timestamp = output_dirs[0].split('_')[-2] + "_" + output_dirs[0].split('_')[-1]
                    datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    print(f"Resuming with timestamp: {timestamp}")
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse timestamp from directory name '{output_dirs[0]}'. Starting new timestamp.")
                    timestamp = None
    
    run_name = f"{args.model_name}_{args.sd_model_id}_{timestamp}"
    
    if not args.no_wandb and is_main_process():
        distributed_config = vars(args).copy()
        distributed_config.update({
            'world_size': world_size,
            'rank': rank,
            'local_rank': local_rank,
        })
        wandb.init(project="cap-to-img", config=distributed_config, name=run_name)
        wandb.run.log_code(".")
        print(f"Wandb initialized with run name: {run_name}")

    try:
        process_file(
            args.file,
            args.model_name,
            args.sd_model_id,
            args.image_width,
            args.image_height,
            args.batch_size,
            timestamp,
            args.checkpoint_dir,
            pipe,
            args.start_index,
            args.end_index,
            is_flux_model,
            is_qwen_model,
            args.num_inference_steps,
            args.guidance_scale,
            args.resume,
            args.no_wandb,
            rank,
            world_size,
            args.outdir,
        )
    finally:
        cleanup_distributed()
        if not args.no_wandb and is_main_process():
            wandb.finish()

if __name__ == "__main__":
    main()