# -------------------------------------------------
# Install requirements
# -------------------------------------------------
# pip install -qU transformers accelerate bitsandbytes

# -------------------------------------------------
# Import modules
# -------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
# For FSDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

import wandb
import numpy as np
from datetime import datetime
import json
from tqdm.auto import tqdm
import gc
import math
import time
import os
import sys

# -------------------------------------------------
# Constants
# -------------------------------------------------
MAX_LENGTH = 128

# -------------------------------------------------
# Finetune config
# -------------------------------------------------
class Config:
    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-1.7B"
    dataset_name = "vietgpt/wikipedia_vi"
    
    # Training configuration
    output_dir = "./qwen-vietnamese-wiki-finetuned"
    # output_dir = "./qwen-vietnamese-wiki-finetuned-2"
    num_train_epochs = 5
    per_device_train_batch_size = 2
    per_device_valid_batch_size = 2
    gradient_accumulation_steps = 8
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_length = MAX_LENGTH

    # Optimization settings
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # Logging and saving
    logging_steps = 40
    save_strategy = "epoch"
    valid_strategy = "epoch"
    
    # Other settings
    fp16 = True
    num_workers = os.cpu_count()
    
    # W&B configuration
    use_wandb = True
    wandb_run_id = None
    wandb_project = "PARADIS-Qwen3_0.6B"
    # wandb_project = "PARADIS-Qwen3_1.7B"
    wandb_run_name = "FSDP"

    # HuggingFace configuration
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-FSDP"
    # hf_repo = "h9art/PARADIS-Qwen3_1.7B-10kWikiVi-FSDP"
    
    # Dataset
    train_size = 10000
    valid_size = 10000
    test_size = 5000
    min_text_length = 50
    random_seed = 42


# -------------------------------------------------
# Custom dataset
# -------------------------------------------------
class WikiViDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get data
        item = self.dataset[idx]
        combined_text = f"Tiêu đề: {item['title']}\n\nNội dung: {item['text']}"

        # Tokenize data
        tokenized_text = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Prepare data from tokenizer output
        input_ids = tokenized_text["input_ids"].squeeze()
        attention_mask = tokenized_text["attention_mask"].squeeze()
        labels = input_ids.clone() # In causal LM, labels is the same with input_ids
        labels[attention_mask == 0] = -100 # Do not calculate loss on padding tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# -------------------------------------------------
# Preprocessing functions
# -------------------------------------------------
def filter_function(example, config):
    """Filter out empty or very short texts"""
    
    return (
        example['text'] is not None and 
        example['title'] is not None and
        len(example['text'].strip()) > config.min_text_length
    )

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_epoch(rank, model, dataloader, optimizer, scheduler, scaler, device, epoch, config):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        if config.fp16:
            # For mixed precision
            with torch.autocast(device_type=device.type):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Chia loss cho gradient_accumulation_steps
                # Nếu không nhận được loss sẽ gấp <gradient_accumulation_steps> lần loss thực sự
                loss = outputs.loss / config.gradient_accumulation_steps
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / config.gradient_accumulation_steps
        
        # Backward pass
        if config.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item()
        
        # Update weights every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Logging
        if (step + 1) % config.logging_steps == 0 and rank == 0:
            
            avg_loss = total_loss / (step + 1) * config.gradient_accumulation_steps
            print(f"Step {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

            if config.use_wandb:
                wandb.log({
                    "train_loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "train_step": epoch * len(dataloader) + step + 1
                })
    
    return total_loss / len(dataloader) * config.gradient_accumulation_steps

# -------------------------------------------------
# Validation function
# -------------------------------------------------
def validate(model, dataloader, device, config):
    """Validate the model."""
    
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if config.fp16:
                with torch.autocast(device_type=device.type):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            
            progress_bar.set_postfix({'valid_loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# -------------------------------------------------
# Generation function
# -------------------------------------------------
def generate_text(
    model,
    tokenizer,
    device,
    prompt,
    max_length=MAX_LENGTH,
    temperature=0.7,
    top_p=0.9,
    top_k=50
):
    """Generate text using the model."""
    
    model.eval()
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Generate
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# -------------------------------------------------
# FSDP setup functions
# -------------------------------------------------
def setup_fsdp(rank, world_size):
    """Sets up the process group for distributed training."""
    
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the distributed process group using NCCL backend (optimized for GPUs)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the current device based on the rank
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    """Cleans up the distributed process group."""
    
    if dist.is_initialized():
        dist.destroy_process_group()

def fsdp_training(rank, world_size):
    """Train model with FSDP"""
    # -------------------------------------------------
    # Setup FSDP
    # -------------------------------------------------
    # Set up distributed environment for current rank
    setup_fsdp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(rank)
    print(dist.get_rank())
    sys.exit(0)

    # -------------------------------------------------
    # Environment variables
    # -------------------------------------------------
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # -------------------------------------------------
    # Get secrets
    # -------------------------------------------------
    if rank == 0:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
        WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")

    # -------------------------------------------------
    # Random seed
    # -------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------
    # Init finetune config
    # -------------------------------------------------
    config = Config()
    config_dict = {
        k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)
    }

    # -------------------------------------------------
    # Setup wandb
    # -------------------------------------------------
    if rank == 0:
        wandb.login(key=WANDB_API_KEY)
        if config.use_wandb:
            if config.wandb_run_id is None:
                wandb.init( # New run
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config_dict,
                )
            else:
                wandb.init( # Resume to created run
                    project=config.wandb_project,
                    id=config.wandb_run_id,
                    resume='allow',
                )

    # -------------------------------------------------
    # Setup HuggingFace
    # -------------------------------------------------
    if rank == 0:
        if config.use_hf:
            from huggingface_hub import login, HfApi
            login(HF_TOKEN)
            hf_api = HfApi()

    # -------------------------------------------------
    # Model and tokenizer
    # -------------------------------------------------
    if rank == 0: print("Loading tokenizer and model...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cấu hình 4-bit quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model = model.to(device)

    # Turn on gradient checkpointing to save memory
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Num parameters
    if rank == 0: print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # Wrap the model with FSDP for distributed training
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )

    # -------------------------------------------------
    # Load wikipedia_vi dataset
    # -------------------------------------------------
    if rank == 0: print("Loading dataset...")
    dataset = load_dataset(config.dataset_name, split="train")
    if rank == 0: print(f"Dataset loaded. Total samples: {len(dataset)}")

    # -------------------------------------------------
    # Preprocess data
    # -------------------------------------------------
    # Keep only title and text column
    dataset = dataset.select_columns(['title', 'text'])

    # Filter out too short samples
    dataset = dataset.filter(filter_function, fn_kwargs={"config": config})
    if rank == 0: print(f"After filtering: {len(dataset)} samples")
    
    # -------------------------------------------------
    # Create splits
    # -------------------------------------------------
    dataset = dataset.shuffle(seed=config.random_seed)

    train_split = dataset.select(range(
        config.train_size
    ))

    valid_split = dataset.select(range(
        config.train_size,
        config.train_size + config.valid_size
    ))

    test_split = dataset.select(range(
        config.train_size + config.valid_size,
        config.train_size + config.valid_size + config.test_size
    ))
    if rank == 0:
        print(f'train split: {len(train_split)} samples')
        print(f'valid split: {len(valid_split)} samples')
        print(f'test split: {len(test_split)} samples')

    train_ds = WikiViDataset(train_split, tokenizer, config.max_length)
    valid_ds = WikiViDataset(valid_split, tokenizer, config.max_length)
    test_ds = WikiViDataset(test_split, tokenizer, config.max_length)

    # -------------------------------------------------
    # Data loader
    # -------------------------------------------------
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_ds,
        batch_size=config.per_device_valid_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    if rank == 0:
        print(f"Train batches: {len(train_dataloader)}")
        print(f"Valid batches: {len(valid_dataloader)}")

    # -------------------------------------------------
    # Optimizer & scheduler
    # -------------------------------------------------
    total_steps = len(train_dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    if rank == 0:
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon
    )

    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Setup gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(device) if config.fp16 else None

    # -------------------------------------------------
    # Test before training
    # -------------------------------------------------
    test_prompts = [
        "Việt Nam là một quốc gia",
        "Tiêu đề: Hà Nội\n\nNội dung:",
        "Lịch sử Việt Nam bắt đầu từ",
        "Văn hóa truyền thống của người Việt",
        "Tiêu đề: Phở\n\nNội dung: Phở là"
    ]
    if rank == 0:   
        print("\n" + "=" * 50)
        print("TESTING THE ORIGINAL MODEL")
        print("=" * 50)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            print("-" * 40)
            
            generated = generate_text(model, tokenizer, device, prompt)
            print(f"Generated: {generated}")

    # -------------------------------------------------
    # Main training loop
    # -------------------------------------------------
    if rank == 0:
        print("Starting training...")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Training history
        training_history = {
            'train_losses': [],
            'train_times': [],
            'valid_losses': [],
            'valid_perplexities': [],
            'valid_times': [],
            'learning_rates': []
        }

        best_valid_loss = float('inf')

    for epoch in range(config.num_train_epochs):
        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
            print(f"{'=' * 50}")
        
        # Training
        if rank == 0: start_time = time.time()
        train_loss = train_epoch(rank, model, train_dataloader, optimizer, scheduler, scaler, device, epoch, config)
        if rank == 0: end_time = time.time()
        
        if rank == 0:
            elapsed_time = end_time - start_time
            train_mins, train_secs = divmod(elapsed_time, 60)
            training_history['train_times'].append(train_mins)
            print(f"Training Time: {int(train_mins)} mins {int(train_secs)} seconds")
        
            training_history['train_losses'].append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        if rank == 0: start_time = time.time()
        valid_loss, perplexity = validate(model, valid_dataloader, device, config)
        if rank == 0: end_time = time.time()
        
        if rank == 0:
            elapsed_time = end_time - start_time
            valid_mins, valid_secs = divmod(elapsed_time, 60)
            training_history['valid_times'].append(valid_mins)
            print(f"Training Time: {int(valid_mins)} mins {int(valid_secs)} seconds")
            
            training_history['valid_losses'].append(valid_loss)
            training_history['valid_perplexities'].append(perplexity)
            print(f"Validation Loss: {valid_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
        
        if rank == 0:
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_time (m)": train_mins,
                    "valid_time (m)": valid_mins,
                    "valid_loss": valid_loss,
                    "perplexity": perplexity,
                })
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                
                model.save_pretrained(config.output_dir)
                tokenizer.save_pretrained(config.output_dir)
                print(f"New best model! Saved to {config.output_dir}")
                
                if config.use_hf:
                    model.push_to_hub(config.hf_repo)
                    tokenizer.push_to_hub(config.hf_repo)
                    print(f"Also saved to repo {config.hf_repo}")
                
            # Save training state
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
                'training_history': training_history
            }, os.path.join(config.output_dir, 'training_state.pt'))
            print(f"Training state saved to {config.output_dir}!")

            if config.use_hf:
                hf_api.upload_file(
                    path_or_fileobj=os.path.join(config.output_dir, 'training_state.pt'),
                    path_in_repo="training_state.pt",
                    repo_id=config.hf_repo,
                    repo_type="model",
                )
            print(f"Training state pushed to repo {config.hf_repo}!")
            
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    if rank == 0: 
        # -------------------------------------------------
        # Test after training
        # -------------------------------------------------
        print("\n" + "=" * 60)
        print("TESTING THE FINE-TUNED MODEL")
        print("=" * 60)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            print("-" * 40)
            
            generated = generate_text(model, tokenizer, device, prompt)
            print(f"Generated: {generated}")

        # -------------------------------------------------
        # Save training log
        # -------------------------------------------------
        # Save comprehensive training log
        training_log = {
            'config': vars(config),
            'model_info': {
                'model_name': config.model_name,
                'num_parameters': model.num_parameters(),
                'dataset_name': config.dataset_name,
                'train_samples': len(train_ds),
                'valid_samples': len(valid_ds)
            },
            'training_results': {
                'best_valid_loss': best_valid_loss,
                'final_perplexity': training_history['valid_perplexities'][-1],
                'total_epochs': config.num_train_epochs,
                'total_steps': total_steps
            },
            'training_history': training_history,
            'training_date': datetime.now().isoformat()
        }

        with open(os.path.join(config.output_dir, 'training_log.json'), 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        print(f"\nTraining log saved to {config.output_dir}/training_log.json")

        if config.use_hf:
            hf_api.upload_file(
                path_or_fileobj=os.path.join(config.output_dir, 'training_log.json'),
                path_in_repo="training_log.json",
                repo_id=config.hf_repo,
                repo_type="model",
            )
        print(f"\nTraining log pushed to repo {config.hf_repo}")

        # -------------------------------------------------
        # Clean up
        # -------------------------------------------------
        if config.use_wandb:
            wandb.finish()

    # -------------------------------------------------
    # Clean up FSDP
    # -------------------------------------------------
    # Clean up the distributed process group
    cleanup_fsdp()

def check_gpu_availability():
    """Check the number of GPUs is available."""
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    return num_gpus

def main():
    """Spawns processes for multi-GPU training."""

    num_gpus = check_gpu_availability()
    
    if num_gpus >= 2:
        print(f"Running with distributed training on {num_gpus} GPUs")
        world_size = num_gpus
        # Spawn processes for distributed training using the available GPUs
        torch.multiprocessing.spawn(fsdp_training, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Not enough GPUs for distributed training!")

# -------------------------------------------------
# Run the script
# -------------------------------------------------
if __name__ == "__main__":
    main()
