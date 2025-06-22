# -------------------------------------------------
# Import modules
# -------------------------------------------------
# Pytorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# HuggingFace
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# General modules
import wandb
import math
import time
import os
import gc

# DeepSpeed
import deepspeed

# -------------------------------------------------
# Constants
# -------------------------------------------------
RANDOM_SEED = 42
MAX_LENGTH = 128
TEST_PROMPTS = [
    "Việt Nam là một quốc gia",
    "Tiêu đề: Hà Nội\n\nNội dung:",
    "Lịch sử Việt Nam bắt đầu từ",
    "Văn hóa truyền thống của người Việt",
    "Tiêu đề: Phở\n\nNội dung: Phở là"
]

# -------------------------------------------------
# Finetune config
# -------------------------------------------------
class Config:
    """All config for this finetune"""

    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-1.7B"
    dataset_name = "vietgpt/wikipedia_vi"
    
    # Training configuration
    output_dir = "./qwen-vietnamese-wiki-finetuned"
    num_train_epochs = 5
    per_device_train_batch_size = 2
    per_device_valid_batch_size = 2
    gradient_accumulation_steps = 1
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
    fp16 = False
    num_workers = os.cpu_count()
    
    # W&B configuration
    use_wandb = True
    wandb_project = "PARADIS-Qwen3_0.6B"
    # wandb_project = "PARADIS-Qwen3_1.7B"
    wandb_run_name = "ZeRO3"

    # HuggingFace configuration
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-ZeRO3"
    # hf_repo = "h9art/PARADIS-Qwen3_1.7B-10kWikiVi-ZeRO3"
    
    # Dataset
    train_size = 10000
    valid_size = 10000
    test_size = 5000
    min_text_length = 50

# -------------------------------------------------
# Custom dataset
# -------------------------------------------------
class WikiViDataset(Dataset):
    """Custom dataset for Wikipedia Vietnamese Dataset on HuggingFace"""

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
# Environment variables
# -------------------------------------------------
def set_env_var():
    """Set value for environment variables"""
    pass

# -------------------------------------------------
# Get Kaggle secrets
# -------------------------------------------------
def get_secrets():
    """Get Kaggle secrets"""
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    return HF_TOKEN, WANDB_API_KEY

# -------------------------------------------------
# Setup wandb
# -------------------------------------------------
def setup_wandb(config, config_dict, api_key):
    """Login and create new wandb run or join existing run"""
    
    # Login to wandb
    wandb.login(key=api_key)
    
    # Create run
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config_dict,
        )

# -------------------------------------------------
# Setup HuggingFace
# -------------------------------------------------
def setup_hf(config, api_key):
    """Login and setup HuggingFace API for download/upload files"""
    if config.use_hf:
        from huggingface_hub import login, HfApi
        login(api_key)
        hf_api = HfApi()
        return hf_api
    return None

# -------------------------------------------------
# Model and tokenizer
# -------------------------------------------------
def load_model_n_tokenizer(config):
    """Download and set model and tokenizer up"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Do not use past_key_values
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer

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
# Load wikipedia_vi dataset
# -------------------------------------------------
def load_n_preprocess_data(config):
    """Download Wikipedia Vietnamese dataset from HuggingFace and preprocess"""
    # Download dataset
    dataset = load_dataset(config.dataset_name, split="train")
    # Keep only title and text column
    dataset = dataset.select_columns(['title', 'text'])
    # Filter out too short samples
    dataset = dataset.filter(filter_function, fn_kwargs={"config": config})
    return dataset

# -------------------------------------------------
# Create data splits
# -------------------------------------------------
def create_train_valid_set(dataset, tokenizer, config):
    """Shuffle data and select samples to train split and valid split"""
    dataset = dataset.shuffle(seed=RANDOM_SEED)

    train_split = dataset.select(range(
        config.train_size
    ))

    valid_split = dataset.select(range(
        config.train_size,
        config.train_size + config.valid_size
    ))

    train_ds = WikiViDataset(train_split, tokenizer, config.max_length)
    valid_ds = WikiViDataset(valid_split, tokenizer, config.max_length)
    return train_ds, valid_ds
    
# -------------------------------------------------
# Data loader
# -------------------------------------------------
def create_train_valid_loader(train_ds, valid_ds, config):
    """Create distributed samplers and data loader for train and valid set"""
    # Train data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # Validation data loader
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.per_device_valid_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config):
    """Train model for one epoch."""
    
    model.train()
    total_loss = 0
    optimizer.zero_grad()
        
    for step, batch in enumerate(dataloader):
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
        
        # Calculate total loss
        total_loss += loss.item()

        # Update weights every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            
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
        for batch in dataloader:
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
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

            # Logging
            if total_steps % config.logging_steps == 0:
                avg_loss = total_loss / total_steps
                print(f"Step {total_steps}/{len(dataloader)}, Loss: {avg_loss:.4f}")

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
# Test model generation
# -------------------------------------------------
def test_model_generation(model, tokenizer, device, test_prompts):
    """Run model generation with some test prompts"""
    print("\n" + "=" * 50)
    print("TESTING THE MODEL")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        generated = generate_text(model, tokenizer, device, prompt)
        print(f"Generated: {generated}")

# -------------------------------------------------
# Main training function for each GPU
# -------------------------------------------------
def distributed_training():
    """Distributed training the model"""

    # -------------------------------------------------
    # General setup
    # -------------------------------------------------
    # Set up env vars
    set_env_var()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    # Get secrets
    HF_TOKEN, WANDB_API_KEY = get_secrets()

    # Init finetune config
    config = Config()
    config_dict = {
        k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)
    }

    # Set up wandb
    setup_wandb(config, config_dict, WANDB_API_KEY)
    
    # Set up HuggingFace
    hf_api = setup_hf(config, HF_TOKEN)

    # Notify when setup have been done
    print(f"\n{'=' * 50}")
    print("General setup has been done!")
    print(f"{'=' * 50}")

    # -------------------------------------------------
    # Model & Tokenizer
    # -------------------------------------------------
    print(f"\n{'=' * 50}")
    print("Model & Tokenizer")
    print(f"{'=' * 50}")
    # Load model and tokenizer
    print("Loading tokenizer and model...")
    model, tokenizer = load_model_n_tokenizer(config)
    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    print(f"\n{'=' * 50}")
    print("Dataset")
    print(f"{'=' * 50}")
    # Load and preprocess dataset
    dataset = load_n_preprocess_data(config)
    print(f"Total: {len(dataset)} samples")
    
    # Create splits
    train_ds, valid_ds = create_train_valid_set(dataset, tokenizer, config)
    
    print(f'Train samples: {len(train_ds)}')
    print(f'Valid samples: {len(valid_ds)}')

    # Create data loader
    train_dataloader, valid_dataloader = create_train_valid_loader(train_ds, valid_ds, config)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Valid batches: {len(valid_dataloader)}")

    # -------------------------------------------------
    # Optimizer & scheduler
    # -------------------------------------------------
    print(f"\n{'=' * 50}")
    print("Optimizer & scheduler")
    print(f"{'=' * 50}")
    # Calculate training step
    total_steps = len(train_dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
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
    # Setup DeepSpeed ZeRO 3
    # -------------------------------------------------
    ds_config = "zero_stage3_offload_config.json"
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # -------------------------------------------------
    # Setup for saving checkpoint
    # -------------------------------------------------
    # Make output directory to store checkpoint
    os.makedirs(config.output_dir, exist_ok=True)

    # Best model metric
    best_valid_loss = float('inf')

    # -------------------------------------------------
    # Main training loop
    # -------------------------------------------------
    try:
        # Start training
        for epoch in range(config.num_train_epochs):
            # Notify new epoch
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
            print(f"{'=' * 50}")
            
            # Training
            print("Training...")
            start_time = time.time()
            train_loss = train_epoch(model_engine, train_dataloader, optimizer, scheduler, model_engine.device, epoch, config)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            train_mins, train_secs = divmod(elapsed_time, 60)
            print(f"Training Time: {int(train_mins)} mins {int(train_secs)} seconds")            
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            print("Validating...")
            start_time = time.time()
            valid_loss, perplexity = validate(model_engine, valid_dataloader, model_engine.device, config)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            valid_mins, valid_secs = divmod(elapsed_time, 60)
            print(f"Validation Time: {int(valid_mins)} mins {int(valid_secs)} seconds")                
            print(f"Validation Loss: {valid_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_time (m)": train_mins,
                    "valid_time (m)": valid_mins,
                    "valid_loss": valid_loss,
                    "perplexity": perplexity,
                })
            
            # Check if this is the best model so far
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # save best model

            # save every done epoch

            # Clean up GPU memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Finish wandb run
        if config.use_wandb: wandb.finish()

# -------------------------------------------------
# Run the script
# -------------------------------------------------
if __name__ == "__main__":
    distributed_training()
