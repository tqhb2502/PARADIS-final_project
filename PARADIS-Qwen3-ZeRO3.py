# -------------------------------------------------
# Import modules
# -------------------------------------------------
# Pytorch
import torch
from torch.utils.data import DataLoader, Dataset

# HuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# General modules
import wandb
import math
import time
import os
import json

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
    max_length = MAX_LENGTH
    
    # Logging and saving
    logging_steps = 40
    save_strategy = "epoch"
    valid_strategy = "epoch"
    
    # Other settings
    num_workers = os.cpu_count()
    
    # W&B configuration
    use_wandb = False
    wandb_project = "PARADIS-Qwen3_0.6B"
    # wandb_project = "PARADIS-Qwen3_1.7B"
    wandb_run_name = "ZeRO3"

    # HuggingFace configuration
    use_hf = False
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-ZeRO3"
    # hf_repo = "h9art/PARADIS-Qwen3_1.7B-10kWikiVi-ZeRO3"
    
    # Dataset
    train_size = 10000
    valid_size = 10000
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
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", os.getenv("WORLD_SIZE", "1"))
    os.environ.setdefault("RANK", os.getenv("RANK", "0"))

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
        pin_memory=True,
    )

    # Validation data loader
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.per_device_valid_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_epoch(engine, dataloader, epoch, config):
    """Train model for one epoch."""
    
    engine.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(engine.local_rank)
        attention_mask = batch['attention_mask'].to(engine.local_rank)
        labels = batch['labels'].to(engine.local_rank)
        
        # Forward pass
        outputs = engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass
        engine.backward(loss)
        
        # Calculate total loss
        total_loss += loss.item()

        # Update weights
        engine.step()
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            
            avg_loss = total_loss / (step + 1)
            print(f"Step {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

            if config.use_wandb:
                wandb.log({
                    "train_loss": avg_loss,
                    "train_step": epoch * len(dataloader) + step + 1
                })

    return total_loss / len(dataloader)

# -------------------------------------------------
# Validation function
# -------------------------------------------------
def validate(engine, dataloader, config):
    """Validate the model."""
    
    engine.eval()
    total_loss = 0
    
    with torch.no_grad():        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(engine.local_rank)
            attention_mask = batch['attention_mask'].to(engine.local_rank)
            labels = batch['labels'].to(engine.local_rank)
            
            # Forward pass
            outputs = engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Logging
            if (step + 1) % config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

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

    # Set up distributed environment for current rank
    deepspeed.init_distributed()
    
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
    # Setup DeepSpeed ZeRO 3
    # -------------------------------------------------
    ds_cfg = json.load(open("zero_stage3_offload_config.json"))
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_cfg,
        model_parameters=model.parameters()
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
            train_loss = train_epoch(engine, train_dataloader, epoch, config)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            train_mins, train_secs = divmod(elapsed_time, 60)
            print(f"Training Time: {int(train_mins)} mins {int(train_secs)} seconds")            
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            print("Validating...")
            start_time = time.time()
            valid_loss, perplexity = validate(engine, valid_dataloader, config)
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
                engine.save_checkpoint(config.output_dir, tag=f"best_epoch{epoch+1}")

            # save every done epoch
            engine.save_checkpoint(config.output_dir, tag=f"regular_epoch{epoch+1}")
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
