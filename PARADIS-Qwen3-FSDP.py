# -------------------------------------------------
# Import modules
# -------------------------------------------------
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# Pytorch Distributed
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# HuggingFace
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP

# General modules
import wandb
import math
import time
import os
import gc
import functools

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
    wandb_project = "PARADIS-Qwen3_0.6B"
    # wandb_project = "PARADIS-Qwen3_1.7B"
    wandb_run_name = "FSDP-No_Distributed_Sampler"

    # HuggingFace configuration
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-FSDP-No_Distributed_Sampler"
    # hf_repo = "h9art/PARADIS-Qwen3_1.7B-10kWikiVi-FSDP"
    
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING "] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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
def setup_wandb(rank, config, config_dict, api_key):
    """Login and create new wandb run or join existing run"""
    
    # Login to wandb
    wandb.login(key=api_key)
    
    # Create run
    if config.use_wandb:
        # New run_id
        run_id = None

        # Primary process, init new run
        if rank == 0:
            new_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config_dict,
                settings=wandb.Settings(
                    x_label=f"rank_{rank}",
                    mode="shared",
                    x_primary=True,
                    x_stats_gpu_device_ids=[rank],
                ),
            )
            run_id = new_run.id

        # Broadcast run_id from primary process (rank 0) to other processes
        run_id_list = [run_id]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]

        # Worker process, join primary process run
        if rank != 0:
            wandb.init(
                project=config.wandb_project,
                id=run_id,
                settings=wandb.Settings(
                    x_label=f"rank_{rank}",
                    mode="shared",
                    x_primary=False,
                    x_stats_gpu_device_ids=[rank],
                    x_update_finish_state=False,
                ),
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
def load_model_n_tokenizer(config, device):
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
    model = model.to(device)

    # Do not use past_key_values
    model.config.use_cache = False

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
def create_train_valid_loader(train_ds, valid_ds, rank, world_size, config):
    """Create distributed samplers and data loader for train and valid set"""
    # # Train data loader
    # train_sampler = DistributedSampler(
    #     train_ds,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True,
    #     drop_last=True
    # )
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=config.per_device_train_batch_size,
    #     sampler=train_sampler,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # # Validation data loader
    # valid_sampler = DistributedSampler(
    #     valid_ds,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False
    # )
    # valid_loader = DataLoader(
    #     valid_ds,
    #     batch_size=config.per_device_valid_batch_size,
    #     sampler=valid_sampler,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    # )
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
def train_epoch(rank, model, dataloader, optimizer, scheduler, scaler, device, epoch, config):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            
            avg_loss = total_loss / (step + 1) * config.gradient_accumulation_steps
            print(f"[Rank {rank}] Step {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

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
def validate(rank, model, dataloader, device, config):
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
                print(f"[Rank {rank}] Step {total_steps}/{len(dataloader)}, Loss: {avg_loss:.4f}")

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

    # Display process group info
    print(f"[Rank {rank}] Process group initialized!")
    print(f"[Rank {rank}] Backend: {dist.get_backend()}")
    print(f"[Rank {rank}] World Size: {dist.get_world_size()}")
    print(f"[Rank {rank}] NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'NOT SET')}")

def cleanup_fsdp():
    """Cleans up the distributed process group."""
    
    if dist.is_initialized():
        dist.destroy_process_group()

# -------------------------------------------------
# FSDP utilities
# -------------------------------------------------
def custom_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    # min_num_params: int = int(1e8),
) -> bool:
    """
    Specify which layer need to be wrapped
    """
    # return nonwrapped_numel >= min_num_params
    return isinstance(module, (nn.Embedding, nn.Linear))

def fsdp_wrap(model):
    """Wrap the model with FSDP for distributed training"""
    # Wrap each Qwen3DecoderLayer of the model
    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen3DecoderLayer,
        },
    )
    
    # Wrap layers that have more than 1000
    my_custom_auto_wrapper_policy = functools.partial(
        custom_auto_wrap_policy
    )
    
    # Apply mixed precision training to model
    fp16_policy = MixedPrecision(
        # Param precision
        param_dtype=torch.float16,
        # Gradient communication precision
        reduce_dtype=torch.float16,
        # Buffer precision
        buffer_dtype=torch.float16,
    )

    # Let's wrap
    sharded_model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        mixed_precision=fp16_policy,
        # cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )

    return sharded_model

def fsdp_activation_checkpointing(model):
    """
    Apply activation checkpointing (gradient checkpointing) to model
    This help reduce GPU memory consumption
    """
    check_fn = lambda submodule: isinstance(submodule, Qwen3DecoderLayer)

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    apply_activation_checkpointing(
        model, 
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )

    return model

# -------------------------------------------------
# Save model checkpoint functions
# -------------------------------------------------
def save_model_checkpoint_old(rank, model, epoch, fullstate_saving_policy, config, hf_api, type: int = 0):
    """Save model with FULL_STATE_DICT checkpoint type"""
    # Notification
    if rank == 0:
        if type == 1: # best model so far
            print("\n" + "-" * 25)
            print("Best save mode")
            print("-" * 25)
        else: # regular save every epoch
            print("\n" + "-" * 25)
            print("Regular save mode")
            print("-" * 25)
    
    # Move model to CPU memory
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_saving_policy):
        cpu_state = model.state_dict()
    print(f"[Rank {rank}] Done writing model state dict")

    # Start saving
    if rank == 0:
        print(f"Saving model...")
        
        # Local saving
        if type == 1: # best model so far
            save_path = os.path.join(config.output_dir, "best-model-ckpt.pt")
        else: # regular save every epoch
            save_path = os.path.join(config.output_dir, "last-model-ckpt.pt")
        
        torch.save(cpu_state, save_path)
        
        if type == 1: # best model so far
            print(f"New best model saved to {save_path}")
        else: # regular save every epoch
            print(f"Model at epoch {epoch + 1} saved to {save_path}")
        
        # Push to HuggingFace hub
        if config.use_hf:
            hf_api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=os.path.basename(save_path),
                repo_id=config.hf_repo,
                repo_type="model",
            )
            print(f"Also saved to Huggingface repo {config.hf_repo}")

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State.
    Since this object is compliant with the Stateful protocol,
    DCP will automatically call state_dict/load_stat_dict as needed in
    the dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods
    on the model and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's,
        # as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

def save_model_checkpoint_new(model, optimizer, epoch, config, hf_api, type: int = 0):
    # Notification
    if type == 1: # best model so far
        print("\n" + "-" * 25)
        print("Best save mode")
        print("-" * 25)
    else: # regular save every epoch
        print("\n" + "-" * 25)
        print("Regular save mode")
        print("-" * 25)
    
    # Local saving
    if type == 1: # best model so far
        save_path = os.path.join(config.output_dir, "best-model-optim-ckpt.pt")
    else: # regular save every epoch
        save_path = os.path.join(config.output_dir, "last-model-optim-ckpt.pt")

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=save_path)
    
    if type == 1: # best model so far
        print(f"New best model and its optimizer saved to {save_path}")
    else: # regular save every epoch
        print(f"Model and its optimizer at epoch {epoch} saved to {save_path}")
    
    # Push to HuggingFace hub
    if config.use_hf:
        hf_api.upload_file(
            path_or_fileobj=save_path,
            path_in_repo=os.path.basename(save_path),
            repo_id=config.hf_repo,
            repo_type="model",
        )
        print(f"Also saved to Huggingface repo {config.hf_repo}")

# -------------------------------------------------
# Main training function for each GPU
# -------------------------------------------------
def fsdp_training(rank, world_size):
    """Train model with FSDP"""

    # -------------------------------------------------
    # General setup
    # -------------------------------------------------
    # Set up env vars
    set_env_var()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    # Set up distributed environment for current rank
    setup_fsdp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Get secrets
    HF_TOKEN, WANDB_API_KEY = get_secrets()

    # Init finetune config
    config = Config()
    config_dict = {
        k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)
    }

    # Set up wandb
    setup_wandb(rank, config, config_dict, WANDB_API_KEY)
    
    # Set up HuggingFace
    hf_api = setup_hf(config, HF_TOKEN)

    # Notify when setup have been done
    if rank == 0:
        print(f"\n{'=' * 50}")
        print("General setup has been done!")
        print(f"{'=' * 50}")

    # -------------------------------------------------
    # Model & Tokenizer
    # -------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 50}")
        print("Model & Tokenizer")
        print(f"{'=' * 50}")
    # Load model and tokenizer
    if rank == 0: print("Loading tokenizer and model...")
    model, tokenizer = load_model_n_tokenizer(config, device)
    if rank == 0: print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # Wrap model with FSDP transformer wrapper
    model = fsdp_wrap(model)
    if rank == 0: print(model)

    # Apply FSDP activation checkpointing
    model = fsdp_activation_checkpointing(model)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 50}")
        print("Dataset")
        print(f"{'=' * 50}")
    # Load and preprocess dataset
    dataset = load_n_preprocess_data(config)
    if rank == 0: print(f"Total: {len(dataset)} samples")
    
    # Create splits
    train_ds, valid_ds = create_train_valid_set(dataset, tokenizer, config)
    if rank == 0:
        print(f'Train samples: {len(train_ds)}')
        print(f'Valid samples: {len(valid_ds)}')

    # Create data loader
    train_dataloader, valid_dataloader = create_train_valid_loader(
        train_ds, valid_ds, rank, world_size, config
    )
    if rank == 0:
        print(f"Train batches: {len(train_dataloader)}")
        print(f"Valid batches: {len(valid_dataloader)}")

    # -------------------------------------------------
    # Optimizer & scheduler
    # -------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 50}")
        print("Optimizer & scheduler")
        print(f"{'=' * 50}")
    # Calculate training step
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
    scaler = ShardedGradScaler() if config.fp16 else None

    # -------------------------------------------------
    # Setup for saving checkpoint
    # -------------------------------------------------
    # Create saving policy
    fullstate_saving_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # Make output directory to store checkpoint
    if rank == 0: os.makedirs(config.output_dir, exist_ok=True)

    # Best model metric
    best_valid_loss = float('inf')

    # -------------------------------------------------
    # Main training loop
    # -------------------------------------------------
    try:
        # Start training
        for epoch in range(config.num_train_epochs):
            # Notify new epoch
            if rank == 0:
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
                print(f"{'=' * 50}")
            
            # Training
            if rank == 0: print("Training...")
            if rank == 0: start_time = time.time()
            train_loss = train_epoch(rank, model, train_dataloader, optimizer, scheduler, scaler, device, epoch, config)
            if rank == 0: end_time = time.time()
            
            if rank == 0:
                elapsed_time = end_time - start_time
                train_mins, train_secs = divmod(elapsed_time, 60)
                print(f"Training Time: {int(train_mins)} mins {int(train_secs)} seconds")            
                print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            if rank == 0: print("Validating...")
            if rank == 0: start_time = time.time()
            valid_loss, perplexity = validate(rank, model, valid_dataloader, device, config)
            if rank == 0: end_time = time.time()
            
            if rank == 0:
                elapsed_time = end_time - start_time
                valid_mins, valid_secs = divmod(elapsed_time, 60)
                print(f"Validation Time: {int(valid_mins)} mins {int(valid_secs)} seconds")                
                print(f"Validation Loss: {valid_loss:.4f}")
                print(f"Perplexity: {perplexity:.2f}")
            
            # Log to wandb
            if rank == 0 and config.use_wandb:
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
                save_model_checkpoint_old(rank, model, epoch, fullstate_saving_policy, config, hf_api, type=1)
                # save_model_checkpoint_new(model, optimizer, epoch, config, hf_api, type=1)

            # save every done epoch
            save_model_checkpoint_old(rank, model, epoch, fullstate_saving_policy, config, hf_api, type=0)
            # save_model_checkpoint_new(model, optimizer, epoch, config, hf_api, type=0)

            # Clean up GPU memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
    finally:
        # Clean up the distributed process group
        cleanup_fsdp()
        # Finish wandb run
        if rank == 0 and config.use_wandb: wandb.finish()

# -------------------------------------------------
# Number of GPUs
# -------------------------------------------------
def check_gpu_availability():
    """Check the number of GPUs is available."""
    num_gpus = torch.cuda.device_count()
    return num_gpus

# -------------------------------------------------
# Main function for spawning process for each GPU
# -------------------------------------------------
def main():
    """Spawns processes for multi-GPU training."""
    num_gpus = check_gpu_availability()
    if num_gpus >= 2:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"[Rank {rank}] Starting with world_size = {world_size}")
        fsdp_training(rank, world_size)
    else:
        print("Not enough GPUs for distributed training!")

# -------------------------------------------------
# Run the script
# -------------------------------------------------
if __name__ == "__main__":
    main()
