import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
import wandb

#======================================================
# Config and Setup
#======================================================
RANDOM_SEED = 42

class Config:
    model_name = "Qwen/Qwen3-0.6B"
    dataset_name = "vietgpt/wikipedia_vi"
    output_dir = "./qwen-vietnamese-wiki-finetuned"
    num_train_epochs = 3
    per_device_train_batch_size = 2
    per_device_valid_batch_size = 2
    gradient_accumulation_steps = 8
    learning_rate = 5e-5
    max_length = 128
    warmup_ratio = 0.1
    weight_decay = 0.01

    # Logging and saving
    logging_steps = 40
    save_strategy = "epoch"
    valid_strategy = "epoch"

    # Optimization settings
    adam_epsilon = 1e-8
    max_grad_norm = 1.0

    # Other settings
    fp16 = True
    num_workers = os.cpu_count()

    # W&B configuration
    use_wandb = True
    wandb_project = "PARADIS-Qwen3_0.6B"
    wandb_run_name = "TORCH.PIPELINE"

    # HuggingFace configuration
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-torch.dist_pipeline"

    train_size = 10000
    valid_size = 2000


def set_env_var():
    """Set value for environment variables"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING "] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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

def get_secrets():
    """Get Kaggle secrets"""
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    return HF_TOKEN, WANDB_API_KEY

def setup_hf(config, api_key):
    """Login and setup HuggingFace API for download/upload files"""
    if config.use_hf:
        from huggingface_hub import login, HfApi
        login(api_key)
        hf_api = HfApi()
        return hf_api
    return None

def ddp_setup():
    init_process_group(
        backend="nccl",
        rank=os.environ["LOCAL_RANK"],
        world_size=os.environ["WORLD_SIZE"]
    )
    torch.cuda.set_device(os.environ["LOCAL_RANK"])

#======================================================
# Dataset
#======================================================
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
        labels = input_ids.clone()  # In causal LM, labels is the same with input_ids
        labels[attention_mask == 0] = -100  # Do not calculate loss on padding tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def filter_function(example, config):
    """Filter out empty or very short texts"""

    return (
            example['text'] is not None and
            example['title'] is not None and
            len(example['text'].strip()) > config.min_text_length
    )


def load_n_preprocess_data(config):
    """Download Wikipedia Vietnamese dataset from HuggingFace and preprocess"""
    # Download dataset
    dataset = load_dataset(config.dataset_name, split="train")
    # Keep only title and text column
    dataset = dataset.select_columns(['title', 'text'])
    # Filter out too short samples
    dataset = dataset.filter(filter_function, fn_kwargs={"config": config})
    return dataset


def create_train_valid_set(dataset, tokenizer, config):
    """Shuffle data and select samples to train split and valid split"""
    dataset = dataset.shuffle(seed=42)

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


def create_train_valid_loader(train_ds, valid_ds, rank, world_size, config):
    """Create distributed samplers and data loader for train and valid set"""
    # Train data loader
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation data loader
    valid_sampler = DistributedSampler(
        valid_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.per_device_valid_batch_size,
        sampler=valid_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader

#======================================================
# Model
#======================================================
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

#======================================================
# Training
#======================================================
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, local_rank, rank, world_size, model, train_data, optimizer, output_dir, num_epochs,
                 save_every, report_rate, max_grad_norm):

        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(local_rank)
        self.model = DDP(self.model, device_ids=[local_rank])
        self.train_data = train_data
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.report_rate = report_rate
        self.local_rank = local_rank
        self.max_grad_norm = max_grad_norm

        if self.local_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        """Save model checkpoint (only by the main process)."""
        if self.local_rank == 0:
            save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), save_path)
            print(f"[GPU {self.local_rank}] Checkpoint saved at epoch {epoch} to {save_path}")

    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        batch = {k: v.to(self.local_rank) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def train(self):
        """Run the full training loop"""
        for epoch in range(self.num_epochs):
            self.train_data.sampler.set_epoch(epoch)
            sum_loss = 0.0

            for step, batch in enumerate(self.train_data):
                loss = self.train_step(batch)
                sum_loss += loss

                if step != 0 and step % self.report_rate == 0:
                    avg_loss = sum_loss / self.report_rate
                    print(f"[GPU {self.local_rank}] | Epoch {epoch}/{self.num_epochs} |"
                          f"Step {step} | Loss: {avg_loss:.4f}")

                    # global report: reporting the average loss across all GPUs
                    avg_loss = torch.Tensor([avg_loss]).to(self.local_rank)
                    # this will get the sum of the tensor from all GPUs, and put result only on GPU 0
                    dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM)
                    if self.local_rank == 0:
                        all_gpus_avg_loss = avg_loss / self.world_size
                        print(f"All_GPUs_Loss: {all_gpus_avg_loss.item():.4f}")
                    # reset the loss
                    sum_loss = 0.0

            # save checkpoint
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self.save_checkpoint(epoch + 1)


def distributed_train(rank, world_size):
    set_env_var()
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    ddp_setup()
    device = torch.device(f"cuda:{rank}")

    config = Config()
    config_dict = {
        k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)
    }

    HF_TOKEN, WANDB_API_KEY = get_secrets()

    setup_wandb(rank, config, config_dict, WANDB_API_KEY)
    hf_api = setup_hf(config, HF_TOKEN)
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


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(
        local_rank=int(os.environ["LOCAL_RANK"]),
        rank=int(os.environ['RANK']),
        world_size=int(os.environ["WORLD_SIZE"]),
        model=model,
        train_data=train_dataloader,
        optimizer=optimizer,
        output_dir=config.output_dir,
        num_epochs=config.num_train_epochs,
        save_every=500,
        report_rate=config.logging_steps,
        max_grad_norm=config.max_grad_norm
    )

    # Step 5: Begin Training
    trainer.train()

    # Step 6: Clean up DDP Resources
    destroy_process_group()

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed_train(rank, world_size)
    else:
        print("Not enough GPUs for distributed training!")

if __name__ == "__main__":
    main()