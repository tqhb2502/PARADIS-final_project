import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
deimport gc
import deepspeed

# Constants
RANDOM_SEED = 42
MAX_LENGTH = 128

# Config class remains unchanged
class Config:
    model_name = "Qwen/Qwen3-0.6B"
    dataset_name = "vietgpt/wikipedia_vi"
    output_dir = "./qwen-vietnamese-wiki-finetuned"
    num_train_epochs = 5
    per_device_train_batch_size = 2
    per_device_valid_batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_length = MAX_LENGTH
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    logging_steps = 40
    save_strategy = "epoch"
    valid_strategy = "epoch"
    fp16 = False
    num_workers = os.cpu_count()
    use_wandb = True
    wandb_project = "PARADIS-Qwen3_0.6B"
    wandb_run_name = "ZeRO3"
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen3_0.6B-10kWikiVi-ZeRO3"
    train_size = 10000
    valid_size = 10000
    test_size = 5000
    min_text_length = 50

# Custom dataset unchanged
class WikiViDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        combined_text = f"Tiêu đề: {item['title']}\n\nNội dung: {item['text']}"
        tokenized = self.tokenizer(combined_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Set distributed env vars
def set_env_var():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", os.getenv("WORLD_SIZE", "1"))
    os.environ.setdefault("RANK", os.getenv("RANK", "0"))

# Secrets and setups remain same

def get_secrets():
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    return user_secrets.get_secret("HF_TOKEN"), user_secrets.get_secret("WANDB_API_KEY")

def setup_wandb(config, config_dict, api_key):
    wandb.login(key=api_key)
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config_dict)

def setup_hf(config, api_key):
    if config.use_hf:
        from huggingface_hub import login, HfApi
        login(api_key)
        return HfApi()
    return None

# Load model/tokenizer unchanged

def load_model_n_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer

# Data loading/preprocessing unchanged

def filter_function(example, config):
    return example['text'] and example['title'] and len(example['text'].strip()) > config.min_text_length

def load_n_preprocess_data(config):
    ds = load_dataset(config.dataset_name, split="train").select_columns(['title', 'text'])
    return ds.filter(filter_function, fn_kwargs={"config": config})

def create_train_valid_set(dataset, tokenizer, config):
    ds = dataset.shuffle(seed=RANDOM_SEED)
    train = ds.select(range(config.train_size))
    valid = ds.select(range(config.train_size, config.train_size+config.valid_size))
    return WikiViDataset(train, tokenizer, config.max_length), WikiViDataset(valid, tokenizer, config.max_length)

def create_loaders(train_ds, valid_ds, config):
    train_loader = DataLoader(train_ds, batch_size=config.per_device_train_batch_size, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.per_device_valid_batch_size, num_workers=config.num_workers, pin_memory=True)
    return train_loader, valid_loader

# Updated training/validation functions using DeepSpeed engine

def train_epoch(engine, dataloader, epoch, config):
    engine.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        inputs, masks, labels = batch['input_ids'].to(engine.local_rank), batch['attention_mask'].to(engine.local_rank), batch['labels'].to(engine.local_rank)
        outputs = engine(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        engine.backward(loss)
        engine.step()
        total_loss += loss.item()
        if (step+1) % config.logging_steps == 0:
            avg = total_loss/(step+1)
            print(f"Epoch {epoch} Step {step+1}/{len(dataloader)}, Loss: {avg:.4f}")
            if config.use_wandb:
                wandb.log({"train_loss": avg, "epoch": epoch, "step": step+1})
    return total_loss/len(dataloader)

def validate(engine, dataloader, config):
    engine.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs, masks, labels = batch['input_ids'].to(engine.local_rank), batch['attention_mask'].to(engine.local_rank), batch['labels'].to(engine.local_rank)
            outputs = engine(input_ids=inputs, attention_mask=masks, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss/len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss, math.exp(avg_loss)

# Main
if __name__ == "__main__":
    set_env_var()
    deepspeed.init_distributed()
    torch.manual_seed(RANDOM_SEED)
    config = Config()
    cfg_dict = {k:v for k,v in Config.__dict__.items() if not k.startswith("__")}
    HF_TOKEN, WANDB_API_KEY = get_secrets()
    setup_wandb(config, cfg_dict, WANDB_API_KEY)
    setup_hf(config, HF_TOKEN)
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_n_tokenizer(config)
    dataset = load_n_preprocess_data(config)
    train_ds, valid_ds = create_train_valid_set(dataset, tokenizer, config)
    train_loader, valid_loader = create_loaders(train_ds, valid_ds, config)
    total_steps = len(train_loader)*config.num_train_epochs
    warmup_steps = int(total_steps*config.warmup_ratio)
    optim = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    # DeepSpeed
    ds_cfg = json.load(open("zero_stage3_offload_config.json"))
    engine, optim, _, sched = deepspeed.initialize(model=model, optimizer=optim, lr_scheduler=sched,
                                                  config_params=ds_cfg, model_parameters=model.parameters())
    best_loss = float('inf')
    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch+1}/{config.num_train_epochs}")
        train_loss = train_epoch(engine, train_loader, epoch+1, config)
        valid_loss, ppl = validate(engine, valid_loader, config)
        if valid_loss < best_loss:
            best_loss = valid_loss
            engine.save_checkpoint(config.output_dir, tag=f"best_epoch{epoch+1}")
    if config.use_wandb:
        wandb.finish()
