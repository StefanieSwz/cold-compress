import os
import sys
import time
import pdb
from typing import *
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributed.pipelining import Pipe
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch._dynamo.config
import torch._inductor.config
from datasets import load_dataset
from tp import maybe_init_dist

from tokenizer import get_tokenizer, encode, TokenizersChatFormat

from generation_utils import (
    load_model,
    device_sync,
    setup_caches,
)


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def setup_env_variables():
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"  # Set master address
    os.environ["MASTER_PORT"] = "12345"  # Set master port
    os.environ["LOCAL_WORLD_SIZE"] = "8"  # Total number of processes
    print("Rank: ", os.environ.get("RANK"))
    print("Rank: ", os.environ.get("LOCAL_RANK"))
    os.environ["RANK"] = os.environ.get("RANK", "0")  # Default to rank 0
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")  # Default to GPU 0


# Call the function to set environment variables
setup_env_variables()

# Configuration
CHECKPOINT_PATH = Path("./checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth")
TOKENIZER_PATH = Path(CHECKPOINT_PATH).parent / "tokenizer.model"
DATASET_PATH = "HuggingFaceH4/ultrachat_200k"
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 5e-5
BLOCK_SIZE = 128
MAX_SEQ_LENGTH = 2048
IGNORE_INDEX = -100


assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
if not TOKENIZER_PATH.is_file():
    # If there's no tokenizer.model, try to load the tokenizer from the parent directory
    # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
    TOKENIZER_PATH = CHECKPOINT_PATH.parent


# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pdb.set_trace()
rank = maybe_init_dist()
print("Rank: ", rank)
use_tp = rank is not None
# if use_tp:
#     if rank != 0:
#         # only print on rank 0
#         print = lambda *args, **kwargs: None

print(f"Using device={DEVICE}")
PRECISION = torch.bfloat16
IS_CHAT = (
    "chat" in str(CHECKPOINT_PATH).lower() or "instruct" in str(CHECKPOINT_PATH).lower()
)

print("Loading model ...")
t0 = time.time()
model = load_model(CHECKPOINT_PATH, DEVICE, PRECISION, use_tp)
print("Model Type:", model.__class__.__name__)
device_sync(device=DEVICE)  # MKG
print(f"Time to load model: {time.time() - t0:.02f} seconds")

# Load tokenizer
tokenizer_ultrachat = get_tokenizer(TOKENIZER_PATH, CHECKPOINT_PATH, is_chat=IS_CHAT)

# Define the cache strategy and related configurations
cache_strategy = ["lightweight"] * len(
    model.layers
)  # Example: lightweight for all layers
max_cache_length = [MAX_SEQ_LENGTH] * len(model.layers)
recent_window = [32] * len(model.layers)
prompt_compression_strategy = ["none"] * len(model.layers)

# copied from the generation kwargs
cache_kwargs = {
    # "prompt": "What is a cold compress?",
    "max_new_tokens": 512,
    "cache_config": None,
    "checkpoint_path": Path("checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth"),
    "profile": None,
    "compile": False,
    "device": "cuda",
    "attn_top_k": 1.0,
    "max_cache_length": [1.0],
    "cache_bits": None,
    "cache_length_pattern": "tile",
    "cache_strategy": ["lightweight"],
    "cache_strategy_pattern": "tile",
    "feed_long_prompts": False,
    "prompt_compression_strategy": ["recent_global"],
    "global_tokens": 1,
    "recent_window": 10,
    "history_window_size": 1,
    "attn_thresholding": False,
    "model_type": "linear",
    "min_recovery_frac": 0.9,
}

setup_caches(
    model, tokenizer_ultrachat, DEVICE, MAX_SEQ_LENGTH, cache_kwargs
)  # access compression technique


# Function to partition layers dynamically across available GPUs
def create_pipeline(model, num_gpus):
    # Get the total number of layers
    total_layers = len(model.layers)

    # Compute the number of layers per GPU
    layers_per_gpu = (total_layers + num_gpus - 1) // num_gpus  # Ceiling division

    # Partition layers into chunks for each GPU
    partitions = []
    for i in range(num_gpus):
        start = i * layers_per_gpu
        end = min((i + 1) * layers_per_gpu, total_layers)
        if start < end:  # Avoid empty partitions
            partition = nn.Sequential(*model.layers[start:end]).to(f"cuda:{i}")
            partitions.append(partition)

    # Create the pipeline
    return Pipe(nn.Sequential(*partitions), chunks=1)


# Load dataset
# Dataset Preparation
class PromptIterableDataset(IterableDataset):
    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: TokenizersChatFormat,
        max_seq_length: Optional[int] = 512,
        teacher_forcing: Optional[bool] = True,
        truncate_method: Optional[str] = "tail",
    ):
        assert hasattr(
            raw_dataset, "__iter__"
        ), f"The dataset must have __iter__ method. Dataset is {raw_dataset}"
        assert hasattr(
            raw_dataset, "__len__"
        ), f"The dataset must have __len__ method. Dataset is {raw_dataset}"

        self.raw_dataset = raw_dataset
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, "Teacher forcing must be enabled."

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", "Only tail truncation is supported."

    def tokenize_example(self, example):
        """
        Tokenizes a single example using the chat tokenizer's encoding method.
        """
        # Use the chat tokenizer to encode the dialog
        dialog = example["messages"]  # Assumes "messages" contains the list of messages
        tokenized_ids = self.tokenizer.encode_dialog_prompt(dialog)

        # Convert to a tensor
        tokenized_ids = torch.tensor(tokenized_ids, dtype=torch.long)

        # Create labels for teacher forcing
        labels = tokenized_ids.clone()

        # Compute masking indices for user tokens
        current_token_position = 0  # Track token positions
        for i, message in enumerate(dialog):
            if (
                i < 2
            ):  # for memory reasons, only consider the first 2 messages, prompt and assistant
                # Get tokenized length of the current message
                message_tokens = self.tokenizer.encode_dialog_prompt([message])
                message_length = len(message_tokens)

                if message["role"] == "user":
                    # Mask user message tokens
                    labels[
                        current_token_position : current_token_position + message_length
                    ] = IGNORE_INDEX

                # Update token position
                current_token_position += message_length

        return {
            "input_ids": tokenized_ids,
            "labels": labels,
        }

    def truncate(self, tokenized_example):
        """
        Truncates tokenized examples to the maximum sequence length.
        """
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][
                    : -(old_len - self.max_seq_length)
                ]
        return tokenized_example

    def __iter__(self):
        """
        Iterates over the raw dataset and yields tokenized and truncated examples.
        """
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        """
        Get a specific item by index. Note: Requires the raw_dataset to support indexing.
        """
        # Ensure the raw_dataset supports indexing
        if not hasattr(self.raw_dataset, "__getitem__"):
            raise TypeError("raw_dataset must support indexing to use __getitem__.")

        example = self.raw_dataset[index]
        tokenized_example = self.tokenize_example(example)
        tokenized_example = self.truncate(tokenized_example)
        return tokenized_example


ultrachat_datadic = load_dataset(DATASET_PATH)
train_dataset = PromptIterableDataset(
    ultrachat_datadic["train_gen"].select(range(10)),
    tokenizer_ultrachat,
    MAX_SEQ_LENGTH,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# for name, param in model.named_parameters():
#     print(name)
for name, param in model.named_parameters():
    if "attention.kv_cache.models" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

pdb.set_trace()

# Optimizer
optimizer = AdamW(model.get_lightweight_params(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()

# Training Loop
print("Starting training...")
model.train()
model.set_train_mode(True)
for epoch in range(EPOCHS):
    total_loss = 0
    pdb.set_trace()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        batch = {
            key: value.to(DEVICE) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        input_ids = batch["input_ids"]  # Inputs
        labels = batch["labels"]  # Targets

        # Input positions and prefill flags
        prompt_length = input_ids.size(1)
        input_pos = torch.arange(0, prompt_length, device=DEVICE)

        causal_mask = (
            torch.tril(torch.ones(len(input_pos), len(input_pos), dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(input_ids.device)
        )
        # Forward pass
        optimizer.zero_grad()
        logits = model(
            input_ids, input_pos, mask=causal_mask, is_prefill=True
        )  # prefill = true for training on full sequence
        # pdb.set_trace()
        # Loss computation
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss_epoch = loss(logits, labels)

        # Backward pass
        loss_epoch.backward()

        optimizer.step()

        total_loss += loss_epoch.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} completed. Average Loss: {avg_loss:.4f}")

# Save the model
MODEL_SAVE_PATH = "./lightweight_weights/trained_transformer.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
