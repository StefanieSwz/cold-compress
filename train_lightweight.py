import os
import sys
import time
import argparse
import signal
import warnings
import subprocess
from typing import Any, Dict, Optional, Iterator, Tuple
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch._dynamo.config
import torch._inductor.config
from datasets import load_dataset
from tp import maybe_init_dist, handle_sigint, handle_uncaught_exception

from tokenizer import get_tokenizer, TokenizersChatFormat
from cache_utils import save_lightweight_temp
from generation_utils import load_model, device_sync, setup_caches, reset_caches

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

# support running without installing as a package
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
signal.signal(signal.SIGINT, handle_sigint)
sys.excepthook = handle_uncaught_exception


class PromptIterableDataset(IterableDataset):
    """
    A dataset wrapper that iterates over a raw dataset, tokenizes each example using a chat-format tokenizer,
    and truncates the tokenized sequences to a maximum sequence length. This dataset is iterable and supports
    teacher forcing, assuming that each example contains a dialogue with multiple messages.

    Attributes:
        raw_dataset (Dataset): The raw dataset containing examples with a "messages" field.
        tokenizer (TokenizersChatFormat): The tokenizer used for encoding dialog prompts.
        max_seq_length (int): The maximum sequence length for tokenized examples.
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: TokenizersChatFormat,
        max_seq_length: Optional[int] = 512,
        ignore_index: int = -100,
    ):
        """
        Initialize the PromptIterableDataset.

        Args:
            raw_dataset (Dataset): The raw dataset, which must support both iteration and length queries.
            tokenizer (TokenizersChatFormat): A tokenizer with a method `encode_dialog_prompt` for encoding chat dialogs.
            max_seq_length (Optional[int], optional): Maximum length of the tokenized sequence. Defaults to 512.

        Raises:
            AssertionError: If raw_dataset does not have __iter__ or __len__ methods.
        """
        assert hasattr(
            raw_dataset, "__iter__"
        ), f"The dataset must have __iter__ method. Dataset is {raw_dataset}"
        assert hasattr(
            raw_dataset, "__len__"
        ), f"The dataset must have __len__ method. Dataset is {raw_dataset}"

        # self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index

        self.valid_indices = [
            idx
            for idx in range(len(raw_dataset))
            if self.is_valid(self.truncate(self.tokenize_example(raw_dataset[idx])))
        ]

        # Create valid dataset from valid indices
        self.valid_dataset = [raw_dataset[idx] for idx in self.valid_indices]

    def is_valid(self, tokenized_example):
        """Check if the truncated example still contains assistant labels."""
        labels = tokenized_example["labels"]
        return (labels != self.ignore_index).any()

    def tokenize_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single example using the chat tokenizer's encoding method.

        The method encodes the dialog present in the "messages" field of the example,
        creates a tensor of token IDs, and then masks out user message tokens by setting
        them to IGNORE_INDEX.

        Args:
            example (Dict[str, Any]): A dictionary representing a chat example with a "messages" key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys "input_ids" and "labels" containing tokenized tensors.
        """
        dialog = example["messages"]
        tokenized_ids = self.tokenizer.encode_dialog_prompt(dialog)
        tokenized_ids = torch.tensor(tokenized_ids, dtype=torch.long)
        labels = tokenized_ids.clone()

        current_token_position = 0
        for message in dialog:
            message_tokens = self.tokenizer.encode_dialog_prompt([message])
            message_length = len(message_tokens)

            if message["role"] == "user":
                labels[
                    current_token_position : current_token_position + message_length
                ] = self.ignore_index

            current_token_position += message_length

        return {
            "input_ids": tokenized_ids,
            "labels": labels,
        }

    def truncate(
        self, tokenized_example: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Truncates tokenized examples to the maximum sequence length.

        If the length of the "input_ids" exceeds max_seq_length, each field in the tokenized_example
        is truncated from the tail to ensure the length does not exceed max_seq_length.

        Args:
            tokenized_example (Dict[str, torch.Tensor]): A dictionary containing tokenized data with key "input_ids".

        Returns:
            Dict[str, torch.Tensor]: The tokenized example truncated to max_seq_length.
        """
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][
                    : -(old_len - self.max_seq_length)
                ]
        return tokenized_example

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterates over the raw dataset and yields tokenized and truncated examples.

        Yields:
            Iterator[Dict[str, torch.Tensor]]: An iterator over processed examples.
        """
        for example in self.valid_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.truncate(tokenized_example)
            yield tokenized_example

    def __len__(self) -> int:
        """
        Returns the number of examples in the raw dataset.

        Returns:
            int: The length of the usable items based on max_seq_length.
        """
        return len(self.valid_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a specific item by index from the raw dataset.

        Note: This method requires that the raw_dataset supports indexing via __getitem__.

        Args:
            index (int): The index of the example to retrieve.

        Raises:
            TypeError: If the raw_dataset does not support indexing.

        Returns:
            Dict[str, torch.Tensor]: The tokenized and truncated example at the given index.
        """
        # Ensure the raw_dataset supports indexing
        if not hasattr(self.valid_dataset, "__getitem__"):
            raise TypeError("raw_dataset must support indexing to use __getitem__.")

        example = self.valid_dataset[index]
        tokenized_example = self.tokenize_example(example)
        tokenized_example = self.truncate(tokenized_example)
        return tokenized_example


def add_train_arguments(parser: argparse.ArgumentParser):
    # Generation hparams
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/Qwen/Qwen2-0.5B-Instruct/model.pth",
        help="Model checkpoint path.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "linear"],
        help="Model type for training.",
    )

    parser.add_argument(
        "--vector_convolution",
        type=str,
        default="double_conv",
        choices=["double_conv", "single_conv", "none"],
        help="Method for compressing vectors via convolutional layer.",
    )

    parser.add_argument(
        "--convolution_features",
        type=str,
        nargs="+",
        default=["embedding"],
        choices=["key", "value", "query", "embedding"],
        help="Which features to compress using the convolutional layer.",
    )

    parser.add_argument(
        "--feature_selection",
        type=str,
        nargs="+",
        default=[
            "attn_score",
            "vector_norm",
            # # "vector_cv",
            # # "vector_z_score",
            # "token_profiling",
            # "convolution",
            # "normalized_pos",
        ],
        choices=[
            "attn_score",
            "vector_norm",
            "vector_cv",
            "vector_z_score",
            "token_profiling",
            "convolution",
            "normalized_pos",
        ],
        help="Feature selection for lightweight model. Options: attn_score (attension score), vector_norm (l2 norm), vector_cv (coefficient of variation), vector_z_score (z-score), token_profiling (boolean for specials and punctuation tokens), convolution (selectable with --vector_convolution, adjustable to key, value, query, embedding).",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset path.",
    )

    parser.add_argument(
        "--train_ratio",
        type=int,
        default=0.9,
        help="Train ratio for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generators.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        choices=[1],
        help="Batch size for training. Currently only supports batch size of 1.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,  # or 16
        help="Number of gradient accumulation steps.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,  # 1e-3 -- e-6 --> see which gives best convergence
        help="Learning rate for training.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training.",
    )

    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="Ignore index for loss computation.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of examples to sample for evaluation. Defaults to -1, which uses the full dataset.",
    )

    parser.add_argument(
        "--cache_length_pattern",
        type=str,
        default="tile",
        help="Cache length pattern.",
    )

    parser.add_argument(
        "--global_tokens",
        type=int,
        default=4,
        help="Number of global tokens.",
    )

    parser.add_argument(
        "--recent_window",
        type=int,
        default=10,
        help="Recent window size.",
    )

    parser.add_argument(
        "--evaluate",
        default=False,
        action="store_true",
        help="If set to true, the evaluation script is called and result artifacts are saved in the same wandb run.",
    )


def prepare_dataset(
    tokenizer: nn.Module, args: argparse.Namespace
) -> Tuple[DataLoader, DataLoader]:
    # Load dataset
    ultrachat_datadic = load_dataset(args.dataset_path)
    len_ultrachat = len(ultrachat_datadic["train_gen"])  # full set

    # If num_samples is -1, use the full set; otherwise, use the specified number
    if args.num_samples == -1:
        args.num_samples = len_ultrachat

    indices = np.arange(len_ultrachat)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    selected_indices = indices[: args.num_samples]

    split_point = int(args.train_ratio * args.num_samples)
    train_indices = selected_indices[:split_point]
    val_indices = selected_indices[split_point:]

    train_samples = ultrachat_datadic["train_gen"].select(train_indices.tolist())
    val_samples = ultrachat_datadic["train_gen"].select(val_indices.tolist())

    train_dataset = PromptIterableDataset(train_samples, tokenizer, args.max_seq_length)
    val_dataset = PromptIterableDataset(val_samples, tokenizer, args.max_seq_length)

    actual_train_len = len(train_dataset)
    actual_val_len = len(val_dataset)
    total_actual = actual_train_len + actual_val_len

    if total_actual < args.num_samples:
        warnings.warn(
            f"⚠️ Expected {args.num_samples} samples, but filtering of invalid training examples due to shorter sequence length left only {total_actual}."
            " Consider increasing `args.max_seq_length`."
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    return train_loader, val_loader


def main(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint_path

    # Init wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_account = os.getenv("WANDB_ACCOUNT")
    wandb_model_registry = os.getenv("WANDB_MODEL_REGISTRY")

    # Validate that all required environment variables are present
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable is not set.")
    if not wandb_project:
        raise ValueError("WANDB_PROJECT environment variable is not set.")
    if not wandb_entity:
        raise ValueError("WANDB_ENTITY environment variable is not set.")
    if not wandb_account:
        raise ValueError("WANDB_ACCOUNT environment variable is not set.")
    if not wandb_model_registry:
        raise ValueError("WANDB_MODEL_REGISTRY environment variable is not set.")

    wandb.login(key=wandb_api_key)
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=vars(args),
        group="training lightweight",
    )
    registry_path = f"{wandb_account}/wandb-registry-model/{wandb_model_registry}"

    # Configuration
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = Path(checkpoint_path).parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        # If there's no tokenizer.model, try to load the tokenizer from the parent directory
        # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
        tokenizer_path = checkpoint_path.parent

    precision = torch.bfloat16
    is_chat = (
        "chat" in str(checkpoint_path).lower()
        or "instruct" in str(checkpoint_path).lower()
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device={device}")
    # distributed training
    rank = maybe_init_dist()
    use_tp = rank is not None
    # if use_tp:
    #     if rank != 0:
    #         # only print on rank 0
    #         print = lambda *args, **kwargs: None

    print("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision, use_tp)
    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    cache_kwargs = {
        "checkpoint_path": checkpoint_path,
        "compile": False,
        "device": device,
        "max_cache_length": [1.0],
        "cache_bits": None,  # dont quantize cache
        "cache_length_pattern": args.cache_length_pattern,
        "cache_strategy": ["lightweight"],
        "cache_strategy_pattern": "repeat",
        "feed_long_prompts": False,  # rather truncate incoming prompts
        "prompt_compression_strategy": [
            "full"
        ],  # we are not compressing during training
        "global_tokens": args.global_tokens,
        "recent_window": args.recent_window,
        "model_type": args.model_type,
        "trained_weights": "none",  # train new weights
        "vector_convolution": args.vector_convolution,
        "convolution_features": args.convolution_features,
        "feature_selection": args.feature_selection,
    }
    wandb.config.update(cache_kwargs)

    setup_caches(model, tokenizer, device, args.max_seq_length, cache_kwargs)

    train_loader, val_loader = prepare_dataset(tokenizer, args)

    # count trainable parameters --> less then 10 %
    for name, param in model.named_parameters():
        if "attention.kv_cache" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    num_frozen_params = num_total_params - num_trainable_params

    # trainable_perc_total = num_trainable_params / num_total_params
    trainable_perc_model = float(num_trainable_params) / float(num_frozen_params)
    print(
        f"Percentage of trainable parameters in relation to model size: {trainable_perc_model:.6f}"
    )
    print(f"Total number of trainable parameters: {num_trainable_params:.2f}")

    # Optimizer
    optimizer = AdamW(model.get_lightweight_params(), lr=args.learning_rate)
    loss = nn.CrossEntropyLoss(
        ignore_index=args.ignore_index, reduction="mean"
    )  # reduction is set to 'mean', thus the loss is automatically normalized by the number of tokens that are not ignored.

    # Training Loop
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print("Starting training...")
    model.train()
    model.set_train_mode(True)

    best_val_loss = float("inf")
    best_epoch = 0
    accumulated_loss = 0

    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            # Move batch data to device
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            labels = labels[:, 1:]  # Remove the first token
            labels = labels.reshape(-1)

            # Input positions and prefill flags
            prompt_length = input_ids.size(1)
            input_pos = torch.arange(0, prompt_length, device=device)

            causal_mask = (
                torch.tril(torch.ones(len(input_pos), len(input_pos), dtype=torch.bool))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(input_ids.device)
            )

            # Forward pass
            torch.autograd.set_detect_anomaly(True)
            logits = model(input_ids, input_pos, mask=causal_mask, is_prefill=True)

            # Reshape logits and labels for loss computation
            logits = logits[:, :-1].reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
            loss_epoch = loss(logits, labels)

            # Normalize loss for gradient accumulation
            loss_epoch = loss_epoch / args.gradient_accumulation_steps
            loss_epoch.backward()
            total_loss += loss_epoch.item() * args.gradient_accumulation_steps
            accumulated_loss += loss_epoch.item() * args.gradient_accumulation_steps

            # Update parameters every accumulation_steps mini-batches or at end of epoch
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(
                train_loader
            ):
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"train_batch_loss_acc": accumulated_loss})
                accumulated_loss = 0
            reset_caches(model)

        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} completed. Average Training Loss: {avg_train_loss:.4f}"
        )
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # ---- Validation Loop ----
        model.eval()  # Switch to evaluation mode for validation
        # model.set_train_mode(False)
        total_val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            ):
                # Move batch data to device
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                labels = labels[:, 1:]  # Remove the first token
                labels = labels.reshape(-1)

                prompt_length = input_ids.size(1)
                input_pos = torch.arange(0, prompt_length, device=device)
                causal_mask = (
                    torch.tril(
                        torch.ones(len(input_pos), len(input_pos), dtype=torch.bool)
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(input_ids.device)
                )
                logits = model(input_ids, input_pos, mask=causal_mask, is_prefill=True)
                logits = logits[:, :-1].reshape(-1, logits.size(-1))
                labels = labels.reshape(-1)
                val_loss = loss(
                    logits, labels
                )  # validation loss computed based on the scaled tensors not the compressed sequence
                total_val_loss += val_loss.item()
                accumulated_loss += val_loss.item()
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(
                    val_loader
                ):
                    wandb.log({"val_batch_loss_acc": accumulated_loss})
                    accumulated_loss = 0
                reset_caches(model)

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Average Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss or (epoch == args.epochs - 1):
            best_val_loss = avg_val_loss
            save_path, temp_dir = save_lightweight_temp(
                model, cache_kwargs
            )  # override weights each epoch for the best val loss
            parent_model_name = Path(checkpoint_path).parent.name
            final = False

            if epoch == args.epochs - 1:
                # Save the final model to the model registry
                final = True
                print(
                    f"✅ Final model saved temporally at epoch {epoch + 1} with val loss {avg_val_loss:.4f}"
                )
            else:
                print(
                    f"✅ New best model saved temporally at epoch {epoch + 1} with val loss {avg_val_loss:.4f}"
                )

            artifact = wandb.Artifact(
                name="lightweight_model",
                type="model",
                metadata={
                    "epoch": epoch + 1,
                    "validation_loss": avg_val_loss,
                    "model_type": args.model_type,
                    "parent_model": parent_model_name,
                    "final": final,
                    "trainable_params": num_trainable_params,
                    "total_params_model": num_frozen_params,
                    "trainable_perc_model": trainable_perc_model,
                },
            )  # save one file per artifact
            artifact.add_file(save_path)
            run.link_artifact(
                artifact=artifact,
                target_path=registry_path,
            )

        model.train()  # Switch back to training mode
        # model.set_train_mode(True)

    wandb.finish()

    if args.evaluate:
        eval_command = [
            "python",
            "eval.py",
            "--tasks",
            "ultrachat",
            "--use_wandb",
            "--checkpoint_path",
            str(args.checkpoint_path),
            # "--compile",
            # "--cache_length_pattern", args.cache_length_pattern," # default "tile"
            "--cache_strategy",
            "lightweight",
            "--cache_strategy_pattern",
            "repeat",
            "--prompt_compression_strategy",
            "lightweight",
            "--global_tokens",
            str(args.global_tokens),
            "--recent_window",
            str(args.recent_window),
            "--model_type",
            str(args.model_type),
            "--vector_convolution",
            str(args.vector_convolution),
            "--convolution_features",
            *args.convolution_features,
            "--feature_selection",
            *args.feature_selection,
            "--trained_weights",
            "none",
        ]
        print(f"Executing evaluation script with arguments: {eval_command}")
        subprocess.run(eval_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for lightweight KV-Cache Compression Algorithms."
    )

    add_train_arguments(parser)

    main(args=parser.parse_args())
