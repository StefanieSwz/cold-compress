import math
import operator
import tempfile
from pathlib import Path
import pdb
from typing import Any, Dict, List, Union
import argparse
from abc import ABC, abstractmethod
from collections import Counter
import regex as re
import torch
import torch.nn as nn
import wandb

from prompt_compression import get_prompt_compressor_constructor
from quantization_utils import quantize_tensor, dequantize_tensor


def add_cache_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("cache_args")
    # KV-Cache Kwargs
    group.add_argument(
        "--max_cache_length",
        type=float,
        default=[1.0],
        nargs="+",
        help="Cache size per layer. If len < n layers, the values are tiled. Must have len divisible by n layers. \
        If 0 < x <= 1, it is percent of |prompt| + max new tokens. Otherwise, if > 1, its the maximum size.",
    )

    group.add_argument(
        "--cache_bits",
        default=None,
        type=int,
        choices=[2, 4, 8],
        help="Quantize the cache to reduce memory usage.",
    )

    # ScissorHands (https://arxiv.org/abs/2305.17118) recommends large caches at higher levels --> funnel
    # Yet PyramidKV (https://arxiv.org/abs/2406.02069) recommends the opposite --> pyramid shaped
    group.add_argument(
        "--cache_length_pattern",
        default="tile",
        choices=["tile", "repeat", "funnel", "pyramid"],
    )

    strategies = [
        "full",
        "random",
        "recent_global",
        "heavy_hitter",
        "l2",
        "hybrid",
        "keep_it_odd",
        "lightweight",
    ]
    debug_strategies = [f"debug_{strategy}" for strategy in strategies]
    strategies.extend(debug_strategies)

    group.add_argument(
        "--cache_strategy",
        default=["full"],
        nargs="+",
        choices=strategies,
    )

    group.add_argument(
        "--cache_strategy_pattern",
        default="tile",
        choices=["tile", "repeat"],
        help="How to apply the cache_strategy across layers.",
    )

    # Dealing with Long Prompts
    parser.add_argument(
        "--feed_long_prompts",
        default=False,
        action="store_true",
        help="If True and |prompt| > max_cache_length, prefill with prompt[:max_cache_length], and feed prompt[max_cache_length:] sequentially.",
    )
    group.add_argument(
        "--prompt_compression_strategy",  # This doesn't matter if args.feed_long_prompts is True
        default=["recent_global"],
        nargs="+",
        help="If |prompt| exceeds max_cache_length, we need to specify a strategy for compressing it to max_cache_length.",
    )

    # Optional Cache Kwargs depending on cache_strategy
    group.add_argument(
        "--global_tokens",
        default=1,
        type=int,
        help="The number of initial tokens to always include in the KV-Cache.  \
        If using recent_global strategy, the actual window size becomes max_cache_length - global_tokens.",
    )

    # Locality
    group.add_argument(
        "--recent_window",  # NB: for KVCacheRecentGlobal, recent_window is implicitly set to self.max_cache_length - self.global_tokens.
        default=10,  # 10 is default specified in ScissorHands paper ("r" in Algorithm 2).
        type=float,  # If < 1, it is a fraction of max_cache_length.
        help="The number of recently generated tokens to always spare from eviction.",
    )

    # Heavy Hitter Hyperparameters (--cache_strategy == "heavy_hitter")
    group.add_argument(  ## See Algorithm 2 in ScissorHands arxiv.org/abs/2305.17118
        "--history_window_size",  # Equivalent to "m" in Algorithm 2. 400 is default specified in paper.
        default=1,  # If 1, we accumulate the full history in one slot (effectively, a history_window_size of ∞)
        type=int,
        help="The number of past tokens to consider when computing 'Heavy Hitters' in the KV-Cache.",
    )
    group.add_argument(
        "--attn_thresholding",
        default=False,
        action="store_true",
        help="Whether to accumulate number of times a token was unimportant (binary) versus raw un-normalized probabilities. If true, more memory efficient.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Lightweight model type: 'linear' or 'mlp'.",
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
        default=["attn_score", "vector_norm"],
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
        "--trained_weights",
        type=str,
        default=False,
        help="Path to lightweight model weights: Typically stored in ./lightweight_models.",
    )

    # Hybrid, e.g., FastGen, specific hyperparameters (--cache_strategy == "hybrid")
    parser.add_argument(
        "--min_recovery_frac",
        default=0.9,
        type=float,
        help="Mininum fraction of recovered attentions (|compressed_attn - uncompressed_attn| < epsilon). The lower the value, the higher the compression.",
    )


def cache_compatibility(args):
    for length, cache_strat, prompt_strat in zip(
        args.max_cache_length, args.cache_strategy, args.prompt_compression_strategy
    ):
        if cache_strat == "heavy_hitter":
            assert (
                prompt_strat == "heavy_hitter"
            ), "Heavy Hitter cache strategy currently must be run with --prompt_compression_strategy heavy_hitter to return attention."
        if cache_strat == "hybrid":
            assert (
                not args.compile
            ), "Hybrid cache strategy is currently not supported with compile=True."

        if cache_strat in {"full", "hybrid"}:
            assert (
                length == 1.0
            ), f"{cache_strat} cache strategy only supports max_cache_length=1.0."

    print("The cache argument values you provided appear compatible with each other!")


def create_window_attention_mask(seq_len, window_size, device, global_tokens: int = 4):
    # Initialize the mask tensor with zeros
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    # Add global tokens
    mask[:, :global_tokens] = True
    for i in range(seq_len):
        mask[i, max(0, i + 1 - window_size) : i + 1] = True
    return mask


def save_lightweight(model: nn.Module, kwargs: Dict[str, Any], identifier: str) -> Path:
    """
    Save lightweight model parameters to disk.

    This function extracts the parameters from the model that require gradients,
    and saves them to a file in a folder structured by the model name and type.
    The filename is built using the provided identifier and model type.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be saved.
        kwargs (Dict[str, Any]): A dictionary that may contain:
            - "checkpoint_path": A path or string to determine the model name.
            - "model_type": A string representing the type of model.
        identifier (str): A unique identifier to include in the saved filename.

    Returns:
        Path: The path to the saved weights file.
    """
    checkpoint_path = kwargs.get("checkpoint_path", "default_model")
    model_name = Path(checkpoint_path).parent.name
    model_type = kwargs.get("model_type", "unknown")

    save_dir = Path(f"./lightweight_weights/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{identifier}_{model_type}.pth"

    trained_weights = {
        name: param
        for name, param in model.state_dict().items()
        if name in [n for n, p in model.named_parameters() if p.requires_grad]
    }
    torch.save(trained_weights, save_path)
    print(f"Trained weights saved to {save_path}")

    return save_path


def load_trained_lightweight(
    model: nn.Module, checkpoint_path: Union[str, Path], load_kv: bool = True
) -> None:
    """
    Load trained weights into a model, updating only matching layers for the specified target.

    This function loads a checkpoint and updates the model's parameters with
    those weights that match keys in the model's state dictionary. Only weights
    for the specified target are loaded.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (Union[str, Path]): Path to the saved weights.
        load_kv (bool): Which target to load weights for. Options:
                           True for KV-Cache or "prompt_compressor".

    Returns:
        None
    """
    trained_weights: Dict[str, torch.Tensor] = torch.load(
        checkpoint_path, map_location="cpu"
    )
    model_state: Dict[str, torch.Tensor] = model.state_dict()

    # Determine which substring to look for based on the target.
    if load_kv:
        target_substring = "attention.kv_cache.models"
    elif not load_kv:
        target_substring = "attention.prompt_compressor.models"
    else:
        raise ValueError("load_target must be either 'kv_cache' or 'prompt_compressor'")

    # Filter weights: update only those keys that are both in the model state
    # and correspond to the desired target.
    matched_weights = {
        k: v
        for k, v in trained_weights.items()
        if k in model_state and target_substring in k
    }
    unmatched_weights = set(trained_weights.keys()) - set(matched_weights.keys())
    assert (
        not unmatched_weights
    ), f"The following weights were not found in the model for {target_substring}: {unmatched_weights}"

    # Update and load state.
    model_state.update(matched_weights)
    model.load_state_dict(model_state)
    print(f"Loaded trained weights for {target_substring} from {checkpoint_path}")


def load_best_lightweight_model_wb(
    model,  # nn.Module instance to update
    entity,
    project,
    required_metadata,
    metric_name="validation_loss",
    load_kv=True,
    higher_is_better=False,
):
    """
    Searches the W&B registry for model artifacts matching the required metadata,
    selects the best one based on the given metric, downloads it, and loads its
    weights into the provided model using load_trained_lightweight.

    Parameters:
      - model (nn.Module): The model to load weights into.
      - entity (str): W&B username or team.
      - project (str): W&B project name.
      - required_metadata (dict): Key/value pairs that must be present in the artifact's metadata.
      - metric_name (str): Metric key to compare (default "validation_loss").
      - load_kv (bool): Passed to load_trained_lightweight to select which weights to load.
      - higher_is_better (bool): If True, higher metric is better; otherwise, lower is better.

    Returns:
      The model updated with the best artifact's weights (or the result of load_model_fn).

    Raises:
      Exception if no matching artifact is found.
    """
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}")

    best_artifact = None
    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt

    for run in runs:
        try:
            for artifact in run.logged_artifacts():
                if artifact.type != "model":
                    continue
                meta = artifact.metadata or {}
                # Skip if any required metadata key is missing or mismatched.
                if not all(
                    meta.get(key) == value for key, value in required_metadata.items()
                ):
                    continue
                if metric_name not in meta:
                    continue
                metric_value = meta[metric_name]
                if compare_op(metric_value, best_metric):
                    best_metric = metric_value
                    best_artifact = artifact
        except wandb.errors.CommError as e:
            print(f"Error accessing artifacts for run {run.id}: {e}")
        except Exception as e:
            print(f"Unexpected error accessing artifacts for run {run.id}: {e}")

    if best_artifact is None:
        raise Exception("No matching artifact found in the registry.")

    print(f"Best model found: {best_artifact.name} with {metric_name} = {best_metric}")

    # Download the artifact to a temporary directory
    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact.download(root=artifact_dir)

        checkpoint_path = Path(artifact_dir) / "model.pth"
        load_trained_lightweight(model, checkpoint_path, load_kv=load_kv)
        print(f"Loaded weights from {checkpoint_path} into model.")


class KVCache(ABC, nn.Module):
    # Define which hyperparameters are relevant for the cache.
    # Override as needed for sub-classes.
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "max_seq_length",
        "cache_bits",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=False,  # IFF True, heads can contain different tokens, e.g., cache evictions are "head_specific".
        variable_length=False,  # IFF True, the number of tokens inserted can vary across heads. Only true for KVCacheHybrid.
        **kwargs,
    ):
        super().__init__()

        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.cache_shape = (max_batch_size, n_heads, self.max_cache_length, head_dim)

        # Quantization: 2, 4, 8 bits supported.
        self.quantize = self.cache_bits is not None
        self.n_bit = self.cache_bits
        self.quantization_axis = 2  # Quantize the cache along the sequence length axis

        k_cache = torch.zeros(self.cache_shape, dtype=dtype)
        v_cache = torch.zeros(self.cache_shape, dtype=dtype)

        if self.quantize:
            k_cache, k_scales, k_zeros = quantize_tensor(
                k_cache, n_bit=self.n_bit, axis=self.quantization_axis
            )
            v_cache, v_scales, v_zeros = quantize_tensor(
                v_cache, n_bit=self.n_bit, axis=self.quantization_axis
            )
            self.register_buffer("k_scales", k_scales)
            self.register_buffer("v_scales", v_scales)
            self.register_buffer("k_zero_points", k_zeros)
            self.register_buffer("v_zero_points", v_zeros)

        self.register_buffer("k_cache", k_cache)
        self.register_buffer("v_cache", v_cache)

        # Can we evict different tokens for different heads?
        # If the answer is yes, we need to store self.pos for each head.
        self.n_heads = n_heads
        self.head_specific = head_specific
        self.register_buffer(
            "pos",  # Track pos to keep track of the original positions of each item in cache.
            torch.full(
                (
                    max_batch_size,
                    n_heads if head_specific else 1,
                    self.max_cache_length,
                ),
                -1,
                dtype=torch.int,
            ),
        )
        self.register_buffer(
            "cache_cts",
            torch.zeros((n_heads if variable_length else 1), dtype=torch.int),
        )

        # We need to use a mask since not all heads have same number of tokens. We can't simply truncate.
        # 1 dimension stands for query dimension, which will always be 1 (next token) for KV cache attention.
        kv_mask_shape = (max_batch_size, n_heads, 1, self.max_cache_length)
        self.register_buffer("mask", torch.zeros(kv_mask_shape, dtype=torch.bool))

    def reset(self):
        """
        Resets the cache to its initial state for a new example.

        NB: For more performance, don't reset k_cache and v_cache since we overwrite them in update.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.mask.zero_()
        self.cache_cts.zero_()
        self.pos.fill_(-1)

    def return_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def memory_usage(self):
        tensors = []
        for obj in vars(self).values():
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif isinstance(obj, dict):
                for vv in obj.values():
                    if torch.is_tensor(vv):
                        tensors.append(vv)

        return sum([t.element_size() * t.numel() for t in tensors]) / (1024**3)

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        return {
            "compression_ratio": self.compression_ratio(seq_len).item(),
            "cache_memory_gb": self.memory_usage(),
        }

    def compression_ratio(self, seq_len):
        """
        Returns the compression ratio of the cache.
        """
        # Final token isn't passed to cache so must -1 from seq_len
        n = seq_len - 1
        assert torch.all(self.cache_cts <= self.max_cache_length)
        cache_size = self.cache_cts.clone().float()
        if self.n_bit is not None:
            cache_size *= self.n_bit / 16.0
        return ((n - cache_size) / n).mean()

    def quantize_cache(self):
        if self.quantize:
            self.k_cache, self.k_scales, self.k_zero_points = quantize_tensor(
                self.k_cache, n_bit=self.n_bit, axis=self.quantization_axis
            )
            self.v_cache, self.v_scales, self.v_zero_points = quantize_tensor(
                self.v_cache, n_bit=self.n_bit, axis=self.quantization_axis
            )

    def dequantize_cache(self):
        if self.quantize:
            self.k_cache = dequantize_tensor(
                self.k_cache,
                self.k_scales,
                self.k_zero_points,
                self.cache_shape,
                n_bit=self.n_bit,
                axis=self.quantization_axis,
            )
            self.v_cache = dequantize_tensor(
                self.v_cache,
                self.v_scales,
                self.v_zero_points,
                self.cache_shape,
                n_bit=self.n_bit,
                axis=self.quantization_axis,
            )

    def return_kv_cache(self):
        return self.k_cache, self.v_cache, self.mask

    def update_kv(self, input_pos, k_val, v_val, is_prefill, **kwargs):
        """
        Cache update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.

        Returns a tensor indicating the number of tokens inserted - number of tokens evicted.
        None is equivalent to 0.
        """
        # Dequantize the cache before updating
        self.dequantize_cache()

        if is_prefill:
            num_insertions = self._prefill_update(input_pos, k_val, v_val, **kwargs)
        else:
            num_insertions = self._decoding_update(input_pos, k_val, v_val, **kwargs)
        self.cache_cts += num_insertions[: len(self.cache_cts)]

        # [Optional] Update any internal model state
        k, v, mask = (
            self.return_kv_cache()
        )  # By default, just returns self.k_cache, self.v_cache, self.mask

        # Quantize the cache after updating
        self.quantize_cache()
        return k, v, mask

    def update_state(self, *args, **kwargs):
        """
        Optional method to update cache-specific internal state (excludes self.k_cache, self.v_cache, and self.pos).
        """
        pass

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        """
        Decoding logic for the cache.
        """
        eviction_idx = self._eviction_idx(input_pos)

        # Num insertions means we inserted into an unfilled slot (previous pos == -1)
        # They should be all the same unless variable_length = True
        num_insertions = (
            (self.pos.gather(2, eviction_idx.view(1, -1, 1)).squeeze() == -1)
            .int()
            .view(-1)
        )

        self._fill(input_pos, k_val, v_val, fill_idxs=eviction_idx)

        return num_insertions

    def _eviction_idx(self, input_pos):
        scores = self._token_importances(input_pos)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)

        # Protect global tokens
        scores[:, : self.global_tokens] = float("inf")

        # Evict unfilled slots (pos == -1)
        scores.masked_fill_(self.pos.view(scores.shape) == -1, float("-inf"))

        # Evict least important token
        return torch.argmin(scores, dim=-1)

    def _prefill_update(self, input_pos, k_val, v_val, **kwargs):
        input_pos = input_pos.int()
        fill_idxs = torch.arange(input_pos.shape[-1], device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs=fill_idxs)
        # Saves a fraction of time to return as a tensor rather than integer
        return torch.tensor(
            [input_pos.shape[-1]], dtype=torch.int, device=input_pos.device
        )

    def _fill_contiguous(
        self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs
    ):
        """
        A simple utility to fill the cache and pos.
        """
        self.pos[:, :, fill_idxs] = input_pos
        self.k_cache[:, :, fill_idxs, :] = k_val
        self.v_cache[:, :, fill_idxs, :] = v_val
        update_mask = kwargs.get("update_mask", True)
        if update_mask:
            self.mask[:, :, :, fill_idxs] = True

    @abstractmethod
    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        raise NotImplementedError

    def update_attn_history(self, attn):
        """
        Update the attention history with the most recent attention weights.
        """
        raise Exception(
            f"{self.__class__.__name__} requested return_attn=True but has not yet implemented a update_attn_history function."
        )


class KVCacheHeadConstant(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
        return self._fill_contiguous(input_pos, k_val, v_val, fill_idxs, **kwargs)


class KVCacheHeadSpecific(KVCache):
    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            head_specific=True,
            variable_length=variable_length,
            **kwargs,
        )

    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        # fill_indices [num_heads] or [1]
        # input_pos [seq_len] or [num_heads, seq_len]
        # k_val, v_val [batch_size, n_heads, seq_len, head_dim]
        assert input_pos.shape[-1] == k_val.shape[2] == v_val.shape[2]
        # input_pos is either [seq_len] or [num_heads, seq_len]
        pos_fill_indices = fill_idxs.view(1, -1, 1)
        cache_fill_indices = fill_idxs.view(1, len(fill_idxs), 1, 1).expand(
            1, k_val.shape[1], 1, k_val.shape[-1]
        )
        input_pos = input_pos.view(1, -1, 1).expand(1, k_val.shape[1], 1).int()
        self.pos.scatter_(2, pos_fill_indices, input_pos.int())
        self.k_cache.scatter_(2, cache_fill_indices, k_val)
        self.v_cache.scatter_(2, cache_fill_indices, v_val)

        update_mask = kwargs.get("update_mask", True)
        if update_mask:
            self.mask.scatter_(3, fill_idxs.view(1, -1, 1, 1), True)


class KVCacheFull(KVCacheHeadConstant):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        self.global_tokens = 0  # No global tokens for full cache (they are all global)
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _eviction_idx(self, input_pos):
        # Select the first unfilled slot
        return self.pos[0, 0].argmin().view(1)


class KVCacheRandom(KVCacheHeadConstant):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        "recent_window",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _token_importances(self, input_pos):
        # Assign random importance
        scores = torch.rand(self.max_cache_length, device=input_pos.device)
        # Protect Recent Tokens
        scores[self.pos[0, 0] >= input_pos - self.recent_window] = float("inf")
        return scores


class KVCacheRecentGlobal(KVCacheHeadConstant):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        # NB: "recent_window" is ignored as a relevant kwarg. It is fixed to self.max_cache_length - self.global_tokens.
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            **kwargs,
        )

    def _eviction_idx(self, input_pos):
        return (
            torch.argmin(self.pos[:, :, self.global_tokens :], dim=-1)
            + self.global_tokens
        ).view(1)


class KVCacheL2(KVCacheHeadSpecific):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        "recent_window",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

        key_norm_shape = (max_batch_size, n_heads, self.max_cache_length)
        self.register_buffer("key_norm", torch.zeros(key_norm_shape, dtype=dtype))

    def reset(self):
        super().reset()
        self.key_norm.zero_()

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        # Same as KVCacheHeadSpecific, but we also update the L2 norm of the keys for decoding
        fill_indices = self._eviction_idx(input_pos)
        num_insertions = (
            (self.pos.gather(2, fill_indices.view(1, -1, 1)).squeeze() == -1)
            .int()
            .view(-1)
        )

        self._fill(input_pos, k_val, v_val, fill_idxs=fill_indices)

        # Custom code for L2 -- store the key vector norms
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)
        self.key_norm.scatter_(2, fill_indices.view(1, -1, 1), key_norm)

        return num_insertions

    def _token_importances(self, input_pos):
        # 1. Lowest l2 norms have high importance (- self.key_norm)
        # 2. Lowest score needs to be > -1 : we evict unfilled tokens first (+ max value such that min score is 0)
        # 3. Save Recent Window (+ inf)
        return (
            (self.key_norm.max() - self.key_norm)
            .masked_fill(self.pos >= input_pos - self.recent_window, float("inf"))
            .squeeze(0)
        )

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        pass
        # We will update the L2 norm of the keys for decoding in _decoding_update
        # We do this during the update bc/ we have access to the fill indices of the tokens we are inserting
        if is_prefill:  # For prefill, we cache the norm for the key cache at the time
            self.key_norm.copy_(torch.linalg.vector_norm(self.k_cache, ord=2, dim=-1))


class KVCacheLightweight(KVCacheHeadSpecific):
    """
    Lightweight Key-Value Cache for Transformer Models.

    This class implements a cache compression mechanism based on lightweight
    scoring models (either a linear model or an MLP) to compute token importance.
    It follows similar ideas to methods like ScissorHands and other heavy-hitter
    approaches as well as the L2 Norm based approach. The cache stores key norms,
    value norms, and attention scores, and it provides methods to update and
    compute importance scores for eviction.

    Attributes:
        relevant_kwargs (List[str]): A list of keyword arguments that are relevant for cache configuration.
    """

    relevant_kwargs: List[str] = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "model_config",
        "global_tokens",
        "recent_window",
        "model_type",
        "feature_selection",
        "vector_convolution",
        "convolution_features",
        "token_ids",
        "trained_weights",
    ]

    def __init__(
        self,
        max_batch_size: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        variable_length: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the lightweight KV cache.

        Args:
            max_batch_size (int): Maximum batch size.
            n_heads (int): Number of attention heads.
            head_dim (int): Dimension of each head.
            dtype (torch.dtype, optional): Data type for buffers and models. Defaults to torch.bfloat16.
            variable_length (bool, optional): Whether the cache supports variable sequence lengths. Defaults to False.
            **kwargs: Additional configuration arguments. Should include keys such as "model_type" and "trained_weights".
        """
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, variable_length, **kwargs
        )
        self.feature_space_dim = 0  # Number of final features

        # Initialize buffers for features.
        # norm is l2 norm, cv is coefficient of variation (sd/mean), z is z-score (x_max - mean / sd)
        score_shape = (max_batch_size, n_heads, self.max_cache_length)
        if "attn_score" in kwargs["feature_selection"]:
            self.register_buffer("attn_score", torch.zeros(score_shape, dtype=dtype))
            self.feature_space_dim += 1
        if "vector_norm" in kwargs["feature_selection"]:
            self.register_buffer("key_norm", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("value_norm", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("query_norm", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer(
                "embedding_norm", torch.zeros(score_shape, dtype=dtype)
            )
            self.feature_space_dim += 4
        if "vector_cv" in kwargs["feature_selection"]:
            self.register_buffer("key_cv", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("value_cv", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("query_cv", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("embedding_cv", torch.zeros(score_shape, dtype=dtype))
            self.feature_space_dim += 4
        if "vector_z_score" in kwargs["feature_selection"]:
            self.register_buffer("key_z", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("value_z", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("query_z", torch.zeros(score_shape, dtype=dtype))
            self.register_buffer("embedding_z", torch.zeros(score_shape, dtype=dtype))
            self.feature_space_dim += 4
        if "token_profiling" in kwargs["feature_selection"]:
            special_ids_tensor = torch.tensor(
                kwargs["token_ids"]["special"], dtype=torch.int32
            )
            self.register_buffer("special_ids", special_ids_tensor)

            punctuation_ids_tensor = torch.tensor(
                kwargs["token_ids"]["punctuation"], dtype=torch.int32
            )
            self.register_buffer("punctuation_ids", punctuation_ids_tensor)
            self.register_buffer(
                "token_special_profiling", torch.zeros(score_shape, dtype=dtype)
            )
            self.register_buffer(
                "token_punctuation_profiling", torch.zeros(score_shape, dtype=dtype)
            )
            self.feature_space_dim += 2
        if "convolution" in kwargs["feature_selection"]:
            self.conv_compression_rate = 16  # must be 2^x, keep at 16, otherwise double conv does not necessarily work
            self.conv_hidden_channels = 4  # or 8
            self.conv_layers = nn.ModuleDict()

            # kwargs["convolution_features"] is a list like ["key", "value", "query", "embedding"]
            for feat in kwargs["convolution_features"]:
                if feat == "embedding":
                    result_conv_dim = (
                        kwargs["model_config"].dim // self.conv_compression_rate
                    )
                else:
                    result_conv_dim = head_dim // self.conv_compression_rate
                if kwargs["vector_convolution"] == "double_conv":
                    self.conv_layers[feat] = nn.Sequential(
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=self.conv_hidden_channels,
                            kernel_size=int(math.sqrt(self.conv_compression_rate)),
                            stride=int(math.sqrt(self.conv_compression_rate)),
                        ).to(torch.bfloat16),
                        nn.ReLU(),
                        nn.Conv1d(
                            in_channels=self.conv_hidden_channels,
                            out_channels=1,
                            kernel_size=int(math.sqrt(self.conv_compression_rate)),
                            stride=int(math.sqrt(self.conv_compression_rate)),
                        ).to(torch.bfloat16),
                    )
                    buffer_shape = (
                        max_batch_size,
                        n_heads,
                        self.max_cache_length,
                        result_conv_dim,
                    )
                    self.register_buffer(
                        f"{feat}_conv_values", torch.zeros(buffer_shape, dtype=dtype)
                    )
                    self.feature_space_dim += result_conv_dim

                elif kwargs["embedding_compression"] == "single_conv":
                    self.conv_layers[feat] = nn.Sequential(
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=self.conv_hidden_channels,
                            kernel_size=self.conv_compression_rate
                            * self.conv_hidden_channels,
                            stride=self.conv_compression_rate
                            * self.conv_hidden_channels,
                        ).to(torch.bfloat16),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    buffer_shape = (
                        max_batch_size,
                        n_heads,
                        self.max_cache_length,
                        result_conv_dim,
                    )
                    self.register_buffer(
                        f"{feat}_conv_values", torch.zeros(buffer_shape, dtype=dtype)
                    )
                    self.feature_space_dim += result_conv_dim
                else:
                    print("No convolution feature initialized")
        if "normalized_pos" in kwargs["feature_selection"]:
            # no buffer needed as computed directly in token_importance
            self.feature_space_dim += 1
        # Initialize lightweight models for computing token scores.
        if kwargs["model_type"] == "linear":
            self.models = nn.ModuleList(
                [nn.Linear(self.feature_space_dim, 1).to(dtype) for _ in range(n_heads)]
            )
        elif kwargs["model_type"] == "mlp":
            self.lightweight_hidden_size = 16
            self.models = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_space_dim, self.lightweight_hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.lightweight_hidden_size, 1),
                    ).to(dtype)
                    for _ in range(n_heads)
                ]
            )
        else:
            raise ValueError("Unsupported model_type. Use 'linear' or 'mlp'.")

        def init_weights(m):
            # You can choose a specific initialization for Conv1d and Linear layers
            # if isinstance(m, (nn.Conv1d, nn.Linear)):
            #     # For example, Kaiming normal initialization for layers with ReLU activation:
            #     torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #     if m.bias is not None:
            #         torch.nn.init.constant_(m.bias, 0)
            # Alternatively, if you prefer a normal distribution with mean=0 and std=0.02:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.02)

        # Initialize the models' parameters with random weights.
        # Trained weights are loaded from model perspective
        with torch.no_grad():
            if "convolution" in kwargs["feature_selection"]:
                for feat, conv_block in self.conv_layers.items():
                    conv_block.apply(init_weights)
            for model in self.models:
                model.apply(init_weights)

    def reset(self) -> None:
        """
        Reset the cache buffers.

        This clears all feature buffer.
        """
        super().reset()
        attrs_to_zero = [
            "attn_score",
            "key_norm",
            "value_norm",
            "query_norm",
            "embedding_norm",
            "key_cv",
            "value_cv",
            "query_cv",
            "embedding_cv",
            "key_z",
            "value_z",
            "query_z",
            "embedding_z",
            "special_ids",
            "punctuation_ids",
            "token_special_profiling",
            "token_punctuation_profiling",
            "key_conv_values",
            "value_conv_values",
            "query_conv_values",
            "embedding_conv_values",
        ]

        for attr_name in attrs_to_zero:
            if hasattr(self, attr_name):
                getattr(self, attr_name).zero_()

    def return_attn(self) -> bool:
        """
        Indicate that this cache returns attention scores.

        Returns:
            bool: Always True for this cache.
        """
        return True

    def filter_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Filters vectors from x using indices provided by self.pos.
        self.pos (torch.Tensor): Tensor of shape [B, H, M]. Each element is a token index in [0, T)
                                    or -1 indicating an unfilled slot.

        Args:
            x (torch.Tensor): Tensor of shape [B, T, D] containing token embeddings.

        Returns:
            torch.Tensor: A tensor of shape [B, H, M, D] where for each (B, H, M) slot, if input_pos has a valid
                        index, the corresponding token embedding from x is gathered; otherwise, the slot is zeroed.
        """
        # Get dimensions.
        B, H, M = self.pos.shape
        _, T, D = x.shape

        # Expand x to include a head dimension.
        # x is originally [B, T, D]. We expand it to [B, 1, T, D] and then to [B, H, T, D].
        x_exp = x.unsqueeze(1).expand(B, H, T, D)

        # Create a boolean mask for valid positions (positions not equal to -1).
        valid_mask = self.pos != -1  # shape: [B, H, M]

        # Create safe indices to use with gather: replace -1 with 0.
        safe_indices = self.pos.clone()
        safe_indices[~valid_mask] = 0  # shape: [B, H, M]
        safe_indices = safe_indices.to(torch.long)

        # Prepare safe_indices for torch.gather by unsqueezing and expanding to match x_exp's D dimension.
        safe_indices = safe_indices.unsqueeze(-1).expand(
            B, H, M, D
        )  # shape: [B, H, M, D]

        # Gather the vectors along the sequence dimension (dim=2) from x_exp.
        gathered = torch.gather(
            x_exp, dim=2, index=safe_indices.to(torch.int64)
        )  # shape: [B, H, M, D]

        # Zero out the positions where input_pos was -1.
        filtered = gathered * valid_mask.unsqueeze(-1).type_as(gathered)

        return filtered

    def filter_query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Filters vectors from x using indices provided by self.pos.
                self.pos (torch.Tensor): Tensor of shape [B, H, M]. Each element is a token index in [0, T)
                                or -1 indicating an unfilled slot.

        Args:
            x (torch.Tensor): Tensor of shape [B, H, T, D] containing token embeddings.

        Returns:
            torch.Tensor: A tensor of shape [B, H, M, D] where for each (B, H, M) slot, if input_pos has a valid
                        index, the corresponding token embedding from x is gathered; otherwise, the slot is zeroed.
        """
        # Get dimensions.
        B, H, M = self.pos.shape
        _, _, T, D = x.shape

        # Create a boolean mask for valid positions (positions not equal to -1).
        valid_mask = self.pos != -1  # shape: [B, H, M]

        # Create safe indices to use with gather: replace -1 with 0.
        safe_indices = self.pos.clone()
        safe_indices[~valid_mask] = 0  # shape: [B, H, M]

        # Prepare safe_indices for torch.gather by unsqueezing and expanding to match x's D dimension.
        safe_indices = safe_indices.unsqueeze(-1).expand(
            B, H, M, D
        )  # shape: [B, H, M, D]

        # Gather the vectors along the sequence dimension (dim=2) from x.
        gathered = torch.gather(
            x, dim=2, index=safe_indices.to(torch.int64)
        )  # shape: [B, H, M, D]

        # Zero out the positions where input_pos was -1.
        filtered = gathered * valid_mask.unsqueeze(-1).type_as(gathered)

        return filtered

    def build_ids_mask(self, input_ids, token_ids):
        """
        Builds a special token mask from input_ids and expands it using self.pos.

        Args:
            input_ids (torch.Tensor): A tensor of shape [B, seq_len] containing token IDs.
            token_ids (List(torch.Tensor)): A list of ids that are belonging to either special or punctuation tokens.

        Returns:
            torch.Tensor: A boolean mask of shape [B, n_heads, max_cache_len] where True indicates that
                        the token at that cache position (as given by self.pos) is a special token.
        """
        B, seq_len = input_ids.shape
        ids_mask = torch.isin(input_ids, token_ids)  # Shape: [B, seq_len]
        # Expand to [B, n_heads, seq_len]
        mask_exp = ids_mask.unsqueeze(1).expand(B, self.n_heads, seq_len)

        # Valid mask for self.pos (-1 means unfilled slots)
        valid_mask = self.pos != -1  # Shape: [B, n_heads, max_cache_len]

        # Replace -1 with 0 to safely index into mask_exp
        safe_pos = self.pos.clone()
        safe_pos[~valid_mask] = 0  # Shape: [B, n_heads, max_cache_len]

        # Gather mask values at positions specified by self.pos
        expanded_mask = torch.gather(mask_exp, dim=2, index=safe_pos.to(torch.int64))

        # Ensure positions where self.pos was -1 remain False
        return expanded_mask & valid_mask

    def compute_cv(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the coefficient of variation of a tensor along the last dimension.

        The cv is defined as:
            cv = std / mean
        for each row along the last dimension. If the mean is zero,
        the corresponding cv-score is set to zero to avoid division by zero.

        Args:
            tensor (Tensor): Input tensor of arbitrary shape. The z-score is computed
                            along the last dimension.

        Returns:
            Tensor: A tensor of the same shape as `tensor` with the last dimension reduced,
                    containing the computed cv-scores.
        """
        mean_val = torch.mean(tensor, dim=-1)
        std_val = torch.std(tensor, dim=-1, unbiased=False)
        # Avoid division by zero:
        return torch.where(
            mean_val == 0, torch.zeros_like(mean_val), std_val / mean_val
        )

    def compute_z_score(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the z-score of a tensor along the last dimension.

        The z-score is defined as:
            z = (max - mean) / std
        for each row along the last dimension. If the standard deviation is zero,
        the corresponding z-score is set to zero to avoid division by zero.

        Args:
            tensor (Tensor): Input tensor of arbitrary shape. The z-score is computed
                            along the last dimension.

        Returns:
            Tensor: A tensor of the same shape as `tensor` with the last dimension reduced,
                    containing the computed z-scores.
        """
        max_val = torch.max(tensor, dim=-1).values
        mean_val = torch.mean(tensor, dim=-1)
        std_val = torch.std(tensor, dim=-1, unbiased=False)
        z_score = torch.where(
            std_val == 0, torch.zeros_like(std_val), (max_val - mean_val) / std_val
        )
        return z_score

    def update_state(  # pylint: disable=W0221
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        is_prefill: bool,
        attn: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Update the cache state.

        For prefill mode, this updates the entire cache (attention scores, key norms,
        and value norms) for the current prompt. In generation mode, only the newly
        generated token's information is updated.

        Args:
            input_pos (torch.Tensor): Input positions in the sequence.
            k_val (torch.Tensor): Key values with shape [batch_size, n_heads, seq_len, head_dim].
            v_val (torch.Tensor): Value values with shape [batch_size, n_heads, seq_len, head_dim].
            is_prefill (bool): True if in prefill phase; False if in generation phase.
            attn (torch.Tensor): Attention scores with shape [batch_size, n_heads, seq_len, seq_len].
            **kwargs: Additional keyword arguments.
        """
        x = kwargs[
            "x"
        ]  # x.shape = [batch_size, sequence_length, model_hidden_dim = n_query_heads * head_dim]
        q = kwargs[
            "query"
        ]  # q.shape = [batch_size, n_query_heads, sequence_length, head dim]
        batch_size, n_query_heads, seq_len, head_dim = q.shape
        # seq_len is 1 for not prefill, and complete sequence length for prefill, that can be larger then max_cache_length
        group_size = n_query_heads // self.n_heads
        q_grouped = q.view(batch_size, self.n_heads, group_size, seq_len, head_dim)
        q_avg = q_grouped.mean(dim=2)  # [batch_size, n_heads, seq_len, head_dim]
        max_cache_len = self.pos.shape[2]
        if is_prefill:
            x_filtered = self.filter_embedding(x)
            # shape = [batch_size, n_heads, max_cache_length, model_hidden_dim = n_query_heads * head_dim]
            q_filtered = self.filter_query(q_avg)
            # shape = [batch_size, n_heads, max_cache_length, head_dim]
            if "vector_norm" in self.feature_selection:
                # Calculate norms over the head dimension.
                # key and value already filtered by prompt compressor
                key_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    k_val, ord=2, dim=-1
                )  # [batch_size, n_heads, max_cache_len/prompt_len]
                value_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    v_val, ord=2, dim=-1
                )  # [batch_size, n_heads, max_cache_len/prompt_len]
                query_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    q_filtered, ord=2, dim=-1
                )  # [batch_size, n_heads, max_cache_len]
                embedding_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    x_filtered, ord=2, dim=-1
                )  # [batch_size, n_heads, max_cache_len]
                if seq_len < max_cache_len:
                    self.key_norm[:, :, :seq_len] = key_norm.detach()
                    self.value_norm[:, :, :seq_len] = value_norm.detach()
                    self.query_norm = query_norm.detach()
                    self.embedding_norm = embedding_norm.detach()
                else:
                    self.key_norm = key_norm.detach()
                    self.value_norm = value_norm.detach()
                    self.query_norm = query_norm.detach()
                    self.embedding_norm = embedding_norm.detach()
            if "vector_cv" in self.feature_selection:
                key_cv = self.compute_cv(k_val)
                value_cv = self.compute_cv(v_val)
                query_cv = self.compute_cv(q_filtered)
                embedding_cv = self.compute_cv(x_filtered)
                if seq_len < max_cache_len:
                    self.key_cv[:, :, :seq_len] = key_cv.detach()
                    self.value_cv[:, :, :seq_len] = value_cv.detach()
                    self.query_cv = query_cv.detach()
                    self.embedding_cv = embedding_cv.detach()
                else:
                    self.key_cv = key_cv.detach()
                    self.value_cv = value_cv.detach()
                    self.query_cv = query_cv.detach()
                    self.embedding_cv = embedding_cv.detach()
            if "vector_z_score" in self.feature_selection:
                key_z = self.compute_z_score(k_val)
                value_z = self.compute_z_score(v_val)
                query_z = self.compute_z_score(q_filtered)
                embedding_z = self.compute_z_score(x_filtered)
                if seq_len < max_cache_len:
                    self.key_z[:, :, :seq_len] = key_z.detach()
                    self.value_z[:, :, :seq_len] = value_z.detach()
                    self.query_z = query_z.detach()
                    self.embedding_z = embedding_z.detach()
                else:
                    self.key_z = key_z.detach()
                    self.value_z = value_z.detach()
                    self.query_z = query_z.detach()
                    self.embedding_z = embedding_z.detach()
            if "attn_score" in self.feature_selection:
                if attn.ndim == 4:
                    seq_len_actual = attn.shape[-1]
                    attn = attn.squeeze(0).sum(dim=1) / (seq_len_actual - input_pos)
                attn = attn.view(
                    1, self.n_heads, -1
                )  # [batch_size, n_heads, prompt_len]
                # can handle both cases, already compressed version from prompt compressor or if prompt_len < cache-len
                seq_len_actual = attn.shape[-1]
                self.attn_score[:, :, :seq_len_actual] = attn.detach()
            if "token_profiling" in self.feature_selection:
                # Token profiling features
                # input_ids: shape (batch, seq_len)
                token_special_profiling = self.build_ids_mask(
                    kwargs["input_ids"], self.special_ids
                )
                token_punctuation_profiling = self.build_ids_mask(
                    kwargs["input_ids"], self.punctuation_ids
                )
                # shape (batch, n_heads, max_cache_len)
                self.token_special_profiling = token_special_profiling.detach()
                self.token_punctuation_profiling = token_punctuation_profiling.detach()
            if "convolution" in self.feature_selection:
                feature_dict = {
                    "embedding": x_filtered.detach(),
                    "query": q_filtered.detach(),
                    "key": k_val.detach(),
                    "value": v_val.detach(),
                }
                valid_mask = self.pos != -1  # shape: [B, H, M]
                # Find indices of valid slots.
                valid_idx = torch.nonzero(valid_mask)  # shape: [N_valid, 3]
                for feat in self.convolution_features:
                    # Get the filtered tensor for this feature.
                    feat_tensor = feature_dict[feat]  # shape: [B, H, M, D]

                    # Gather only valid vectors using the valid indices.
                    # valid_idx has columns: [batch_idx, head_idx, cache_slot_idx].
                    token_vectors = feat_tensor[
                        valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2], :
                    ]  # shape: [N_valid, D]

                    # Prepare for convolution: Conv1d expects input [N, in_channels, width].
                    # Here, treat each token vector as width=D with in_channels=1.
                    tokens_for_conv = token_vectors.unsqueeze(
                        1
                    )  # shape: [N_valid, 1, D]

                    # Apply the convolution for this feature.
                    # Assume self.conv_layers[feat] is your convolution module.
                    conv_out = self.conv_layers[feat](tokens_for_conv).squeeze(
                        1
                    )  # shape: [N_valid, F]

                    # Scatter the convolution output back into the appropriate buffer.
                    # Assume the buffer is stored as an attribute named, for example, "embedding_conv_values".
                    buffer_attr = f"{feat}_conv_values"  # e.g., "embedding_conv_values"
                    conv_buffer = getattr(self, buffer_attr)  # shape: [B, H, M, F]

                    conv_buffer[
                        valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2], :
                    ] = conv_out.detach()

        else:
            # Generation phase: update only the new token's key and value norms,
            # and update full attention scores.
            idx = torch.where(self.pos == input_pos)[
                2
            ]  # Find cache index per head for current input_pos
            if "vector_norm" in self.feature_selection:
                # Use advanced indexing: for each head, select the corresponding vector using idx.
                key_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    k_val[:, torch.arange(self.n_heads), idx, :],
                    ord=2,
                    dim=-1,
                    keepdim=True,
                ).squeeze(-1)
                value_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    v_val[:, torch.arange(self.n_heads), idx, :],
                    ord=2,
                    dim=-1,
                    keepdim=True,
                ).squeeze(-1)
                query_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    q_avg.squeeze(2),
                    ord=2,
                    dim=-1,
                    keepdim=True,
                ).squeeze(-1)
                embedding_norm = torch.linalg.vector_norm(  # pylint: disable=E1102
                    x.squeeze(1),
                    ord=2,
                    dim=-1,
                    keepdim=True,
                )
                # Update only the current token's norms.
                self.key_norm[:, torch.arange(self.n_heads), idx] = key_norm.detach()
                self.value_norm[:, torch.arange(self.n_heads), idx] = (
                    value_norm.detach()
                )
                self.query_norm[:, torch.arange(self.n_heads), idx] = (
                    query_norm.detach()
                )
                self.embedding_norm[:, torch.arange(self.n_heads), idx] = (
                    embedding_norm.expand(batch_size, self.n_heads)
                ).detach()
            if "vector_cv" in self.feature_selection:
                key_cv = self.compute_cv(k_val[:, torch.arange(self.n_heads), idx, :])
                value_cv = self.compute_cv(v_val[:, torch.arange(self.n_heads), idx, :])
                query_cv = self.compute_cv(q_avg).squeeze(-1)
                embedding_cv = self.compute_cv(x).expand(batch_size, self.n_heads)
                self.key_cv[:, torch.arange(self.n_heads), idx] = key_cv.detach()
                self.value_cv[:, torch.arange(self.n_heads), idx] = value_cv.detach()
                self.query_cv[:, torch.arange(self.n_heads), idx] = query_cv.detach()
                self.embedding_cv[:, torch.arange(self.n_heads), idx] = (
                    embedding_cv.detach()
                )
            if "vector_z_score" in self.feature_selection:
                key_z = self.compute_z_score(
                    k_val[:, torch.arange(self.n_heads), idx, :]
                )
                value_z = self.compute_z_score(
                    v_val[:, torch.arange(self.n_heads), idx, :]
                )
                query_z = self.compute_z_score(q_avg).squeeze(-1)
                embedding_z = self.compute_z_score(x).expand(batch_size, self.n_heads)
                self.key_z[:, torch.arange(self.n_heads), idx] = key_z.detach()
                self.value_z[:, torch.arange(self.n_heads), idx] = value_z.detach()
                self.query_z[:, torch.arange(self.n_heads), idx] = query_z.detach()
                self.embedding_z[:, torch.arange(self.n_heads), idx] = (
                    embedding_z.detach()
                )
            if "attn_score" in self.feature_selection:
                attn = attn.view(
                    1, self.n_heads, -1
                )  # Reshape attention scores to [batch_size, n_heads, prompt_len]
                self.attn_score[:] = attn.detach()  # update all attention score values
            if "token_profiling" in self.feature_selection:
                special_mask = torch.isin(kwargs["input_ids"], self.special_ids)
                punct_mask = torch.isin(kwargs["input_ids"], self.punctuation_ids)
                special_mask_exp = special_mask.expand(batch_size, self.n_heads)
                punct_mask_exp = punct_mask.expand(batch_size, self.n_heads)

                self.token_special_profiling[:, torch.arange(self.n_heads), idx] = (
                    special_mask_exp.detach()
                )  # save as bool here
                self.token_punctuation_profiling[:, torch.arange(self.n_heads), idx] = (
                    punct_mask_exp.detach()
                )  # save as bool here
            if "convolution" in self.feature_selection:
                feature_dict = {
                    "embedding": x,  # size (batch, 1, head dim) # works like that
                    "query": q_avg.squeeze(2),  # size (batch, n_heads, head dim)
                    "key": k_val[
                        :, torch.arange(self.n_heads), idx, :
                    ],  # size (batch, n_heads, head dim)
                    "value": v_val[
                        :, torch.arange(self.n_heads), idx, :
                    ],  # size (batch, n_heads, head dim)
                }
                for feat in self.convolution_features:
                    # Get the filtered tensor for this feature.
                    feat_tensor = feature_dict[feat]
                    # Get the convolution layer for this feature.
                    conv_block = self.conv_layers[feat]
                    # Get the buffer for this feature.
                    buffer_attr = f"{feat}_conv_values"
                    conv_buffer = getattr(self, buffer_attr)
                    if feat == "embedding":
                        vector_conv = conv_block(feat_tensor)
                        vector_conv = vector_conv.expand(batch_size, self.n_heads, -1)
                        # size (batch, n_heads, result feature dim)
                    else:
                        vector_conv = conv_block(
                            feat_tensor.view(batch_size * self.n_heads, 1, head_dim)
                        )
                        vector_conv = vector_conv.view(batch_size, self.n_heads, -1)
                        # size (batch, n_heads, result feature dim)
                    conv_buffer[:, torch.arange(self.n_heads), idx, :] = (
                        vector_conv.detach()
                    )

    def _eviction_idx(self, input_pos):
        scores = self._token_importances(input_pos)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)

        # Protect global tokens
        scores[:, : self.global_tokens] = float("inf")

        # Evict unfilled slots (pos == -1)
        scores.masked_fill_(self.pos.view(scores.shape) == -1, float("-inf"))

        # Apply masking to protect recent tokens based on the observation window
        scores.masked_fill_(self.pos >= input_pos - self.recent_window, float("inf"))

        # Evict least important token
        return torch.argmin(scores, dim=-1)

    def _token_importances(self, input_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute token importance scores using lightweight models.

        This function extracts features from the entire cache (key norms, value norms,
        and attention scores) and computes an importance score for each token using a
        lightweight model (either linear or MLP) per attention head. The resulting scores
        indicate which tokens in the cache are most important and have shape
        [batch_size, n_heads, max_cache_length].

        Args:
            input_pos (torch.Tensor): The input positions in the sequence.

        Returns:
            torch.Tensor: Token importance scores with shape [batch_size, n_heads, max_cache_length].
        """
        max_cache_length = self.key_norm.shape[-1]
        normalized_pos = self.pos.float() / input_pos[-1].float()

        features_to_cat = []
        if "attn_score" in self.feature_selection:
            features_to_cat.append(self.attn_score.unsqueeze(-1))
        if "vector_norm" in self.feature_selection:
            features_to_cat.extend(
                [
                    self.key_norm.unsqueeze(-1),
                    self.value_norm.unsqueeze(-1),
                    self.query_norm.unsqueeze(-1),
                    self.embedding_norm.unsqueeze(-1),
                ]
            )
        if "vector_cv" in self.feature_selection:
            features_to_cat.extend(
                [
                    self.key_cv.unsqueeze(-1),
                    self.value_cv.unsqueeze(-1),
                    self.query_cv.unsqueeze(-1),
                    self.embedding_cv.unsqueeze(-1),
                ]
            )
        if "vector_z_score" in self.feature_selection:
            features_to_cat.extend(
                [
                    self.key_z.unsqueeze(-1),
                    self.value_z.unsqueeze(-1),
                    self.query_z.unsqueeze(-1),
                    self.embedding_z.unsqueeze(-1),
                ]
            )
        if "token_profiling" in self.feature_selection:
            features_to_cat.extend(
                [
                    self.token_special_profiling.unsqueeze(-1),
                    self.token_punctuation_profiling.unsqueeze(-1),
                ]
            )
        if "convolution" in self.feature_selection:
            for feat in self.convolution_features:
                buffer_attr = f"{feat}_conv_values"
                features_to_cat.append(getattr(self, buffer_attr))
        if "normalized_pos" in self.feature_selection:
            features_to_cat.append(normalized_pos.unsqueeze(-1))

        features_to_cat = [feature.to(torch.bfloat16) for feature in features_to_cat]
        features = torch.cat(features_to_cat, dim=3)

        # Initialize scores with -infinity for all cache slots
        scores = torch.full(
            (features.shape[0], self.n_heads, max_cache_length),
            float("-inf"),
            dtype=features.dtype,
            device=features.device,
        )

        # Compute scores for the current valid positions using the lightweight models
        for head_idx, model in enumerate(self.models):
            valid = self.pos[:, head_idx, :] != -1  # Boolean mask of valid positions
            if valid.any():
                # Compute scores for valid positions and update the scores tensor
                scores[:, head_idx, :][valid] = model(
                    features[:, head_idx, :][valid]
                ).squeeze(-1)

        # Remove batch dimension if batch_size == 1, as the method is not batch-compatible yet
        if scores.shape[0] == 1:
            scores = scores.squeeze(0)  # Resulting shape: [n_heads, max_cache_length]

        return scores


class KVCacheHeavyHitter(KVCacheHeadSpecific):
    # This class mostly follows the logic in ScissorHands (https://arxiv.org/abs/2305.17118)
    # But it is very similar to other Heavy Hitter methods (H20, PyramidKV, etc.)
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            variable_length,
            **kwargs,
        )

        # Initialize a buffer for the attention histories
        history_num_shape = (
            max_batch_size,
            n_heads,
            self.max_cache_length,
            self.history_window_size,
        )
        history_denom_shape = (
            max_batch_size,
            n_heads,
            self.max_cache_length,
        )
        # If attn_thresholding, we store a binary indicator of whether the attention >= uniform attention
        # If not, we store the raw attention values
        # If history_window_size = 1, we accumulate the full history in one slot so we need a dtype with large range
        history_num_dtype = (
            torch.bool
            if self.attn_thresholding
            else torch.float64 if self.history_window_size == 1 else dtype
        )
        self.register_buffer(
            "attn_history_num",
            torch.zeros(history_num_shape, dtype=history_num_dtype),
        )

        # Ideally, we could use the self.pos to track the number of times a token has been attended to
        # But any change to cache management or how self.pos is stored would break this.
        self.register_buffer(
            "attn_history_denom", torch.zeros(history_denom_shape, dtype=torch.int32)
        )

        self.register_buffer("attn_counter", torch.zeros((1,), dtype=torch.int64))

    def reset(self):
        super().reset()
        self.attn_history_num.zero_()
        self.attn_history_denom.zero_()
        self.attn_counter.zero_()

    def return_attn(self) -> bool:
        return True

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """

        # Resize attn to be max cache length with zero padding if need be
        seq_len = attn.shape[-1]

        if (
            is_prefill and attn.ndim == 4
        ):  # Prefill, we may receive the full attention map and have to average across queries
            # Normalize using input_pos to only count non-zero attentions bc/ of causal mask
            attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)

        attn = attn.view(1, self.n_heads, -1, 1)
        attn = (attn >= 1 / self.cache_cts).int() if self.attn_thresholding else attn

        # Torch.compile doesn't support dyanmic slicing so we need to zero-pad to full dimension
        padding = max(self.max_cache_length - seq_len, 0)
        pad_attn = torch.zeros(
            1, self.n_heads, padding, 1, dtype=attn.dtype, device=attn.device
        )
        attn = torch.cat([attn, pad_attn], dim=2)

        history_idx = self.attn_counter % self.history_window_size

        if self.history_window_size == 1:  # We consider the full history
            self.attn_history_num[:, :, :, history_idx] += attn
        else:
            self.attn_history_num[:, :, :, history_idx] = attn
        self.attn_history_denom += 1
        self.attn_counter += 1

    def _eviction_idx(self, input_pos):
        # Identify the token with consistently "lowest" attention
        numerator = self.attn_history_num.sum(dim=-1).float()

        if (
            self.history_window_size == 1
        ):  # We use the full history (there is no clamping around a fixed window)
            denominator = self.attn_history_denom.clamp_min(1)
        else:
            # The denominator is the number of times this token's history has been recorded
            # We only record most self.history_window_size recent scores so need to clamp it
            denominator = self.attn_history_denom.clamp(1, self.history_window_size)

        avg_attn = numerator / denominator

        # Save the global & most recent tokens from being evicted
        avg_attn.masked_fill_(
            torch.logical_or(
                self.pos < self.global_tokens,
                self.pos >= input_pos - self.recent_window,
            ),
            1.0,
        )

        avg_attn.masked_fill_(self.pos == -1, 0.0)

        fill_idxs = avg_attn.argmin(dim=-1).squeeze()

        # Zero-out the attention history for these newly inserted slots
        num_fill = fill_idxs.view(1, -1, 1, 1).expand(
            1, -1, 1, self.attn_history_num.shape[-1]
        )
        denom_fill = fill_idxs.view(1, -1, 1)
        self.attn_history_num.scatter_(
            2, num_fill, torch.zeros_like(num_fill, dtype=self.attn_history_num.dtype)
        )
        self.attn_history_denom.scatter_(
            2, denom_fill, torch.zeros_like(denom_fill, dtype=torch.int32)
        )

        return fill_idxs


class KVCacheHybrid(KVCacheHeavyHitter):
    # This class mostly follows the logic in FastGen (https://arxiv.org/abs/2310.01801)
    # Yet, it allows for a wider set of hybrid strategies to be considered during profiling.
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        "token_ids",
        "min_recovery_frac",
        "hybrid_strategies",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        self.attn_thresholding = False
        self.history_window_size = 400  # Default value for ScissorHands
        self.recent_window = (
            None  # Dummy value: Recent windows are defined per attention head
        )
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            variable_length=True,
            **kwargs,
        )

        self.requires_special = any(
            ["special" in strat["strategy"] for strat in self.hybrid_strategies]
        )
        mask_shape = (max_batch_size, n_heads, self.max_cache_length)
        if self.requires_special:
            special_ids = [torch.tensor(ids) for ids in kwargs["token_ids"]["special"]]
            self.register_buffer("special_ids", torch.nested.nested_tensor(special_ids))
            # As well as a mask showing where special ids are in the KV cache
            # We store this to avoid re-computing the mask every time and having to store all input_ids
            self.register_buffer(
                "special_mask", torch.zeros(mask_shape, dtype=torch.bool)
            )
            self.register_buffer("num_special", torch.zeros((1,), dtype=torch.int))

        self.requires_punc = any(
            ["punc" in strat["strategy"] for strat in self.hybrid_strategies]
        )
        if self.requires_punc:
            # Store the punctuation vocabulary ids
            punc_ids = torch.Tensor(kwargs["token_ids"]["punctuation"])
            self.register_buffer("punc_ids", punc_ids)
            # As well as a mask showing where punctuation ids are in the KV cache
            # We store this to avoid re-computing the mask every time and having to store input_ids
            self.register_buffer("punc_mask", torch.zeros(mask_shape, dtype=torch.bool))
            self.register_buffer("num_punc", torch.zeros((1,), dtype=torch.int))

        self.requires_heavy_hitter = self._init_requires_heavy_hitter()

        # We need to use a mask since not all heads have same number of tokens. We can't simply truncate.
        # 1 dimension stands for query dimension, which will always be 1 (next token) for KV cache attention.
        kv_mask_shape = (max_batch_size, n_heads, 1, self.max_cache_length)
        self.register_buffer("mask", torch.zeros(kv_mask_shape, dtype=torch.bool))

    def return_attn(self):
        return self.requires_heavy_hitter

    def _init_requires_heavy_hitter(self):
        return any(
            ["heavy_hitter" in strat["strategy"] for strat in self.hybrid_strategies]
        )

    def _eviction_idx_for_head(
        self,
        head_idx,
        input_pos,
        recent_window,
        apply_heavy_hitter=False,
        apply_window=False,
        apply_special=False,
        apply_punc=False,
    ):
        if apply_heavy_hitter:
            numerator = (
                self.attn_history_num[:, head_idx, : self.cache_cts[head_idx]]
                .sum(dim=-1)
                .float()
            )

            if self.history_window_size == 1:  # Use full history
                denominator = self.attn_history_denom[
                    :, head_idx, : self.cache_cts[head_idx]
                ]
            else:
                # The denominator is the number of times this token's history has been recorded
                # We only record most self.history_window_size recent scores so need to clamp it
                denominator = self.attn_history_denom[
                    :, head_idx, : self.cache_cts[head_idx]
                ].clamp_max(self.history_window_size)
            score = numerator / denominator
        else:
            score = self.pos[:, head_idx, : self.cache_cts[head_idx]].clone().float()

        save_mask = torch.zeros_like(score, dtype=torch.bool)
        save_mask[:, : self.global_tokens] = 1

        if apply_special:
            save_mask |= self.special_mask[:, head_idx, : self.cache_cts[head_idx]]

        if apply_punc:
            save_mask |= self.punc_mask[:, head_idx, : self.cache_cts[head_idx]]

        if apply_window:
            window_mask = (
                self.pos[:, head_idx, : self.cache_cts[head_idx]]
                > input_pos - recent_window
            )
            save_mask |= window_mask

        score.masked_fill_(save_mask, float("inf"))
        fill_idx = score.argmin(dim=-1)

        return fill_idx

    def _select_fill_idx(self, strategy, head_idx, input_pos, is_punc: bool = False):
        def _end_idx():
            # We need to clone because self.cache_cts will be incremented later and we don't want to have fill_idx as a mutable reference
            return min(self.max_cache_length - 1, self.cache_cts[head_idx].clone())

        strategy = self.hybrid_strategies[strategy]
        name = strategy["strategy"]

        # If is punctuation token and we are preserving, we always add it to the end
        if "punc" in name and is_punc:
            return _end_idx(), False

        if name == "full":
            return _end_idx(), False

        # Every strategy has a budget for global tokens
        budget = torch.tensor(
            [self.global_tokens], dtype=torch.int, device=input_pos.device
        )
        if "special" in name:
            budget += self.num_special

        if "punc" in name:
            budget += self.num_punc

        if "window" in name:
            budget += round(strategy["recent_window"] * self.max_cache_length)

        if "heavy_hitter" in name:
            budget += round(strategy["heavy_hitter_frac"] * self.max_cache_length)

        eviction_required = self.cache_cts[head_idx] >= budget

        if not eviction_required:
            return _end_idx(), False

        if "heavy_hitter" in name or "window" in name:
            recent_window = round(
                strategy.get("recent_window", 0) * self.max_cache_length
            )
            fill_idx = self._eviction_idx_for_head(
                head_idx,
                input_pos,
                recent_window=recent_window,
                apply_heavy_hitter="heavy_hitter" in name,
                apply_window="window" in name,
                apply_punc="punc" in name,
                apply_special="special" in name,
            )

            return fill_idx, True  # Eviction Required

        # If we reach here, we have a hybrid strategy that is not window, heavy hitter, or full
        assert "punc" in name or "special" in name, f"Invalid hybrid strategy {name}"
        return None, False

    def reset(self):
        super().reset()
        self.cache_strategies.fill = None  # Free up memory temporarily
        self.requires_heavy_hitter = self._init_requires_heavy_hitter()

        if hasattr(self, "special_mask"):
            self.special_mask.zero_()
            self.num_special.zero_()

        if hasattr(self, "punc_mask"):
            self.punc_mask.zero_()
            self.num_punc.zero_()

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        input_ids = kwargs.get("input_ids")
        n_heads = k_val.shape[1]

        is_punc = (
            torch.isin(input_ids, self.punc_ids) if hasattr(self, "punc_ids") else False
        )

        # If fill idx is None we place value at the back (which is truncated for attention calculation anyway)
        fill_indices = torch.full(
            (n_heads,),
            self.max_cache_length - 1,
            dtype=torch.int64,
            device=k_val.device,
        )

        cache_ct_incr = torch.zeros_like(fill_indices)

        for head_idx, strategy in enumerate(self.cache_strategies):
            fill_idx, eviction_required = self._select_fill_idx(
                strategy, head_idx, input_pos, is_punc=is_punc
            )

            if fill_idx is None:
                continue

            fill_indices[head_idx] = fill_idx
            if eviction_required:
                if self.requires_heavy_hitter:
                    # Reset attention history since we've inserted a new token
                    self.attn_history_num[:, head_idx, fill_idx, :].fill_(0)
                    self.attn_history_denom[:, head_idx, fill_idx].fill_(0)
            else:
                # Increment cache_ct_incr for heads that have grown (no eviction)
                cache_ct_incr[head_idx] = 1
                # We can't use all fill indices to bulk assign mask because some fill_indices are dummies (self.max_cache_length - 1)
                self.mask[:, head_idx, :, fill_idx] = True

        # We have to insert all new tokens into the cache to be be able to bulk insert
        # But some aren't actually being inserted (fill_idx = self.max_cache_length - 1)
        # Thus we only flip the mask to True for the tokens that are actually being inserted (done above)
        kwargs = {"update_mask": False}
        self._fill(input_pos, k_val, v_val, fill_indices, **kwargs)

        # Only update global self.num_punc once (not once per head)
        # If a head keeps punc tokens, each head will have same number of punc tokens (no punc evictions)
        if is_punc and hasattr(self, "num_punc"):
            self.punc_mask.scatter_(
                2,
                fill_indices.view(1, -1, 1),
                is_punc.view(1, 1, 1).expand(1, n_heads, 1),
            )
            self.num_punc += 1

        return cache_ct_incr

    def build_special_ids_mask(self, input_ids):
        seq_len = input_ids.shape[-1]
        special_ids_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for special_id in self.special_ids:
            # Iterate over input_ids to check for the exact sub-sequence
            id_len = len(special_id)
            if id_len == 1:
                special_ids_mask[torch.where(input_ids == special_id)[0]] = True
            else:
                for i in range(seq_len - id_len + 1):
                    if torch.equal(input_ids[i : i + id_len], special_id):
                        special_ids_mask[i : i + id_len] = True
        return special_ids_mask

    def build_punc_ids_mask(self, input_ids):
        # TODO should be on same device as model with register_buffer
        if self.punc_ids.device != input_ids.device:
            self.punc_ids = self.punc_ids.to(input_ids.device)
        punc_ids_mask = torch.isin(input_ids, self.punc_ids)
        return punc_ids_mask

    def compute_statistics(self, seq_len):
        stats = super().compute_statistics(seq_len)

        # Compute counts of usage for hybrid strategies
        cts = Counter(
            [
                self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies.tolist()
            ]
        )
        stats["avg_strategy_idx"] = sum(self.cache_strategies.tolist()) / len(
            self.cache_strategies
        )
        stats.update(
            {
                strategy: cts.get(strategy, 0) / len(self.cache_strategies)
                for strategy in sorted(
                    list(set([x["strategy"] for x in self.hybrid_strategies]))
                )
            }
        )
        return stats

    def build_masks(self, cum_attn, special_mask, punc_mask, total_len):
        device = cum_attn.device
        n_heads, seq_len = cum_attn.shape
        masks = []
        for s in self.hybrid_strategies:
            strat_mask = torch.zeros(
                n_heads, seq_len, seq_len, dtype=torch.bool, device=device
            )
            # All strategies have global tokens
            strat_mask[:, :, : self.global_tokens] = True

            name = s["strategy"]
            if "special" in name:
                strat_mask |= special_mask.view(1, 1, -1).expand(
                    n_heads, seq_len, seq_len
                )

            if "punc" in name:
                strat_mask |= punc_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)

            if "window" in name:
                assert (
                    "recent_window" in s and s["recent_window"] <= 1
                ), "Window strategy should have recent_window expressed as a fraction <= 1."
                strat_mask |= (
                    create_window_attention_mask(
                        seq_len,
                        max(1, int(s["recent_window"] * total_len)),
                        device,
                        global_tokens=self.global_tokens,
                    )
                    .unsqueeze(0)
                    .expand(n_heads, seq_len, seq_len)
                )

            if "heavy_hitter" in name:
                # Compute heavy hitters over tokens which are still masked
                avail_idxs = torch.where(~strat_mask[0, -1, :])[0]

                attn_slice = cum_attn.gather(
                    1, avail_idxs.unsqueeze(0).expand(n_heads, -1)
                )

                num_hh = math.ceil(
                    min(
                        s["heavy_hitter_frac"] * total_len,
                        len(avail_idxs),
                    )
                )

                heavy_hitters = (
                    attn_slice.topk(num_hh, dim=1, largest=True)
                    .indices.sort(dim=1)
                    .values
                )

                heavy_hitters_idx = (
                    avail_idxs.view(1, -1).expand(n_heads, -1).gather(1, heavy_hitters)
                )
                strat_mask.scatter_(
                    2,
                    heavy_hitters_idx.view(n_heads, 1, num_hh).expand(
                        n_heads, seq_len, num_hh
                    ),
                    True,
                )
            if name == "full":
                strat_mask.fill_(True)

            masks.append(strat_mask)
        return torch.stack(masks)

    def profile_attn_heads(self, input_pos, attn, **kwargs):
        input_ids = kwargs["input_ids"]
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]

        # Only build masks as needed
        special_mask = punc_mask = None
        if self.requires_special:
            special_mask = self.build_special_ids_mask(input_ids)
            self.num_special = special_mask.sum()

        if self.requires_punc:
            punc_mask = self.build_punc_ids_mask(input_ids)
            self.num_punc = punc_mask.sum()

        cum_attn = (
            None  # Only aggregate attention if its needed by one of the strategies
        )
        if any(["heavy_hitter" in s["strategy"] for s in self.hybrid_strategies]):
            # Average of cumulative attention probs (use input_pos to normalize)
            cum_attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)

        masks_for_scoring = self.build_masks(
            cum_attn, special_mask, punc_mask, total_len=seq_len
        )

        # Compute optimal strategies for each head based on prompt proportions
        attn_rep = attn.expand(masks_for_scoring.shape[0], -1, -1, -1)
        compressed_scores = (
            attn_rep.masked_fill(~masks_for_scoring, 0).sum(dim=-1).mean(dim=-1)
        )

        # For each column, return the first row which has cost >= min_recovery_frac
        cache_strategies = (
            (compressed_scores >= self.min_recovery_frac).int().argmax(dim=0)
        )

        # Base insertions on the optimal strategy across full sequence length
        assert self.max_cache_length >= seq_len
        masks_for_filling = self.build_masks(
            cum_attn, special_mask, punc_mask, total_len=self.max_cache_length
        )
        # Take the last query's mask as the initial KV-Cache fill mask
        masks_all = masks_for_filling[:, :, -1, :].transpose(1, 0)
        # Select mask based on self.cache_strategies
        mask_optimal = masks_all.gather(
            1, cache_strategies.view(-1, 1, 1).expand(-1, -1, seq_len)
        ).squeeze(1)

        return cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn

    def profile_and_update(self, input_pos, k_val, v_val, attn, **kwargs):
        """
        Profile the attention heads to determine the optimal KV-cache allocation.
        """
        input_ids = kwargs["input_ids"]
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        n_heads = attn.shape[1]
        dim = k_val.shape[-1]

        # Profile cache attention heads to define strategy for each head
        self.cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn = (
            self.profile_attn_heads(input_pos, attn, **kwargs)
        )

        # Uncomment to show which strategies are selected
        # print([self.hybrid_strategies[i] for i in self.cache_strategies.tolist()])

        # If none of the heads selected a heavy hitter strategy, we don't need to track attention weights
        # Same for punctuation and special tokens
        self.requires_heavy_hitter = any(
            [
                "heavy_hitter" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )
        self.requires_punc = any(
            [
                "punc" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )
        self.requires_special = any(
            [
                "special" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )

        # Put the selected items (true values from mask) to the front. Re-arrange attentions as well.
        order = mask_optimal.int().argsort(dim=1, descending=True)
        order_exp = order.view(1, n_heads, seq_len, 1).expand(-1, -1, -1, dim)

        # We dump all the KV pairs into the cache yet order them based on the optimal strategy
        k_val = k_val.gather(2, order_exp)
        v_val = v_val.gather(2, order_exp)
        input_pos = input_pos.unsqueeze(0).expand(n_heads, -1).gather(1, order).int()
        fill_idxs = torch.arange(seq_len, device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs)

        # Record number of tokens to be inserted into the cache
        self.cache_cts = mask_optimal.sum(dim=1)

        # Can remove for speed: doesn't change code but makes it easier to debug and see what's actually in the cache
        for head_idx in range(n_heads):
            self.pos[:, head_idx, self.cache_cts[head_idx] :].fill_(-1)
            self.k_cache[:, head_idx, self.cache_cts[head_idx] :].fill_(0)
            self.v_cache[:, head_idx, self.cache_cts[head_idx] :].fill_(0)

        if hasattr(self, "special_mask"):
            # We will need to remove special tokens and punctuation from heavy hitter eviction so need to their positions.
            special_mask = special_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
            self.special_mask[:, :, :seq_len] = special_mask

        if hasattr(self, "punc_mask"):
            punc_mask = punc_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
            self.punc_mask[:, :, :seq_len] = punc_mask

        # Update mask to reflect how many items have been inserted into each head
        range_mask = (
            torch.arange(seq_len, device=self.mask.device)
            .view(1, -1)
            .expand(n_heads, -1)
        )
        self.mask[:, :, :, :seq_len] = (
            range_mask < self.cache_cts.view(-1, 1).expand(-1, seq_len)
        ).view(-1, n_heads, 1, seq_len)

        if self.requires_heavy_hitter:
            # Update attention mask to indicate which we attentions are allowed.
            cum_attn = cum_attn.gather(1, order).unsqueeze(0)
            super().update_state(
                input_pos, k_val, v_val, is_prefill=True, attn=cum_attn, **kwargs
            )

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """
        # We handle state updating during prefill for Hybrid as part of the profile and update stage
        if is_prefill:
            self.profile_and_update(input_pos, k_val, v_val, attn, **kwargs)
        elif (
            self.requires_heavy_hitter
        ):  # If none of the heads require attention, there's no state to update
            super().update_state(input_pos, k_val, v_val, is_prefill, attn, **kwargs)
        else:
            assert attn is None, "Attn should be None if no attention is required."


class KVCacheAnalysis(KVCacheFull):
    """
    This cache is triggered by prepending `debug_` to an existing cache strategy.

    It will analyze the attention loss incurred from compressing with that cache strategy.
    """

    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
        "global_tokens",
        "prompt_compression_strategy",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        cache_strategy="heavy_hitter",
        **kwargs,
    ):
        # Never any prompt compression for full cache
        full_kwargs = {
            "global_tokens": 0,  # Every token gets saved (no explicit global tokens)
            "max_cache_length": kwargs["max_seq_length"],
            "prompt_compression_strategy": kwargs["prompt_compression_strategy"],
        }
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **full_kwargs)

        # Initialize the compressed cache we want to analyze.
        self.compressed = get_cache_constructor(cache_strategy=cache_strategy)[0](
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            **kwargs,
        )

        self.register_buffer(
            "attention_losses",
            torch.full((self.max_cache_length,), fill_value=-1, dtype=dtype),
        )

        self.register_buffer(
            "attention_loss_ctr",
            torch.zeros((1,), dtype=torch.int),
        )

        self.prompt_compressor = get_prompt_compressor_constructor(
            self.prompt_compression_strategy
        )(head_specific=self.compressed.head_specific, **kwargs)

        # Necessary for compatability check with prompt compression strategy
        self.head_specific = self.compressed.head_specific

    def return_attn(self):
        return self.compressed.return_attn()

    def update_kv(self, input_pos, k_val, v_val, is_prefill, **kwargs):
        k, v, mask = super().update_kv(input_pos, k_val, v_val, is_prefill, **kwargs)
        # Conditionally update the compressed cache if prompt < max_cache_length

        # If prompt is too long for compressed cache we will need to compress it in update_state before inserting.
        # We need to wait for update_state because we might need attention for compression
        can_update_compressed = input_pos.shape[-1] < self.compressed.max_cache_length
        if can_update_compressed:
            _, _, _ = self.compressed.update_kv(
                input_pos, k_val, v_val, is_prefill, **kwargs
            )

        return k, v, mask

    def reset(self):
        super().reset()
        self.compressed.reset()
        self.attention_losses.fill_(-1)
        self.attention_loss_ctr.zero_()

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        # We might need to compress the prompt and update the compressed cache
        needs_prompt_compression = (
            is_prefill and input_pos.shape[-1] > self.compressed.max_cache_length
        )
        if needs_prompt_compression:
            kwargs = {"attn": attn}
            input_pos, k_val, v_val, attn = self.prompt_compressor(
                input_pos, k_val, v_val, **kwargs
            )
            _, _, _ = self.compressed.update_kv(input_pos, k_val, v_val, is_prefill)
            self.compressed.update_state(input_pos, k_val, v_val, is_prefill, attn)
        elif is_prefill:
            # Don't record attention loss in prefill since compressed and non-compressed prefill attentions are the same
            # Just update the state for the compressed cache and return
            self.compressed.update_state(input_pos, k_val, v_val, is_prefill, attn)
        else:
            assert not is_prefill
            indices = self.compressed.pos.clone().long()
            # Avoid scatter issue we need to assign unfilled indices to last attention value (which will also be 0)
            indices[indices == -1] = attn.shape[-1] - 1
            attn_compressed = attn.squeeze(2).gather(2, indices)
            self.compressed.update_state(
                input_pos, k_val, v_val, is_prefill, attn_compressed
            )

            # Compute attention loss as the sum of the attention probabilities for evicted tokens
            # Equivalently, 1 - the sum of the attention probabilities for the tokens in the compressed cache
            attn_loss = (1 - attn_compressed.sum(dim=-1)).mean()
            self.attention_losses[self.attention_loss_ctr] = attn_loss
            self.attention_loss_ctr += 1

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        stats = super().compute_statistics(seq_len)
        losses = self.attention_losses[: self.attention_loss_ctr]
        assert not torch.any(losses == -1)
        for k in range(500, len(losses), 500):
            stats[f"attention_loss@{k}"] = losses[:k].mean().item()
        stats["attention_loss"] = losses.mean().item()
        return stats


class KVCacheKeepItOdd(KVCacheHeadConstant):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "cache_bits",
        "global_tokens",
        "recent_window",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _token_importances(self, input_pos):
        scores = torch.zeros_like(self.pos[:, 0], dtype=torch.bfloat16)
        scores[self.pos[:, 0] % 2 == 1] = 1.0
        scores[self.pos[:, 0] >= input_pos - self.recent_window] = float("inf")
        return scores


def get_cache_constructor(cache_strategy):
    relevant_kwargs = None
    if cache_strategy == "full":
        cls = KVCacheFull
    elif cache_strategy == "l2":
        cls = KVCacheL2
    elif cache_strategy == "random":
        cls = KVCacheRandom
    elif cache_strategy == "recent_global":
        cls = KVCacheRecentGlobal
    elif cache_strategy == "heavy_hitter":
        cls = KVCacheHeavyHitter
    elif cache_strategy == "hybrid":
        cls = KVCacheHybrid
    elif cache_strategy == "keep_it_odd":
        cls = KVCacheKeepItOdd
    elif cache_strategy == "lightweight":
        cls = KVCacheLightweight
    elif cache_strategy.startswith("debug"):
        cache_strategy = re.sub(r"debug_+", "", cache_strategy).strip()
        relevant_kwargs = get_cache_constructor(cache_strategy)[1] + [
            "prompt_compression_strategy"
        ]
        cls = (
            lambda max_batch_size, n_heads, head_dim, dtype, **kwargs: KVCacheAnalysis(
                max_batch_size,
                n_heads,
                head_dim,
                dtype,
                cache_strategy=cache_strategy,
                **kwargs,
            )
        )
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")

    return cls, relevant_kwargs or cls.relevant_kwargs
