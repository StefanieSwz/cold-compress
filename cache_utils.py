import math
import operator
import tempfile
from pathlib import Path
from typing import Any, Dict, Union
import torch
import torch.nn as nn
import wandb


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


def save_lightweight_temp(model: nn.Module, kwargs: Dict[str, Any]):
    """
    Save lightweight model parameters temporarily.

    Returns:
        (Path, TemporaryDirectory): Path to the saved file and the temporary directory object.
        The caller is responsible for managing the lifetime of the temp directory.
    """
    temp_dir = tempfile.TemporaryDirectory()
    save_path = Path(temp_dir.name) / "model.pth"

    trained_weights = {
        name: param
        for name, param in model.state_dict().items()
        if name in [n for n, p in model.named_parameters() if p.requires_grad]
    }

    torch.save(trained_weights, save_path)
    print(f"Trained weights saved temporarily at {save_path}")

    return save_path, temp_dir


def load_trained_lightweight(
    model: nn.Module, checkpoint_path: Union[str, Path], load_kv: bool = True
) -> None:
    """
    Load trained weights into a model, updating only matching layers for the specified target.

    This function renames trained weights (if needed), matches them with model keys,
    and updates the model accordingly.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (Union[str, Path]): Path to the saved weights.
        load_kv (bool): Determines which target to load weights for.
                        True -> Load KV-Cache ("attention.kv_cache")
                        False -> Load Prompt Compressor ("attention.prompt_compressor")

    Returns:
        None
    """
    # Load trained weights
    trained_weights: Dict[str, torch.Tensor] = torch.load(
        checkpoint_path, map_location="cpu"
    )
    model_state: Dict[str, torch.Tensor] = model.state_dict()

    # Define substrings for model and trained weights
    trained_key_substring = "attention.kv_cache"  # Always in trained weights
    model_key_substring = (
        "attention.kv_cache" if load_kv else "attention.prompt_compressor"
    )

    remapped_weights = {}

    if not load_kv:
        # Rename trained weights to fit the model's "attention.prompt_compressor" structure
        for trained_key, weight in trained_weights.items():
            if trained_key_substring in trained_key:
                model_key = trained_key.replace(
                    trained_key_substring, model_key_substring
                )
                if model_key in model_state:  # Ensure it's a valid model weight
                    remapped_weights[model_key] = weight
    else:
        remapped_weights = trained_weights  # No renaming needed for KV-cache

    # Ensure all trained weights have a matching model weight
    unmatched_weights = set(remapped_weights.keys()) - set(model_state.keys())
    assert (
        not unmatched_weights
    ), f"⚠️ The following trained weights did not match any model weights: {unmatched_weights}"

    # Load the updated weights into the model
    model_state.update(remapped_weights)
    model.load_state_dict(model_state, strict=False)

    print(f"✅ Loaded trained weights for {model_key_substring} from {checkpoint_path}")


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


def prime_factors(n):
    """Returns a list of prime factors of n."""
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors


def get_closest_valid_target(input_dim, target_features):
    """Finds the closest valid divisor to target_features using direct modulo testing."""

    if input_dim % target_features == 0 and target_features >= 3:
        return target_features  # If it's already valid, return it

    # Check upwards and downwards for the closest valid divisor
    lower, upper = target_features, target_features

    while lower >= 3 or upper <= input_dim:
        if lower >= 3 and input_dim % lower == 0:
            return lower
        if upper <= input_dim and input_dim % upper == 0:
            return upper
        lower -= 1
        upper += 1


def get_convolution_params(input_dim, target_features=8):
    """
    Computes kernel sizes and strides based on prime factorization,
    ensuring a structured compression to target_features.
    """
    factors = prime_factors(input_dim)
    num_features = get_closest_valid_target(input_dim, target_features)
    compression_rate = input_dim // num_features  # Needed compression
    feature_factors = prime_factors(num_features)

    remaining_factors = factors.copy()
    for f in feature_factors:
        remaining_factors.remove(f)  # Remove three 2s

    # Split remaining factors as evenly as possible into two groups
    sorted_factors = sorted(remaining_factors, reverse=True)
    part1, part2 = [], []
    for f in sorted_factors:
        if math.prod(part1) <= math.prod(part2):
            part1.append(f)
        else:
            part2.append(f)

    kernel_size_1 = max(1, math.prod(part1))  # Avoid kernel size 0
    kernel_size_2 = max(1, math.prod(part2))  # Avoid kernel size 0
    hidden_channels = sorted_factors[-2] * sorted_factors[-1]

    return kernel_size_1, kernel_size_2, compression_rate, hidden_channels
