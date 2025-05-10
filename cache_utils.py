import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Union
import torch
import torch.nn as nn


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
    Saves only the trainable parameters of a model to a temporary directory.

    This function extracts the parameters of the model that require gradients
    (i.e., were updated during training), saves them to a temporary `.pth` file,
    and returns both the file path and the `TemporaryDirectory` object.

    Args:
        model (nn.Module): The PyTorch model whose trainable parameters are to be saved.
        kwargs (Dict[str, Any]): Additional arguments for compatibility; not used directly.

    Returns:
        Tuple[Path, tempfile.TemporaryDirectory]: A tuple containing:
            - Path: The path to the temporary `.pth` file containing the saved weights.
            - tempfile.TemporaryDirectory: The temporary directory object. The caller is
              responsible for keeping this object alive for as long as the file is needed.

    Example:
        >>> path, tmp_dir = save_lightweight_temp(model, {})
        >>> # use the file at `path`
        >>> tmp_dir.cleanup()  # when done
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
    Loads trained weights into a model, updating only the layers relevant to KV-cache
    or prompt compression.

    Depending on the `load_kv` flag, this function maps trained keys to the appropriate
    module in the model (`attention.kv_cache` or `attention.prompt_compressor`) and
    loads them into the current model. It ensures that only valid keys are used for the update.

    Args:
        model (nn.Module): The model into which the weights should be loaded.
        checkpoint_path (Union[str, Path]): Path to the `.pt` or `.pth` checkpoint file
            containing the trained weights.
        load_kv (bool, optional): If True, loads weights for `attention.kv_cache`.
            If False, loads weights into `attention.prompt_compressor` instead.
            Defaults to True.

    Returns:
        None

    Raises:
        AssertionError: If any of the remapped trained weights do not match existing
            keys in the model's state_dict.
        FileNotFoundError: If the checkpoint file is not found.
        RuntimeError: If the checkpoint file cannot be loaded or is incompatible.

    Example:
        >>> model = MyTransformerModel()
        >>> load_trained_lightweight(model, "checkpoints/kv_cache_weights.pth", load_kv=True)
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


def prime_factors(n):
    """
    Computes the prime factors of a given integer.

    This function returns all prime factors of the input integer `n`, including repeated factors.
    For example, `prime_factors(12)` returns `[2, 2, 3]`.

    Args:
        n (int): The integer to factorize. Must be greater than 1.

    Returns:
        List[int]: A list of prime factors of `n`, in ascending order, including multiplicities.

    Raises:
        ValueError: If `n` is less than or equal to 1.
    """
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
    """
    Finds the closest valid number of target features that evenly divides the input dimension.

    This function searches both downward and upward from the given target_features to find
    the nearest number (greater than or equal to 3) that divides input_dim without a remainder.
    Preference is given to the lower value in case of a tie.

    Args:
        input_dim (int): The original input dimension (must be a positive integer).
        target_features (int): The desired number of output features.

    Returns:
        int: The closest valid number of features that evenly divides the input dimension.

    Raises:
        ValueError: If no valid target can be found in the search range.
    """
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
    Computes convolution parameters based on the prime factorization of the input dimension.

    This function determines kernel sizes and a compression rate to reduce an input dimension
    down to a specified number of target features. It ensures the reduction follows a structured
    pattern based on the factorization of the input and target dimensions.

    Args:
        input_dim (int): The original input dimension (must be a positive integer).
        target_features (int, optional): The desired number of features after compression.
            Defaults to 8.

    Returns:
        tuple: A tuple containing:
            - kernel_size_1 (int): The size of the first convolutional kernel.
            - kernel_size_2 (int): The size of the second convolutional kernel.
            - compression_rate (int): The factor by which the input is compressed.
            - hidden_channels (int): Suggested number of hidden channels for intermediate layers.

    Raises:
        ValueError: If input_dim is not divisible by the target feature dimension
            (after adjustment).
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
