import torch
import torch.nn as nn
import pdb
from abc import ABC, abstractmethod


class PromptCompressor(ABC):
    def __init__(self, head_specific, **kwargs) -> None:
        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.head_specific = head_specific
        assert (
            self.is_compatible()
        ), f"Prompt compressor ({self.__class__.__name__}) is not compatible with the chosen cache strategy."

    def _recent_global_mask(self, input_pos):
        seq_len = input_pos.shape[-1]
        return torch.logical_or(
            input_pos < self.global_tokens,
            input_pos >= seq_len - self.recent_window,
        )

    def _keep_idxs(self, priority):
        return (
            priority.topk(self.max_cache_length, dim=-1)
            .indices.sort(dim=-1)
            .values.squeeze(0)
        )

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        # Assign a score to each token in the prompt to determine filtering priority
        priority = self._token_importances(input_pos, k_val, v_val, **kwargs)

        # Get the self.max_cache_length indices with the highest priority
        keep_idxs = self._keep_idxs(priority)

        # Compress the prompt based on these indices
        k_val, v_val = self._filter_kv(keep_idxs, k_val, v_val)

        return (
            keep_idxs,
            k_val,
            v_val,
            self._update_state(keep_idxs, input_pos, **kwargs),
        )

    def _update_state(self, keep_idxs, input_pos, **kwargs):
        # [Optional] Over-write to return attention scores corresponding to keep_idxs
        return None

    @abstractmethod
    def _filter_kv(self, keep_idxs, k_val, v_val):
        raise NotImplementedError

    @abstractmethod
    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def is_compatible(self) -> bool:
        raise NotImplementedError


class PromptCompressorHeadConstant(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return True

    def _filter_kv(self, keep_idxs, k_val, v_val):
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return k_val, v_val


class PromptCompressorHeadSpecific(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return self.head_specific

    def _filter_kv(self, keep_idxs, k_val, v_val):
        keep_idxs_rep = keep_idxs.view(1, -1, self.max_cache_length, 1).expand(
            -1, -1, -1, k_val.shape[-1]
        )
        k_val = k_val.gather(2, keep_idxs_rep)
        v_val = v_val.gather(2, keep_idxs_rep)
        return k_val, v_val


class PromptCompressorFull(PromptCompressorHeadConstant):
    """
    This is a dummy (pass through) method which returns its inputs
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return True

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        return input_pos, k_val, v_val, None  # noop

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        raise Exception("This method should not be called!")


class PromptCompressorRandom(PromptCompressorHeadConstant):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        seq_len = input_pos.shape[-1]
        save_mask = self._recent_global_mask(input_pos)
        priority = input_pos.masked_fill(save_mask, seq_len)
        # Assign positions in the middle uniform low priority
        priority = priority.masked_fill(~save_mask, -seq_len)
        # Add random noise to randomize the middle priorities
        priority += torch.randperm(seq_len, device=priority.device)
        return priority


class PromptCompressorRecentGlobal(PromptCompressorHeadConstant):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

        window_size = self.max_cache_length - self.global_tokens
        assert (
            window_size > 0
        ), f"Number of global tokens ({self.global_tokens}) cannot exceed the max cache length ({self.max_cache_length})"

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        # Assign Global tokens to max seq length so they are always saved
        return input_pos.masked_fill(
            input_pos < self.global_tokens, input_pos.shape[-1]
        )


class PromptCompressorLightweight(PromptCompressorHeadSpecific):
    """
    Use SnapKV to compress the prompt
    Based on the pseudo code on Page 7 of https://arxiv.org/abs/2404.14469
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

        # Initialize lightweight models for computing token scores.
        if kwargs["model_type"] == "linear":
            self.models = nn.ModuleList(
                [nn.Linear(3, 1).to(kwargs["dtype"]) for _ in range(kwargs["n_heads"])]
            )
        elif kwargs["model_type"] == "mlp":
            self.models = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1)).to(
                        kwargs["dtype"]
                    )
                    for _ in range(kwargs["n_heads"])
                ]
            )
        else:
            raise ValueError("Unsupported model_type. Use 'linear' or 'mlp'.")

        # Initialize the models' parameters with random weights.
        # Trained weights are loaded from model perspective
        with torch.no_grad():
            for model in self.models:
                for name, param in model.named_parameters():
                    if "weight" in name:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
                    elif "bias" in name:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
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
        seq_len = input_pos.shape[-1]
        attn = kwargs["attn"]
        attn_scores = attn.mean(dim=2)
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)
        value_norm = torch.linalg.vector_norm(v_val, ord=2, dim=-1)

        # Extract features from the entire cache
        features = torch.stack(
            [
                key_norm,  # [batch_size, n_heads, seq_len]
                value_norm,  # [batch_size, n_heads, seq_len]
                attn_scores,  # [batch_size, n_heads, seq_len]
            ],
            dim=-1,  # Resulting shape: [batch_size, n_heads, seq_len, 3]
        )

        # Initialize scores with -infinity for all cache slots
        priority = torch.full(
            (features.shape[0], features.shape[1], seq_len),
            float("-inf"),
            dtype=features.dtype,
            device=features.device,
        )

        # Compute scores for the current valid positions using the lightweight models
        for head_idx, model in enumerate(self.models):
            # Compute scores for valid positions and update the scores tensor
            priority[:, head_idx, :] = model(features[:, head_idx, :]).squeeze(-1)

        # Give high score to global and recent tokens
        save_mask = self._recent_global_mask(input_pos).view(1, 1, -1)
        priority = priority.masked_fill(save_mask, float("inf"))

        return priority

    def _update_state(self, keep_idxs, input_pos, **kwargs):
        seq_len = input_pos.shape[-1]
        # Return average attention across prompt to insert into KV Cache's attention history tracker
        cum_attn = kwargs["attn"].sum(dim=2) / (seq_len - input_pos)
        cum_attn = cum_attn.gather(2, keep_idxs.view(1, -1, self.max_cache_length))
        return cum_attn


class PromptCompressorHeavyHitter(PromptCompressorHeadSpecific):
    """
    Use SnapKV to compress the prompt
    Based on the pseudo code on Page 7 of https://arxiv.org/abs/2404.14469
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

        self.kernel_size = 5
        self.observation_len = 16

        # Pooling layer to smooth out the attention distribution
        # Feel free to remove this or optimize the kernel size
        self.pool = torch.nn.AvgPool1d(
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            ceil_mode=False,
            count_include_pad=False,
        )

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        attn = kwargs["attn"]
        seq_len = input_pos.shape[-1]
        obs_len = min(self.observation_len, seq_len)

        priority = attn[:, :, -obs_len:, :].mean(dim=2)
        prev_shape = priority.shape

        # We'll be returning the attention history so we need to keep a copy before it's modified
        priority = self.pool(priority)
        assert (
            priority.shape == prev_shape
        ), f"Pooling operation should not change the dimension: {prev_shape} -> {priority.shape}"
        priority[:, :, -obs_len:] = 1.0  # Ensure the observation window is selected
        priority[:, :, : self.global_tokens] = (
            1.0  # Ensure the global tokens are selected
        )
        return priority

    def _update_state(self, keep_idxs, input_pos, **kwargs):
        seq_len = input_pos.shape[-1]
        # Return average attention across prompt to insert into KV Cache's attention history tracker
        cum_attn = kwargs["attn"].sum(dim=2) / (seq_len - input_pos)
        cum_attn = cum_attn.gather(2, keep_idxs.view(1, -1, self.max_cache_length))
        return cum_attn


class PromptCompressorL2(PromptCompressorHeadSpecific):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        # We want to prioritize the lowest L2 norm tokens so we negate the L2 norm
        priority = -torch.linalg.vector_norm(k_val, ord=2, dim=-1)

        # Give low score to global and recent tokens
        save_mask = self._recent_global_mask(input_pos).view(1, 1, -1)
        priority = priority.masked_fill(save_mask, float("inf"))

        return priority


class PromptCompressorKeepItOdd(PromptCompressorHeadConstant):
    """
    A toy example of a prompt compressor that keeps the odd positions indices of the prompt.
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def _token_importances(self, input_pos, k_val, v_val, **kwargs):
        seq_len = input_pos.shape[-1]
        # Compute odd indices from keep_idxs to input_pos.shape[-1] - window
        priority = input_pos.masked_fill(
            self._recent_global_mask(input_pos), seq_len * 2
        )

        # Lower the priority of even tokens
        priority[input_pos % 2 == 0] -= seq_len

        return priority


def get_prompt_compressor_constructor(strategy):
    if strategy == "full":
        return PromptCompressorFull
    if strategy == "recent_global":
        return PromptCompressorRecentGlobal
    elif strategy == "heavy_hitter":
        return PromptCompressorHeavyHitter
    elif strategy == "l2":
        return PromptCompressorL2
    elif strategy == "random":
        return PromptCompressorRandom
    elif strategy == "keep_it_odd":
        return PromptCompressorKeepItOdd
    elif strategy == "lightweight":
        return PromptCompressorLightweight
    else:
        raise ValueError(f"Unknown prompt compression strategy: {strategy}")
