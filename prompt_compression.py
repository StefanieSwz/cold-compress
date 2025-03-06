import torch
import torch.nn as nn
import pdb
import math
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

        self.feature_space_dim = 0  # Number of final features

        # Get feature size
        if "attn_score" in kwargs["feature_selection"]:
            self.feature_space_dim += 1
        if "vector_norm" in kwargs["feature_selection"]:
            self.feature_space_dim += 4
        if "vector_cv" in kwargs["feature_selection"]:
            self.feature_space_dim += 4
        if "vector_z_score" in kwargs["feature_selection"]:
            self.feature_space_dim += 4
        if "token_profiling" in kwargs["feature_selection"]:
            self.special_ids = torch.tensor(
                kwargs["token_ids"]["special"], dtype=torch.int32
            )

            self.punctuation_ids = torch.tensor(
                kwargs["token_ids"]["punctuation"], dtype=torch.int32
            )
            self.feature_space_dim += 2
        if "convolution" in kwargs["feature_selection"]:
            self.conv_compression_rate = 16  # must be 2^x, keep at 16, otherwise double conv does not necessarily work
            self.conv_hidden_channels = 4  # or 8
            self.conv_layers = nn.ModuleDict()

            # kwargs["convolution_features"] is a list like ["key", "value", "query", "embedding"]
            for feat in kwargs["convolution_features"]:
                if feat == "embedding":
                    result_conv_dim = kwargs["config_dim"] // self.conv_compression_rate
                else:
                    result_conv_dim = kwargs["head_dim"] // self.conv_compression_rate
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
                    self.feature_space_dim += result_conv_dim
                else:
                    print("No convolution feature initialized")
        if "normalized_pos" in kwargs["feature_selection"]:
            self.feature_space_dim += 1
        # Initialize lightweight models for computing token scores.
        if kwargs["model_type"] == "linear":
            self.models = nn.ModuleList(
                [
                    nn.Linear(self.feature_space_dim, 1).to(kwargs["dtype"])
                    for _ in range(kwargs["n_heads"])
                ]
            )
        elif kwargs["model_type"] == "mlp":
            self.lightweight_hidden_size = 16
            self.models = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.feature_space_dim, self.lightweight_hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.lightweight_hidden_size, 1),
                    ).to(kwargs["dtype"])
                    for _ in range(kwargs["n_heads"])
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
        features_to_cat = []
        x_expanded = (
            kwargs["x"].unsqueeze(1).expand(-1, k_val.shape[1], *kwargs["x"].shape[1:])
        )
        n_heads = k_val.shape[1]
        batch_size, n_query_heads, _, head_dim = kwargs["query"].shape
        group_size = n_query_heads // n_heads
        q_grouped = kwargs["query"].view(
            batch_size, n_heads, group_size, seq_len, head_dim
        )
        q_avg = q_grouped.mean(dim=2)  # [batch_size, n_heads, seq_len, head_dim]

        if "attn_score" in kwargs["feature_selection"]:
            attn = kwargs["attn"]
            attn_score = attn.mean(dim=2)
            features_to_cat.append(attn_score.unsqueeze(-1))
        if "vector_norm" in kwargs["feature_selection"]:
            key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)
            value_norm = torch.linalg.vector_norm(v_val, ord=2, dim=-1)
            query_norm = torch.linalg.vector_norm(q_avg, ord=2, dim=-1)
            embedding_norm = torch.linalg.vector_norm(x_expanded, ord=2, dim=-1)
            features_to_cat.extend(
                [
                    key_norm.unsqueeze(-1),
                    value_norm.unsqueeze(-1),
                    query_norm.unsqueeze(-1),
                    embedding_norm.unsqueeze(-1),
                ]
            )
        if "vector_cv" in kwargs["feature_selection"]:
            key_cv = torch.std(k_val, dim=-1, unbiased=False) / torch.mean(
                k_val, dim=-1
            )
            value_cv = torch.std(v_val, dim=-1, unbiased=False) / torch.mean(
                v_val, dim=-1
            )
            query_cv = torch.std(q_avg, dim=-1, unbiased=False) / torch.mean(
                q_avg, dim=-1
            )
            embedding_cv = torch.std(x_expanded, dim=-1, unbiased=False) / torch.mean(
                x_expanded, dim=-1
            )
            features_to_cat.extend(
                [
                    key_cv.unsqueeze(-1),
                    value_cv.unsqueeze(-1),
                    query_cv.unsqueeze(-1),
                    embedding_cv.unsqueeze(-1),
                ]
            )
        if "vector_z_score" in kwargs["feature_selection"]:
            key_z = (
                torch.max(k_val, dim=-1).values - torch.mean(k_val, dim=-1)
            ) / torch.std(k_val, dim=-1, unbiased=False)
            value_z = (
                torch.max(v_val, dim=-1).values - torch.mean(v_val, dim=-1)
            ) / torch.std(v_val, dim=-1, unbiased=False)
            query_z = (
                torch.max(q_avg, dim=-1).values - torch.mean(q_avg, dim=-1)
            ) / torch.std(q_avg, dim=-1, unbiased=False)
            embedding_z = (
                torch.max(x_expanded, dim=-1).values - torch.mean(x_expanded, dim=-1)
            ) / torch.std(x_expanded, dim=-1, unbiased=False)
            features_to_cat.extend(
                [
                    key_z.unsqueeze(-1),
                    value_z.unsqueeze(-1),
                    query_z.unsqueeze(-1),
                    embedding_z.unsqueeze(-1),
                ]
            )
        if "token_profiling" in kwargs["feature_selection"]:
            B, seq_len = kwargs["input_ids"].shape
            special_ids_mask = torch.isin(
                kwargs["input_ids"], self.special_ids
            )  # Shape: [B, seq_len]
            token_special_profiling = special_ids_mask.unsqueeze(1).expand(
                B, n_heads, seq_len
            )

            punct_ids_mask = torch.isin(
                kwargs["input_ids"], self.punctuation_ids
            )  # Shape: [B, seq_len]
            token_punctuation_profiling = punct_ids_mask.unsqueeze(1).expand(
                B, n_heads, seq_len
            )
            features_to_cat.extend(
                [
                    token_special_profiling.unsqueeze(-1),
                    token_punctuation_profiling.unsqueeze(-1),
                ]
            )
        if "convolution" in kwargs["feature_selection"]:
            feature_dict = {
                "embedding": x_expanded,
                "query": q_avg,
                "key": k_val,
                "value": v_val,
            }  # all (batch, n_heads, seq_len, dim/head_dim)
            for feat in kwargs["convolution_features"]:
                feat_tensor = feature_dict[feat]  # shape: [B, H, M, D]
                feat_tensor_reshaped = feat_tensor.reshape(
                    batch_size * n_heads * seq_len, 1, feat_tensor.shape[-1]
                )
                conv_out = self.conv_layers[feat](
                    feat_tensor_reshaped
                )  # shape: [N_valid, 1, F]
                conv_out_reshaped = conv_out.reshape(
                    batch_size, n_heads, seq_len, -1
                )  # shape: [B, H, M, F]
                features_to_cat.append(conv_out_reshaped)
        if "normalized_pos" in kwargs["feature_selection"]:
            normalized_pos = input_pos.float() / input_pos[-1].float()
            features_to_cat.append(
                normalized_pos.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, n_heads, -1)
                .unsqueeze(-1)
            )

        features_to_cat = [feature.to(torch.bfloat16) for feature in features_to_cat]
        features = torch.cat(features_to_cat, dim=3)

        # Initialize scores with -infinity for complete sequence len
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
        # only update attention, rest is done in kv_cache.update_state()
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
