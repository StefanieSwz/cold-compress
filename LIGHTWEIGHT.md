# Lightweight: Trainable KV Cache Compression for Transformer Models

This repository provides the implementation of **Lightweight**, a trainable, feature-based method for key–value (KV) cache compression in decoder-only transformer models.  
Developed as part of a Master’s thesis, Lightweight addresses the growing memory overhead caused by KV caching during inference. KV caching is commonly used in autoregressive large language models (LLMs) to avoid redundant attention computations by storing intermediate representations. However, its memory usage scales linearly with sequence length and model size, making efficient compression increasingly important for improving inference on resource-constrained hardware. Lightweight contributes to this active area of research by learning which tokens to retain based on token-level features.

Lightweight introduces a per-head scoring module that learns which tokens to retain based on a broad set of token-level features, including attention statistics, vector norms, positional cues, and token profiles. A small MLP computes these scores independently for each attention head, and the scoring logic can be relaxed to be fully differentiable to enable gradient-based training.  
Once trained, the scoring module can be integrated as a plug-in during inference without modifying or retraining the base model.

This documentation covers installation, usage, supported models, and configuration options for running and adapting the Lightweight method.

---

## Lightweight

The `LightweightKVCache` module implements a per-head, trainable token scoring mechanism for KV cache compression in decoder-only transformers.

### Core Structure

Each attention head is assigned a lightweight scoring model, either:

- a **linear layer**, or  
- a **multi-layer perceptron (MLP)**,  

as selected via the `--model_type` argument.

These models take as input a concatenated vector of token-level features and output a scalar importance score per token, used to decide which tokens are retained in the cache.

### Key Methods

- **`__init__`**:  
  Initializes internal buffers to hold token-level features such as norms, attention statistics, convolution outputs, and profiling flags. Also sets up the scoring models (MLP or linear) per head.

- **`update_state(...)`**:  
  Populates and updates the feature buffers during both prefill and generation phases. This includes:
  - Computing norms (`vector_norm`), coefficient of variation (`vector_cv`), z-scores (`vector_z_score`)
  - Storing cumulative attention scores (`attn_score`)
  - Applying 1D convolutions on selected features (`convolution_features`) and stores the multidimensional feature in buffer
  - Boolean token profiling for special and punctuation tokens

  The only feature not affected by `update_state(...)` is normalized positon, as it is computed on the fly during `_token_importances(...)`.

- **`_token_importances(...)`**:  
  Loads all configured features, concatenates them into a feature vector per token, and computes importance scores using the per-head scoring models. These scores are used in `_eviction_idx` to evict the least important tokens from the cache.

### Feature Inputs

Features are selected via `--feature_selection` and `--convolution_features`. The following features are supported:

| Feature           | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `attn_score`      | Cumulative attention score per token across heads                       |
| `vector_norm`     | L2 norm of key, value, query, and embedding vectors                     |
| `vector_cv`       | Coefficient of variation (std/mean) across token dimensions             |
| `vector_z_score`  | Max-deviation z-score across token dimensions                           |
| `token_profiling` | Boolean indicators for special or punctuation tokens                    |
| `normalized_pos`  | Normalized token position relative to sequence length                   |
| `convolution`     | 1D convolutions over embedding/key/value/query vectors                  |

### Integration

Once trained, the `LightweightKVCache` module can be used during inference by specifying:

- `--cache_strategy lightweight`
- `--prompt_compression_strategy lightweight`
- `--trained_weights <path_to_weights>` (or `--lightweight_model_version <version>` when using the evaluation script)

This integrates the trained scoring module without modifying the base transformer model.  

---

## Quickstart

### Installation

```bash
pip install -r requirements.txt
bash scripts/prepare_<model>.sh
```

Replace `<model>` with the name of the model family, e.g., `llama32` or `qwen2_05`, to be found in the `script` folder.

Lightweight supports any decoder-only transformer model for which a configuration dictionary is defined in `model.py`.

Example supported models:

* Meta LLaMA 3.2 3B `llama32`
* Qwen 2 0.5B/1.5B `qwen2_05`/ `qwen2_15`

### Training

The `train_lightweight.py` script provides a complete pipeline to train the Lightweight scoring module on top of a frozen decoder-only transformer model. Only the lightweight scoring layers are trained, while the backbone LLM remains untouched.

#### What the Script Does

The script covers all core components needed for training:

- **Model Initialization**  
  Loads a pretrained LLM checkpoint and configures the lightweight modules (MLP or linear) per attention head using the `--model_type` argument. The transformer weights remain frozen.

- **Tokenization & Data Loading**  
  Uses a prompt-aware tokenizer and wraps the [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset with a custom `PromptIterableDataset`. Prompts are tokenized in chat format. A fixed number of samples (`--num_samples`) is selected, and examples, where their prompt is exceed the maximum sequence length (`--max_seq_length`) are filtered out, since they cannot be used for training. Usable examples are truncated.

- **Feature Selection**  
  Token-level features are selected via `--feature_selection`, optionally including 1D convolutions over keys, values, queries, or embeddings (`--convolution_features`). These features are concatenated and passed to the scoring module.

- **Training Loop**  
  Performs training using cross-entropy loss on assistant outputs. Supports gradient accumulation (`--gradient_accumulation_steps`) to simulate larger batch sizes with minimal GPU memory. Training and validation loss are logged to [Weights & Biases](https://wandb.ai/).

- **Model Saving & Tracking**  
  After completing all epochs, the learned scoring module is saved and logged as an artifact to Weights & Biases, including metadata such as model version, feature configuration, and training statistics.

- **Optional Post-Evaluation**  
  If `--evaluate` is passed, the script automatically launches `eval.py` with matching configuration and the correct model artifact.

#### Key Properties

- **Frozen Backbone**  
  The transformer model remains fixed during training — only lightweight scoring layers are updated.

- **Flexible Feature Config**  
  Modular feature selection enables extensive experimentation (e.g., attention scores, norms, token profiles).

- **Built-in Logging & Evaluation**  
  Full support for Weights & Biases logging and optional automatic evaluation for reproducibility and tracking.

### Example 

```bash
python train_lightweight.py \
  --checkpoint_path checkpoints/qwen/.../model.pth \
  --feature_selection attn_score vector_norm convolution \
  --convolution_features query
  --epochs 10
  --num_samples 5000
  --gradent_accumulation_steps 150
  --learning_rate 5e-3
```

See `add_train_arguments()` in the code for all available options.

---

### Evaluation

#### Singel Evaluation

Once training is complete, you can evaluate the performance of the trained Lightweight compression model using:

```bash
python eval.py \
  --task ultrachat \
  --cache_strategy lightweight \
  --prompt_compression_strategy lightweight \
  --lightweight_model_version v200 \
  --global_tokens 4 \
  --recent_window 10 \
  --num_samples 700 \
  --max_cache_length 0.5 \
  --use_wandb
```

**Important**:
The argument `--lightweight_model_version` loads the trained Lightweight model from Weights & Biases and automatically restores all internal configuration details, including:

* Token-level features used during training
* Model type (MLP or linear)
* Convolution settings
* Other compression parameters

All other flags like `--task`, `--num_samples`, or `--max_cache_length` are task- or evaluation-specific and do **not** affect the internal scoring logic.

The reported quality metrics (e.g., ROUGE, BERTScore) depend on the specified `--task`.

#### Parallel Evaluation

To benchmark multiple Lightweight model versions across different compression levels and tasks, you can run parallelized evaluations using the `parallelize_evals_lightweight.py` script:

```bash
python parallelize_evals_lightweight.py \
  --lightweight_model_versions v200 v201 v202 \
  --tasks ultrachat gsm8k \
  --cache_sizes 0.75 0.5 0.25 0.1 0.05 \
  --num_samples 700 \
  --num_gpus 8
```

**Important**:

* Runs evaluation for each `(version, cache_size, task)` combination.
* Automatically dispatches jobs across the specified number of GPUs (one job at once per GPU).
* The script will log results via Weights & Biases, including compression metrics, accuracy scores, and runtime, one run per job.

---

## Project Structure

```text
.
├── train_lightweight.py         # Entry point for training
├── eval.py                      # Evaluation script
├── model.py                     # Model registry and config dicts
├── cache.py                     # Lightweight compression classe
├── scripts/                     # Model setup scripts 
├── requirements.txt             # Required packages
├── LIGHTWEIGHT.md               # Lightweight-specific documentation
└── README.md                    # General Project documentation
```

---

## Contributions

This implementation was developed by **Stefanie Schwarz** as part of a Master's thesis on memory-efficient inference through KV cache compression.

For questions, suggestions, or collaboration, feel free to reach out via st_schwarz@outlook.de ([GitHub](https://github.com/StefanieSwz)).
