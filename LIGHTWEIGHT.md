# Lightweight: Trainable KV Cache Compression for Transformer Models

This repository provides the implementation of **Lightweight**, a trainable, feature-based method for key–value (KV) cache compression in decoder-only transformer models.  
The method was developed as part of a Master's thesis to address the growing memory overhead of autoregressive inference in large language models (LLMs), where KV caching—used to avoid redundant attention computations—scales linearly with sequence length and model size.

Lightweight introduces a per-head scoring module that learns which tokens to retain based on a broad set of token-level features, including attention statistics, vector norms, positional cues, and token profiles. A small MLP computes these scores independently for each attention head, and the scoring logic can be relaxed to be fully differentiable to enable gradient-based training.  
Once trained, the scoring module can be integrated as a plug-in during inference without modifying or retraining the base model.

This documentation covers installation, usage, supported models, and configuration options for running and adapting the Lightweight method.

---

## Lightweight

The `Lightweight` module computes token retention scores for KV cache compression based on learned token-level features.  
It replaces fixed heuristics (e.g. attention magnitude or key norm) with a compact, trainable scoring model that operates independently per attention head.

Each head is assigned a small model through the `--model_type` argument:

- a **linear layer**, or
- a **multi-layer perceptron (MLP)**

These models take as input a concatenated vector of token-level features and output a scalar **importance score** per token.  

Available Features are:

| Feature           | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `attn_score`      | Cumulative attention score per token across heads                       |
| `vector_norm`     | L2 norm of key, value, query, and embedding vectors                     |
| `vector_cv`       | Coefficient of variation (std/mean) across token dimensions             |
| `vector_z_score`  | Max-deviation z-score across token dimensions                           |
| `token_profiling` | Boolean indicators for special or punctuation tokens                    |
| `normalized_pos`  | Normalized token position relative to sequence length                   |
| `convolution`     | Feature projection using 1D convolutions (single or double convolution) |

The features are included in the training via the `--feature_selection` and `--convolution_features` arguments.

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

---

### Training

The `train_lightweight.py` script trains the Lightweight KV cache compression module using a fixed pretrained LLM (e.g. LLaMA or Qwen).

This launches a full training pipeline consisting of:

* **Dataset loading & preprocessing**: Uses [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) with a prompt-based tokenizer (`PromptIterableDataset`). Samples are filtered to meet the `max_seq_length` requirement and are teacher-forced during training.
* **Feature extraction**: Computes token-level features like attention scores, vector norms, z-scores, position, and convolutional embeddings.
* **Per-head scoring model training**: Each attention head is assigned a small model (linear or MLP) that maps the extracted features to token importance scores.
* **Gradient accumulation**: Supports batch size 1 with accumulation (via `--gradient_accumulation_steps`) for memory efficiency.
* **Loss**: Cross-entropy loss is used over the full prompt, comparing predictions to the ground truth assistant responses.
* **Validation**: At each epoch, average validation loss is computed and logged via [Weights & Biases](https://wandb.ai/).
* **Model saving**: After training, the scoring weights are saved as a separate artifact and optionally evaluated.

**Important**:

* The base language model remains **frozen** — only the lightweight scoring modules are updated.
* All models are stored and tracked via Weights & Biases, including automatic artifact versioning.
* The training script supports optional post-training evaluation using `eval.py` by passing `--evaluate`.

Example usage with evaluation:

```bash
python train_lightweight.py \
  --checkpoint_path checkpoints/qwen/.../model.pth \
  --feature_selection attn_score vector_norm convolution \
  --epochs 10 
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

**Key Point**:
The argument `--lightweight_model_version` loads the trained Lightweight model from Weights & Biases and automatically restores all internal configuration details, including:

* Token-level features used during training
* Model type (MLP or linear)
* Convolution settings
* Other compression parameters

All other flags like `--task`, `--num_samples`, or `--max_cache_length` are task- or evaluation-specific and do **not** affect the internal scoring logic.

This script reports generation quality metrics (e.g., ROUGE, BERTScore) as well as memory and runtime benchmarks under compression.

--- 

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

**What this does**:

* Runs evaluation for each `(version, cache_size, task)` combination.
* Automatically dispatches jobs across the specified number of GPUs.
* Ensures that the `lightweight_model_version` restores the correct scoring model and internal configuration (features, convolution, etc.) for each version.
* The script will log results via Weights & Biases, including compression metrics, accuracy scores, and runtime.

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

For questions, suggestions, or collaboration, feel free to reach out via [GitHub](https://github.com/StefanieSwz).
