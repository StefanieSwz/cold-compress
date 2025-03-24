# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os
import time
import argparse
import math
import json
import contextlib
from pathlib import Path
from typing import Optional, List
from collections import defaultdict, Counter
import shutil
import itertools
import tempfile

import regex as re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv
import wandb

import torch
import torch._dynamo.config
import torch._inductor.config

from cache import (
    add_cache_arguments,
    cache_compatibility,
    get_cache_constructor,
)
from model import Transformer, ModelArgs
from generation_utils import (
    add_generation_arguments,
    compile_funcs,
    compute_max_seq_length,
    device_sync,
    get_cache_stats,
    merge_cache_config,
    reset_caches,
    setup_caches,
)
from tokenizer import encode, TokenizerInterface

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
DEBUG_COMPILE = False
if DEBUG_COMPILE:
    import logging

    level = logging.DEBUG
    torch._logging.set_logs(dynamo=level, inductor=level)
    torch._dynamo.config.verbose = True

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tokenizer import get_tokenizer
from generation_utils import load_model, generate
from task import TASK_MAPPING, AutoTask

load_dotenv()


def flatten_dict(in_dict: dict) -> dict:
    out_dict = {}
    for k, v in in_dict.items():
        if type(v) == dict:
            for kk, vv in v.items():
                out_dict[f"{k}_{kk}"] = vv
        else:
            out_dict[k] = v
    return out_dict


def compress_list(l):
    if len(l) < 3:
        return l
    else:
        counter = Counter(l)
        return [f"{k}x{v}" for k, v in counter.items()]


def args_to_str(args):
    if "debug" in args.cache_strategy[0]:
        debug_suffix = "__debug"
        cache_strategy = [
            re.sub(r"debug_+", "", cs).strip() for cs in args.cache_strategy
        ]
    else:
        cache_strategy = args.cache_strategy
        debug_suffix = ""
    RELEVANT_CACHE_KWARGS = list(
        sorted(
            set(
                itertools.chain(
                    *[get_cache_constructor(cs)[1] for cs in cache_strategy]
                )
            )
        )
    )

    def process_num(n):
        # Return integer floats as "1" not 1.0
        # Otherwise, no op
        if type(n) == float and int(n) == n:
            return int(n)
        return n

    RELEVANT_CACHE_KWARGS.append("cache_length_pattern")
    RELEVANT_CACHE_KWARGS.append("cache_strategy_pattern")
    if hasattr(args, "attn_top_k") and args.attn_top_k != 1.0:
        RELEVANT_CACHE_KWARGS.append("attn_top_k")

    args_dict = vars(args).copy()

    # Hybrid Strategies will be too long to save in a file name so just need to pick the strategy
    if "hybrid_strategies" in args_dict:
        args_dict["hybrid_strategies"] = [
            x["strategy"] for x in args_dict["hybrid_strategies"]
        ]

    return (
        "__".join(
            sorted(
                [
                    (
                        f"{k}="
                        + ",".join(compress_list([str(process_num(m)) for m in v]))
                        if type(v) == list
                        else f"{k}={process_num(v)}"
                    )
                    for k, v in args_dict.items()
                    if k in RELEVANT_CACHE_KWARGS
                ]
            )
        )
        + debug_suffix
    )


def serialize_value(v):
    """Convert non-serializable objects to JSON-compatible types."""
    if isinstance(v, Path):
        return str(v)  # Convert Path to string
    elif isinstance(v, torch.Tensor):
        return v.tolist()  # Convert Tensor to list
    elif isinstance(v, argparse.Namespace):
        return vars(v)  # Convert Namespace to dict
    elif isinstance(v, set):
        return list(v)  # Convert set to list
    elif isinstance(v, ModelArgs):
        return v.to_dict()
    return v  # Keep other values as they are


def run_task(
    args: argparse.Namespace,
    task: AutoTask,
    model: Transformer,
    prefill: callable,
    decode_one_token: callable,
    tokenizer: TokenizerInterface,
    is_chat: bool = False,
    profile: Optional[Path] = None,
    feed_long_prompts=False,
    decode_first_token=False,
    device=default_device,
    cache_kwargs: dict = {},
    use_tp: bool = False,
    rank: int = None,
    terminator_ids: List[int] = None,
):
    aggregate_metrics = defaultdict(list)
    predictions = []
    all_probs = []
    task_metrics = {}

    test = task.get_test()

    if len(test) == 0:
        print(
            f"No test data found for {task.__class__.__name__}. Skipping. Possibly all filtered out by tokenizer for being too long."
        )
        return None, None, None

    prompts = test["prompt"]

    inputs = [
        encode(tokenizer, prompt, device="cpu", is_chat=is_chat)
        for prompt in tqdm(prompts, desc="Encoding Prompts")
    ]

    if task.requires_perplexity:
        assert (
            len(test["labels"][0]) == 1
        ), "Only one label supported for perplexity tasks"
        label_ids = [
            encode(tokenizer, label[0], device="cpu", is_chat=False, bos=False)
            for label in tqdm(test["labels"], desc="Encoding Labels")
        ]
        _, max_seq_length = compute_max_seq_length(model, inputs, label_ids, 0)
    else:
        label_ids = None
        _, max_seq_length = compute_max_seq_length(model, inputs, None, task.max_tokens)

    # Estimate median sequence length
    median_seq_length = int(np.median([len(i) for i in inputs]) + task.max_tokens / 2)

    target_length = (
        max_seq_length
        if any([x in {"full", "hybrid"} or "debug" in x for x in args.cache_strategy])
        else median_seq_length
    )

    task_cache_kwargs = setup_caches(
        model, tokenizer, device, target_length, cache_kwargs.copy()
    )

    for i in tqdm(range(len(inputs))):
        input = inputs[i].to(device)
        next_tokens = None if label_ids is None else label_ids[i].to(device)
        prompt_length = input.size(0)
        max_new_tokens = min(task.max_tokens, max_seq_length - prompt_length)
        assert max_new_tokens > 0, f"Prompt too long for model: {prompt_length}"

        device_sync(device=device)  # MKG

        if not profile or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, probs, perf_stats = generate(
                model,
                input,
                prefill,
                decode_one_token,
                max_new_tokens=max_new_tokens,
                next_tokens=next_tokens,
                terminator_ids=terminator_ids if next_tokens is None else None,
                attn_top_k=args.attn_top_k,
                feed_long_prompts=feed_long_prompts,
                decode_first_token=decode_first_token,
            )

        for k, v in perf_stats.items():
            aggregate_metrics[k].append(v)

        if next_tokens is not None:
            nll = -torch.tensor(
                [
                    torch.log(probs[j][next_tokens[j]])
                    for j in range(next_tokens.size(0))
                ]
            )
            for k in range(500, len(nll), 500):
                aggregate_metrics[f"ppl@{k}"].append(
                    float(torch.exp(torch.mean(nll[:k])).item())
                )
            aggregate_metrics["ppl"].append(float(torch.exp(torch.mean(nll)).item()))

        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG

        cache_stats = get_cache_stats(model, prompt_length, perf_stats["decode_tokens"])
        for k, v in cache_stats.items():
            aggregate_metrics[k].append(v)

        if (
            not task.requires_perplexity
        ):  # Perplexity tasks don't decode from model so don't save predictions
            # Decode: remove EoT and prompt
            end = y.size(0)
            if y[-1] in terminator_ids:
                end = -1
            pred = tokenizer.decode(y[prompt_length:end].tolist())

            if args.debug:
                print(f"Prediction: {pred}")

            predictions.append(pred)
            if task.requires_logits:
                all_probs.append(
                    {k: v for k, v in zip(tokenizer.get_vocab(), probs[-1].tolist())}
                )

        # Reset KV Cache state
        reset_caches(model)

    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['total_toks_per_sec'])).item():.2f}"
    )
    max_mem_gb = torch.cuda.max_memory_reserved() / 1e9
    print(f"Memory used: {max_mem_gb} GB")
    task_metrics["max_memory_gb"] = max_mem_gb

    for k, v in aggregate_metrics.items():
        task_metrics[k] = sum(v) / len(v)

        # For toks_per_sec, we also want to report the average of the highest 10% toks/second
        # This is useful to get a sense of toks / second without the one-time impact of compilation
        if "toks_per_sec" in k:
            # Useful to save toks_per_sec for each example for better understanding of how it changes over time with compile
            task_metrics[k] = v
            # Also save the top 10% average (likely unaffected by compile)
            v.sort()
            cutoff = math.ceil(len(v) / 10)
            task_metrics[f"{k}_top_10p"] = sum(v[-cutoff:]) / cutoff

        if k == "total_seconds":
            task_metrics[f"{k}_min"] = min(aggregate_metrics[k])
            task_metrics[f"{k}_max"] = max(aggregate_metrics[k])
            task_metrics[f"{k}_median"] = float(np.median(aggregate_metrics[k]))

    if task.requires_perplexity:
        pred_df = None
    else:
        pred_units = all_probs if task.requires_logits else predictions
        task_metrics.update(flatten_dict(task.test_metrics(pred_units)))
        pred_df = pd.DataFrame({"prompt": prompts, "prediction": predictions})

    return task_metrics, pred_df, task_cache_kwargs


def main(
    args: argparse.Namespace,
    tasks: List[str],
    debug: bool = False,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    profile: Optional[Path] = None,
    compile=True,
    feed_long_prompts=False,
    decode_first_token=False,
    device=default_device,
    cache_kwargs: dict = {},
    out_dir: Path = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""

    global print
    from tp import maybe_init_dist

    if args.use_wandb:
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
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=vars(args),
            group="evaluation",
        )
        # registry_path = f"{wandb_account}/wandb-registry-model/{wandb_model_registry}"
        # execute eval directly after training
        if isinstance(cache_kwargs["cache_strategy"], str):
            cache_strategy_list = [cache_kwargs["cache_strategy"]]
        else:
            cache_strategy_list = cache_kwargs["cache_strategy"]
        if isinstance(cache_kwargs["prompt_compression_strategy"], str):
            prompt_strategy_list = [cache_kwargs["prompt_compression_strategy"]]
        else:
            prompt_strategy_list = cache_kwargs["prompt_compression_strategy"]
        if (
            "lightweight" in cache_strategy_list
            or "lightweight" in prompt_strategy_list
        ):
            model_ref = "lightweight_model:" + args.lightweight_model_version
            model_artifact = wandb.use_artifact(model_ref, type="model")
            print("Artifact contents:", model_artifact.file())
            temp_dir_obj = (
                tempfile.TemporaryDirectory()
            )  # Keep reference, don't use `with`
            temp_dir = temp_dir_obj.name  # Get the path
            model_artifact.download(root=temp_dir)  # Download to a temp directory
            model_path = os.path.join(temp_dir, "model.pth")
            cache_kwargs["trained_weights"] = model_path
            print(f"Using lightweight model from {model_path}")
            # get configs
            run_config = model_artifact.logged_by().config

            print("Metadata associated with model version:")
            checkpoint_path = Path(run_config.get("checkpoint_path"))
            print(f"Checkpoint path: {checkpoint_path}")
            cache_kwargs["model_type"] = run_config.get("model_type", None)
            cache_kwargs["vector_convolution"] = run_config.get(
                "vector_convolution", None
            )
            cache_kwargs["feature_selection"] = run_config.get(
                "feature_selection", None
            )
            cache_kwargs["convolution_features"] = run_config.get(
                "convolution_features", None
            )
            print(f"Model type: {cache_kwargs['model_type']}")
            print(f"Vector convolution: {cache_kwargs['vector_convolution']}")
            print(f"Feature selection: {cache_kwargs['feature_selection']}")
            print(f"Convolution features: {cache_kwargs['convolution_features']}")

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        # If there's no tokenizer.model, try to load the tokenizer from the parent directory
        # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
        tokenizer_path = checkpoint_path.parent

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_chat = (
        "chat" in str(checkpoint_path).lower()
        or "instruct" in str(checkpoint_path).lower()
    )

    print("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    if (
        cache_kwargs["cache_strategy"] == "hybrid"
        or cache_kwargs["cache_strategy"] == "lightweight"
    ):
        # We need to pass the special and punctuation token ids to the cache via cache_kwargs
        cache_kwargs["token_ids"] = {
            "special": tokenizer.special_ids(),
            "punctuation": tokenizer.punctuation_ids(),
        }

    terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)

    task_kwargs = {
        "model_max_length": model.config.max_length,
        "num_samples": args.num_samples,
        "tokenizer": tokenizer.encode_prompt if is_chat else tokenizer.encode,
        "seq_length": args.seq_length,
    }
    if tasks == ["all"]:
        # Evaluate all tasks
        tasks = list(TASK_MAPPING.keys())
    eval_tasks = {task: AutoTask.from_name(task, **task_kwargs) for task in tasks}

    task_metrics = defaultdict(dict)
    args_fn = out_dir / "args.json"
    all_out_fn = out_dir / "all_metrics.json"
    for task_name, task in eval_tasks.items():
        print(f"Running task {task_name} ...")
        task_out_fn = out_dir / f"{task_name}_metrics.json"
        task_args_out_fn = out_dir / f"{task_name}_args.json"
        pred_out_fn = out_dir / f"{task_name}_predictions.csv"
        if task_out_fn.exists() and not cache_kwargs["overwrite"]:
            print(f"Task {task_name} already evaluated. Skipping.")
            with open(task_out_fn, "r") as fd:
                task_metrics[task_name] = json.load(fd)
        else:
            prefill, decode_one_token = compile_funcs(compile)
            task_metrics[task_name], predictions, task_args = run_task(
                args,
                task,
                model,
                prefill,
                decode_one_token,
                tokenizer,
                is_chat,
                profile,
                feed_long_prompts,
                decode_first_token,
                device,
                cache_kwargs,
                use_tp,
                rank,
                terminator_ids,
            )

            if task_metrics[task_name] is None:
                continue

            if predictions is not None:
                predictions.to_csv(pred_out_fn, index=False)

            if debug:
                print(f"Results for {task_name}:")
                print(task_metrics[task_name])

            with open(task_out_fn, "w") as fd:
                print(f"Saving results for {task_name} to {task_out_fn}")
                json.dump(task_metrics[task_name], fd, indent=4)

            with open(task_args_out_fn, "w") as fd:
                print(f"Saving dynamic args for {task_name} to {task_args_out_fn}")
                # Convert Path objects to strings
                task_args_json = {k: serialize_value(v) for k, v in task_args.items()}
                json.dump(task_args_json, fd, indent=4)

            if args.use_wandb:
                artifact = wandb.Artifact(
                    name="evaluation_results",
                    type="evaluation",
                    metadata={
                        "tasks": task_name,
                        "model_checkpoint": str(checkpoint_path),
                        "features": model_artifact.metadata.get(
                            "feature_selection", None
                        ),
                    },
                )

                # Add general result files
                if args_fn.exists():
                    artifact.add_file(str(args_fn))  # Save args
                if all_out_fn.exists():
                    artifact.add_file(str(all_out_fn))  # Save all metrics

                # Add each task's results
                if task_out_fn.exists():
                    artifact.add_file(str(task_out_fn))
                if task_args_out_fn.exists():
                    artifact.add_file(str(task_args_out_fn))
                if pred_out_fn.exists():
                    artifact.add_file(str(pred_out_fn))

                # Log to W&B
                wandb.log_artifact(artifact)
                print("✅ W&B Artifact saved with all evaluation results.")

        if not args_fn.exists():
            # Only save args once and only save if we've gotten through a full eval and are ready to dump metrics
            with open(args_fn, "w") as fd:
                # Convert Path objects to strings
                cache_kwargs_json = {
                    k: serialize_value(v) for k, v in cache_kwargs.items()
                }
                json.dump(cache_kwargs_json, fd, indent=4)

    with open(all_out_fn, "w") as fd:
        json.dump(task_metrics, fd, indent=4)

    if args.use_wandb:
        if cache_kwargs["cache_strategy"] == "lightweight":
            temp_dir_obj.cleanup()
        wandb.finish()


def setup(args) -> Path:
    # sub_dir = args_to_str(args) if args.out_dir is None else args.out_dir
    sub_dir = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(__file__).parent
        / "results"
        / args.checkpoint_path.parent.name
        / "__".join(compress_list(args.cache_strategy))
        / sub_dir
    )

    print(f"Saving to {out_dir}")
    # Make out_dir and don't err out if it already exists
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists.")
        if args.overwrite:
            print(f"Removing {out_dir}.")
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_compatibility(args)

    for k, v in vars(args).items():
        print(f"{k} -> {v}")

    return out_dir


def add_eval_args(parser):
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["truthfulqa"],
        choices=list(TASK_MAPPING.keys()) + ["all"],
        help="List of tasks to be evaluated.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for results. If not specified, will be a concatenation of the program args.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode uses first 10 examples in dataset.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of examples to sample for evaluation. Defaults to None, which uses the full dataset.",
    )

    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Whether to over-write existing results if they exist.",
    )

    # Only for --tasks PG19
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Specify the number of tokens for the dataset.",
    )

    parser.add_argument(
        "--cache_config",
        type=str,
        default=None,
        help="Name of YAML file in ./cache_configs.",
    )

    parser.add_argument(
        "--decode_first_token",
        default=False,
        action="store_true",
        help="If True will truncate cache after prefill and then decode the first token.",
    )

    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="If weights and biases should be used during evaluation for loading the lightweight model or saving the eval results as artifacts.",
    )

    parser.add_argument(
        "--lightweight_model_version",
        type=str,
        default="latest",
        help="Version of the lightweight model to use.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for different KV-Cache Compression Algorithms."
    )

    add_eval_args(parser)
    add_generation_arguments(parser)
    add_cache_arguments(parser)

    args = merge_cache_config(parser.parse_args())

    if args.tasks[0] == "all":
        args.tasks = list(TASK_MAPPING.keys())
        print(f"Running all tasks: {', '.join(args.tasks)}")

    out_dir = setup(args)

    main(
        args,
        args.tasks,
        args.debug,
        args.checkpoint_path,
        args.profile,
        args.compile,
        args.feed_long_prompts,
        args.decode_first_token,
        args.device,
        cache_kwargs=vars(args),
        out_dir=out_dir,
    )
