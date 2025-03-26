#!/bin/bash

set -e

# MODELS=(
# 	"checkpoints/meta-llama/Llama-3.2-3B-Instruct/model.pth"
# 	"checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth"
# ) # in model version, just a reminder
NUM_SAMPLES=700 # keep
CACHE_SIZES="0.75 0.5 0.25 0.1 0.05" # keep
TASKS="ultrachat"
LIGHTWEIGHT_MODEL_VERSIONS="v149 v151 v152 v153 v154 v155 v156 v157 v158 v159" # Version defines the model, rest must fit to the model version


python parallelize_evals_lightweight.py \
	--lightweight_model_versions $LIGHTWEIGHT_MODEL_VERSIONS \
	--tasks $TASKS \
	--cache_sizes $CACHE_SIZES \
	--num_samples $NUM_SAMPLES \
	--num_gpus 8 \
	--add_full

