#!/bin/bash

set -e

MODELS=(
	"checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth"
	"checkpoints/meta-llama/Llama-3.2-3B-Instruct/model.pth"

)
NUM_SAMPLES=700
CACHE_SIZES="0.75 0.5 0.25 0.1 0.05"
TASKS="ultrachat"
CACHE_CONFIGS="random l2 heavy_hitter recent_global fastgen"
MIN_REC_FRAC="0.75 0.6 0.5 0.25 0.1"

for MODEL in ${MODELS[@]}; do
	echo "Starting evals for ${MODEL}"
	python parallelize_evals.py \
		--checkpoint_path $MODEL \
		--config_names $CACHE_CONFIGS \
		--tasks $TASKS \
		--cache_sizes $CACHE_SIZES \
		--min_recovery_fractions $MIN_REC_FRAC \
		--num_samples $NUM_SAMPLES \
		--num_gpus 8 \
		--add_full
done
