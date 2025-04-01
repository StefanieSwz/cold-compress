#!/bin/bash

set -e

MODELS=(
	"checkpoints/meta-llama/Llama-3.2-3B-Instruct/model.pth"
)
NUM_SAMPLES=700
CACHE_SIZES="0.1 0.05"
TASKS="ultrachat"
CACHE_CONFIGS="random l2 heavy_hitter recent_global"

for MODEL in ${MODELS[@]}; do
	echo "Starting evals for ${MODEL}"
	python parallelize_evals.py \
		--checkpoint_path $MODEL \
		--config_names $CACHE_CONFIGS \
		--tasks $TASKS \
		--cache_sizes $CACHE_SIZES \
		--num_samples $NUM_SAMPLES \
		--num_gpus 2 \
		--add_full
done
