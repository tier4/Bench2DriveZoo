#!/bin/bash

# Simple script to run evaluation on all checkpoints and both splits

for split in val train; do
	for checkpoint in /workspace/Bench2Drive/test_files/*.ckpt; do
		echo "Running: $(basename $checkpoint) on $split"
		python3 test_openloop_vad.py \
			--metric_method vad \
			--checkpoint "$checkpoint" \
			--device cuda \
			--model_type vad \
			--split $split
	done
done
