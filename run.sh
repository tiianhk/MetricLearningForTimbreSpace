#!/bin/bash

batch_size=32
alpha=0.7
beta=0.5

python train.py --batch_size $batch_size \
		--alpha $alpha \
		--beta $beta \
		--hierarchy

