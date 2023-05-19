#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=9:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=tianhaokun
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ht2208@nyu.edu

module purge

singularity exec --nv \
	    --overlay /scratch/ht2208/music-trees-env/music-trees.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate metriclearningmir; ./run.sh"
