#!/bin/bash
#SBATCH -J FourCastNet
#SBATCH -p qgpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --ntasks-per-node=4
#SBATCH -o logs/fourcastnetdistribute_jhm.loop

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 train_jhm.py

