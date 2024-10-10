#!/bin/bash
#SBATCH -J swint_inference
#SBATCH -p qgpu_3090
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH -o inflogs/10.loop

/hpcfs/fhome/yangjh5/anaconda3/envs/climax_v5/bin/python3.8 inference_10.py
