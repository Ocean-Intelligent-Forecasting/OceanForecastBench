#!/bin/bash
#SBATCH -J fourcastinf
#SBATCH -p qgpu_3090
#SBATCH --gres=gpu:1
#SBATCH --mem=30g

/hpcfs/fhome/yangjh5/anaconda3/envs/fourcastnet/bin/python3.8 /hpcfs/fhome/yangjh5/benchproject/FourCastNet-master/inference/inference.py
