#!/bin/bash
#SBATCH -J data_process
#SBATCH -p qgpu_3090
#SBATCH --gres=gpu:1
#SBATCH --mem=10g

/hpcfs/fhome/yangjh5/anaconda3/envs/fourcastnet/bin/python3.8 /hpcfs/fhome/yangjh5/benchproject/FourCastNet-master/data/transfer.py