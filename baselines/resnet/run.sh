#!/bin/bash
#SBATCH -J ResNet50
#SBATCH -p qgpu_3090
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/L10_20241015.loop

srun /hpcfs/fhome/yangjh5/anaconda3/envs/climax_v6/bin/python3.8 src/climax/global_forecast/train.py --config configs/global_forecast_23_l10_1.40625_20241015.yaml --trainer.max_epochs=300 --trainer.devices=8 --data.root_dir=/hpcfs/fhome/yangjh5/jhm_data/1.40625deg_npy --data.batch_size=16 --data.num_workers=2 --model.pretrained_path=''
