#!/bin/bash
#SBATCH -J ClimaX_60M
#SBATCH -p qgpu_3090
#SBATCH --gres=gpu:8
#SBATCH --mem=50g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/L10p1e512depth4decoder2_ocean_0811mlp2.loop
srun /hpcfs/fhome/yangjh5/anaconda3/envs/climax_v2/bin/python3.8 src/climax/global_forecast/train.py --config configs/global_forecast_23_l10_1.40625_0810_v2.yaml --trainer.max_epochs=300 --trainer.devices=8 --data.root_dir=/hpcfs/fhome/yangjh5/jhm_data/1.40625deg_npy --data.batch_size=1 --data.num_workers=1 --model.pretrained_path=''
