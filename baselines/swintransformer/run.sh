#!/bin/bash
#SBATCH -J swintransformer_base
#SBATCH -p qgpu_4090
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/L10p2e192.loop
srun /hpcfs/fhome/yangjh5/anaconda3/envs/climax_v5/bin/python3.8 src/climax/global_forecast/train.py --config configs/global_forecast_23_l10.yaml --trainer.max_epochs=300 --trainer.devices=1 --data.root_dir=/hpcfs/fhome/yangjh5/jhm_data/1.40625deg_npy --data.batch_size=8 --data.num_workers=2 --model.pretrained_path=''
