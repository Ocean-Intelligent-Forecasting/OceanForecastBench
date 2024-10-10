import torch
import torchdata.datapipes as dp
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule

from climax.arch_final import ClimaX
from climax.global_forecast.module import GlobalForecastModule

import numpy as np
import os
from tqdm import tqdm
import glob
from datetime import datetime,timedelta
def inference(input_npy, output_npy, lead_time):
    in_transform = get_normalize(in_variables)
    out_transform = get_denormalize(out_variables)
    #val_fp = '/'+os.path.join(*input_npy.split('/')[:-1],'val')
    #fp = sorted(list(dp.iter.FileLister(val_fp)))[-ildy:]+sorted(list(dp.iter.FileLister(input_npy)))[:-ildy]
    fp = sorted(list(dp.iter.FileLister(input_npy)))
    for ifp in tqdm(fp):
        ymd = datetime.strptime(ifp.split('/')[-1].split('.')[0][-8:], '%Y%m%d')
        print('Date of predict is ' + ymd.strftime('%Y%m%d'))
        p_ymd = ymd + timedelta(days=ildy)
        p_ymd = p_ymd.strftime('%Y%m%d')
        if 'pred_mra5_'+p_ymd+'.npy' in os.listdir(output_npy):
            continue
        x = np.load(ifp)
        mask = np.isnan(x)
        mask = np.concatenate((mask[:,[0,3],:,:],mask[:,4:,:,:]), axis=1)
        data_x = read_data(default_vars,in_variables,x,in_transform)
        first_iteration = True
        if first_iteration:
            first_iteration = False
            np.save('./mask.npy', mask)
        y_hat = model.forward(data_x, lead_time, in_variables, out_variables)
        y_hat = out_transform(y_hat)
        y_hat = y_hat.detach().numpy()
        y_hat[mask] = np.nan
        print('Date of '+p_ymd+'finished')
        np.save(os.path.join(output_npy,'pred_mra5_'+p_ymd+'.npy'), y_hat)

def create_var_map(variables):
    var_map = {}
    idx = 0
    for var in variables:
        var_map[var] = idx
        idx += 1
    return var_map


def read_data(default_vars, in_variables, x, in_transform):
    var_map = create_var_map(default_vars)
    index_in = []
    for var in in_variables:
        index_in.append(var_map[var])
    data_x = x[:,index_in,:,:]
    data_x = torch.from_numpy(data_x).float()
    data_x = in_transform(data_x)
    data_x[np.isnan(data_x).bool()] = 0
    return data_x
#    return torch.from_numpy(data_x).float()
def get_normalize(variables):
    normalize_mean = dict(np.load(os.path.join('/hpcfs/fhome/yangjh5/jhm_data/normalize_1.40625deg','normalize_mean_23_1.40625deg.npz')))
    mean = []
    for var in variables:
        mean.append(normalize_mean[var])
    normalize_mean = np.concatenate(mean)
    normalize_std = dict(np.load(os.path.join('/hpcfs/fhome/yangjh5/jhm_data/normalize_1.40625deg','normalize_std_23_1.40625deg.npz')))
    normalize_std = np.concatenate([normalize_std[var] for var in variables])
    return transforms.Normalize(normalize_mean, normalize_std)
def get_denormalize(variables):
    normalization = get_normalize(variables)
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1/std_norm
    return transforms.Normalize(mean_denorm, std_denorm)
if __name__ == '__main__':
    ildy = 10
    #input_npy = '/hpcfs/fhome/yangjh5/jhm_data/1.40625deg_npy/data/test'
    input_npy = '/hpcfs/fhome/yangjh5/jhm_data/0817/input_data_2022-2023'
    output_npy = f'/hpcfs/fhome/yangjh5/jhm_data/baseline_1.40625/ClimaX2022-2023/{ildy}'
    #model_fp = f'/hpcfs/fhome/yangjh5/jhm_data/oceanrongqiyun/climaxswin_1.40625/global_forecast_ocean_test/swin_checkpoints/L{ildy}p1e288/best*.ckpt'
    #model_fp = glob.glob(model_fp)[0]
    model_fp ='/hpcfs/fhome/yangjh5/benchproject/ClimaX/global_forecast_ocean/bench_climax_checkpoints/L10p2e1024depth4decoder2/best_013_val_loss0.1883.ckpt'
    default_vars = [
        'zos',
        'u',
        'v',
        'sst',
        'thetao_0',
        'so_0',
        'uo_0',
        'vo_0',
        'thetao_2',
        'so_2',
        'uo_2',
        'vo_2',
        'thetao_4',
        'so_4',
        'uo_4',
        'vo_4',
        'thetao_6',
        'so_6',
        'uo_6',
        'vo_6',
        'thetao_8',
        'so_8',
        'uo_8',
        'vo_8',
        'thetao_10',
        'so_10',
        'uo_10',
        'vo_10',
        'thetao_12',
        'so_12',
        'uo_12',
        'vo_12',
        'thetao_14',
        'so_14',
        'uo_14',
        'vo_14',
        'thetao_16',
        'so_16',
        'uo_16',
        'vo_16',
        'thetao_18',
        'so_18',
        'uo_18',
        'vo_18',
        'thetao_20',
        'so_20',
        'uo_20',
        'vo_20',
        'thetao_21',
        'so_21',
        'uo_21',
        'vo_21',
        'thetao_22',
        'so_22',
        'uo_22',
        'vo_22',
        'thetao_23',
        'so_23',
        'uo_23',
        'vo_23',
        'thetao_24',
        'so_24',
        'uo_24',
        'vo_24',
        'thetao_25',
        'so_25',
        'uo_25',
        'vo_25',
        'thetao_26',
        'so_26',
        'uo_26',
        'vo_26',
        'thetao_27',
        'so_27',
        'uo_27',
        'vo_27',
        'thetao_28',
        'so_28',
        'uo_28',
        'vo_28',
        'thetao_29',
        'so_29',
        'uo_29',
        'vo_29',
        'thetao_30',
        'so_30',
        'uo_30',
        'vo_30',
        'thetao_31',
        'so_31',
        'uo_31',
        'vo_31',
        'thetao_32',
        'so_32',
        'uo_32',
        'vo_32'
    ]
    in_variables = [  # 模型输入的变量（按顺序）
        'u',
        'v',
        'zos',
        'sst',
        'thetao_0',
        'so_0',
        'uo_0',
        'vo_0',
        'thetao_2',
        'so_2',
        'uo_2',
        'vo_2',
        'thetao_4',
        'so_4',
        'uo_4',
        'vo_4',
        'thetao_6',
        'so_6',
        'uo_6',
        'vo_6',
        'thetao_8',
        'so_8',
        'uo_8',
        'vo_8',
        'thetao_10',
        'so_10',
        'uo_10',
        'vo_10',
        'thetao_12',
        'so_12',
        'uo_12',
        'vo_12',
        'thetao_14',
        'so_14',
        'uo_14',
        'vo_14',
        'thetao_16',
        'so_16',
        'uo_16',
        'vo_16',
        'thetao_18',
        'so_18',
        'uo_18',
        'vo_18',
        'thetao_20',
        'so_20',
        'uo_20',
        'vo_20',
        'thetao_21',
        'so_21',
        'uo_21',
        'vo_21',
        'thetao_22',
        'so_22',
        'uo_22',
        'vo_22',
        'thetao_23',
        'so_23',
        'uo_23',
        'vo_23',
        'thetao_24',
        'so_24',
        'uo_24',
        'vo_24',
        'thetao_25',
        'so_25',
        'uo_25',
        'vo_25',
        'thetao_26',
        'so_26',
        'uo_26',
        'vo_26',
        'thetao_27',
        'so_27',
        'uo_27',
        'vo_27',
        'thetao_28',
        'so_28',
        'uo_28',
        'vo_28',
        'thetao_29',
        'so_29',
        'uo_29',
        'vo_29',
        'thetao_30',
        'so_30',
        'uo_30',
        'vo_30',
        'thetao_31',
        'so_31',
        'uo_31',
        'vo_31',
        'thetao_32',
        'so_32',
        'uo_32',
        'vo_32'
    ]
    out_variables = [  # 模型输出的顺序（按顺序）
        'zos',
        'sst',
        'thetao_0',
        'so_0',
        'uo_0',
        'vo_0',
        'thetao_2',
        'so_2',
        'uo_2',
        'vo_2',
        'thetao_4',
        'so_4',
        'uo_4',
        'vo_4',
        'thetao_6',
        'so_6',
        'uo_6',
        'vo_6',
        'thetao_8',
        'so_8',
        'uo_8',
        'vo_8',
        'thetao_10',
        'so_10',
        'uo_10',
        'vo_10',
        'thetao_12',
        'so_12',
        'uo_12',
        'vo_12',
        'thetao_14',
        'so_14',
        'uo_14',
        'vo_14',
        'thetao_16',
        'so_16',
        'uo_16',
        'vo_16',
        'thetao_18',
        'so_18',
        'uo_18',
        'vo_18',
        'thetao_20',
        'so_20',
        'uo_20',
        'vo_20',
        'thetao_21',
        'so_21',
        'uo_21',
        'vo_21',
        'thetao_22',
        'so_22',
        'uo_22',
        'vo_22',
        'thetao_23',
        'so_23',
        'uo_23',
        'vo_23',
        'thetao_24',
        'so_24',
        'uo_24',
        'vo_24',
        'thetao_25',
        'so_25',
        'uo_25',
        'vo_25',
        'thetao_26',
        'so_26',
        'uo_26',
        'vo_26',
        'thetao_27',
        'so_27',
        'uo_27',
        'vo_27',
        'thetao_28',
        'so_28',
        'uo_28',
        'vo_28',
        'thetao_29',
        'so_29',
        'uo_29',
        'vo_29',
        'thetao_30',
        'so_30',
        'uo_30',
        'vo_30',
        'thetao_31',
        'so_31',
        'uo_31',
        'vo_31',
        'thetao_32',
        'so_32',
        'uo_32',
        'vo_32'
    ]
    img_size = [121, 256]
    patch_size = 2
    embed_dim = 1024
    depth = 4
    decoder_depth = 2
    num_heads = 16
    mlp_ratio = 4
    drop_path = 0.2
    drop_rate = 0.1

    print("Loading checkpoint from %s,...." % model_fp)
    checkpoint = torch.load(model_fp)
    checkpoint_model = checkpoint["state_dict"]

    model = ClimaX(default_vars=in_variables, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=4, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.2, drop_rate=0.1)

    for k in list(checkpoint_model.keys()):
        if "net." in k:
            checkpoint_model[k.replace("net.", "")] = checkpoint_model[k]
            del checkpoint_model[k]
    model.load_state_dict(state_dict=checkpoint_model)
    model.eval()
    predict_range = torch.ones(1).to(torch.long) * ildy
    lead_time = predict_range / 100

    os.makedirs(output_npy, exist_ok=True)
    inference(input_npy=input_npy,output_npy=output_npy, lead_time = lead_time)

