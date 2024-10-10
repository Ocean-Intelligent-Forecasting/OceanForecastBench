import numpy as np
from tqdm import tqdm
import click
import glob
import os
import time
from datetime import datetime, timedelta


# npy
# '/public/home/wangxiang02/jhm/{resolution}_npy/{depth}/mra5_{cur_date}.npy'

def get_mean(root_dir, save_dir):
    pass
def get_Data(resolution, root_dir, save_dir):
    var_list = ['thetao', 'so', 'uo', 'vo']
    depth_list_1to22 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]
    depth_list_23to33 = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    depths = {'surface':12, 'deep':11}
    variables1 = {'zos':0, 'u':1, 'v':2, 'sst':3}
    start_date = datetime(1993, 1, 1)
    end_date = datetime(2017, 12, 31)
    n_days = (end_date - start_date).days + 1
    latlon = {'5.625deg':(30,64), '2.8125deg':(60,128), '1.40625deg':(121,256)}
    for depth in depths:
        if depth == 'surface':
            variables2 = {f'{var}_{depth_list_1to22[i//4]}':i+4 for i, var in enumerate(var_list*depths[depth])}
            data_dir = os.path.join(root_dir, 'data1to22')
        elif depth == 'deep':
            variables2 = {f'{var}_{depth_list_23to33[i//4]}':i+4 for i, var in enumerate(var_list*depths[depth])}
            data_dir = os.path.join(root_dir, 'data23to33')
        variables = {**variables1, **variables2}
        daily_data = {}
        normalize_mean = {}
        printed = False
        for i in tqdm(range(n_days)):
            cur_date = start_date + timedelta(days=i)
            cur_date_str = cur_date.strftime("%Y%m%d")
            fp = os.path.join(data_dir, f'mra5_{cur_date_str}.npy')
            data = np.load(fp)
            for var, idx in variables.items():
                cur_var = data[0, idx, :, :]
                if not printed:
                    print('before reshape size:', cur_var.shape)
                    printed = True
                no_mask = ~np.isnan(cur_var)
                cur_var[~no_mask] = 0
                if var not in daily_data:
                    daily_data[var] = [cur_var.reshape(1, 1, latlon[resolution][0], latlon[resolution][1]).astype(np.float32).sum(axis=(0,2,3)) / no_mask.sum()]
                else:
                    daily_data[var].append(cur_var.reshape(1, 1, latlon[resolution][0], latlon[resolution][1]).astype(np.float32).sum(axis=(0,2,3)) / no_mask.sum())
        for var in variables:
            daily_data[var] = np.stack(daily_data[var], axis=0)
            if var not in normalize_mean:
                normalize_mean[var] = daily_data[var].mean(axis=0).astype(np.float32)
        if depth == 'surface':
            normalize_mean_1to22 = normalize_mean
        elif depth == 'deep':
            normalize_mean_23to33 = normalize_mean
    normalize_mean_23 = {**normalize_mean_1to22, **normalize_mean_23to33}
    mean_name = f"normalize_mean_23_{resolution}.npz"
    np.savez(os.path.join(save_dir, mean_name), **normalize_mean_23)




if __name__ == "__main__":

    #resolutions = ['5.625deg', '2.8125deg', '1.40625deg']
    resolutions = ['2.8125deg', '1.40625deg']

    #resolutions = ['5.625deg']
    for resolution in resolutions:
        root_dir = f'/public/home/wangxiang02/jhm/{resolution}_npy/'
        save_dir = f'/public/home/wangxiang02/jhm/normalize_{resolution}/'
        os.makedirs(save_dir, exist_ok = True)
        get_Data(resolution, root_dir, save_dir)
