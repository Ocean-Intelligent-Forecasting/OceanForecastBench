import os
import xarray as xr
import numpy as np
import datetime
import time
from cdo import Cdo
from tqdm import tqdm
import glob
cdo = Cdo()

def transData(fp, save_path, resolution):

    m_r = {'5.625deg':(64,32), '2.8125deg':(128,64), '1.40625deg':(256,128)}
    fp_id = fp.split('/')[-1][:-3] + '_' + resolution + '.nc'
    cdo.sellonlatbox('-180,180,-80,90',input='-remapbil,r{0}x{1} '.format(m_r[resolution][0],m_r[resolution][1])+fp, output=save_path+'/'+fp_id, options='-P 32')
    print(f'{fp_id} processed')
resolutions = ['5.625deg', '2.8125deg', '1.40625deg']
for r in resolutions:
    fp_mkt = '/public/home/acct230421094230/ERA5WIND'
    save_mkt = f'/public/home/wangxiang02/jhm/era5_{r}'
    files = os.path.join(fp_mkt,f"era5*.nc")
    files = glob.glob(files)
    for fp in tqdm(files):
        fp_ver = fp.split('/')[-1][:-3] + '_' + r + '.nc'
        if fp_ver in os.listdir(save_mkt):
            print(f'{fp_ver} '+'already exists')
            continue
        transData(fp, save_mkt, r)
