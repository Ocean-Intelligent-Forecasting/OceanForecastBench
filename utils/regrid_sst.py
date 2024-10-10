import os
import xarray as xr
import numpy as np
from datetime import datetime,timedelta
import time
from cdo import Cdo
from tqdm import tqdm
import glob
cdo = Cdo()

def transData(fp, save_path, resolution, ymd):

    m_r = {'5.625deg':(64,32), '2.8125deg':(128,64), '1.40625deg':(256,128)}
    fp_id = 'GHRSST_'+ymd+'_'+resolution+'.nc'
    cdo.sellonlatbox('-180,180,-80,90',input='-remapbil,r{0}x{1} '.format(m_r[resolution][0],m_r[resolution][1])+fp, output=save_path+'/'+fp_id, options='-P 32')
    print(f'{fp_id} processed')
resolutions = ['5.625deg', '2.8125deg', '1.40625deg']
for r in resolutions:
    start_date = datetime(1993,1,1)
    end_date = datetime(2020,12,31)
    n_days = (end_date-start_date).days + 1
    fp_mkt = '/public/home/acct230421094230/SST_1993_2022/data_raw/'
    save_mkt = f'/public/home/wangxiang02/jhm/sst_{r}'
    for i in tqdm(range(n_days)):
        cur_date = start_date + timedelta(days=i)
        year = cur_date.strftime("%Y")
        ym = cur_date.strftime("%Y%m")
        ymd = cur_date.strftime("%Y%m%d")
        fp = f'/public/home/acct230421094230/SST_1993_2022/data_raw/{year}/{ym}/{ymd}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc'
        fp_ver = 'GHRSST_'+ymd+'_'+r+'.nc'
        if fp_ver in os.listdir(save_mkt):
            print(f'{fp_ver} '+'already exists')
            continue
        transData(fp, save_mkt, r, ymd)
