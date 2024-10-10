import numpy as np
#import torch
from netCDF4 import Dataset
import os
from tqdm import tqdm
import time
import glob
from datetime import datetime, timedelta
import xarray as xr
# generate npy file from mkt,era5(u,v),ghrsst nc file
# mkt : /public/home/wangxiang02/jhm/{resolution}/mercatorglorys12v1_gl12_mean_20190101_{resolution}.nc
# era5: /public/home/wangxiang02/jhm/era5_{resolution}/era5_wind_1999_{resolution}.nc
# ghrsst: /public/home/wangxiang02/jhm/sst_{resolution}/GHRSST-20190101_{resolution}.nc
def generatedata(resolution, save_path):
    start_date = datetime(1993, 1, 1)
    end_date = datetime(2020, 12, 31)
    n_days = (end_date - start_date).days + 1
    for i in tqdm(range(n_days)):
        cur_date = start_date + timedelta(days=i)
        cur_date_str = cur_date.strftime("%Y%m%d")
        mkt = f'/public/home/wangxiang02/jhm/{resolution}/mercatorglorys12v1_gl12_mean_{cur_date_str}_{resolution}.nc'
        era5 = f'/public/home/wangxiang02/jhm/era5_{resolution}/era5_wind_{cur_date_str[:4]}_{resolution}.nc'
        ghrsst = f'/public/home/wangxiang02/jhm/sst_{resolution}/GHRSST_{cur_date_str}_{resolution}.nc'
        year = datetime(cur_date.year, 1, 1)
        day = (cur_date - year).days
        mkt_data = xr.open_dataset(mkt)
        era5_data = xr.open_dataset(era5)
        ghrsst_data = xr.open_dataset(ghrsst)
        depth_list = {'surface':[0, 2, 4, 6, 8, 10, 12 ,14, 16, 18, 20, 21], 'deep':[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]}
        latlon = {'5.625deg':(30,64), '2.8125deg':(60,128), '1.40625deg':(121,256)}
        data_1to22 = np.empty((1,52,latlon[resolution][0],latlon[resolution][1]))
        data_1to22[:,0,:,:] = mkt_data['zos'].values
        data_1to22[0,1,:,:] = era5_data['u10'][day*4:(day+1)*4,:,:].data.mean(axis=0)
        data_1to22[0,2,:,:] = era5_data['v10'][day*4:(day+1)*4,:,:].data.mean(axis=0)
        data_1to22[:,3,:,:] = ghrsst_data['analysed_sst'].values - 273.15
        # thetao
        loc = [4*i for i in range(1,len(depth_list['surface'])+1)]
        data_1to22[:,loc,:,:] = mkt_data['thetao'][:, depth_list['surface'], :, :].values
        #so
        loc = [4*i+1 for i in range(1,len(depth_list['surface'])+1)]
        data_1to22[:,loc,:,:] = mkt_data['so'][:, depth_list['surface'], :, :].values
        #uo
        loc = [4*i+2 for i in range(1,len(depth_list['surface'])+1)]
        data_1to22[:,loc,:,:] = mkt_data['uo'][:, depth_list['surface'], :, :].values
        #vo
        loc = [4*i+3 for i in range(1,len(depth_list['surface'])+1)]
        data_1to22[:,loc,:,:] = mkt_data['vo'][:, depth_list['surface'], :, :].values
        result = data_1to22.astype(np.float16)
        mkt_data.close();era5_data.close();ghrsst_data.close()

        cur_date = start_date + timedelta(days=i)
        cur_date_str = cur_date.strftime("%Y%m%d")
        mkt = f'/public/home/wangxiang02/jhm/{resolution}/mercatorglorys12v1_gl12_mean_{cur_date_str}_{resolution}.nc'
        era5 = f'/public/home/wangxiang02/jhm/era5_{resolution}/era5_wind_{cur_date_str[:4]}_{resolution}.nc'
        ghrsst = f'/public/home/wangxiang02/jhm/sst_{resolution}/GHRSST_{cur_date_str}_{resolution}.nc'
        year = datetime(cur_date.year, 1, 1)
        day = (cur_date - year).days
        mkt_data = xr.open_dataset(mkt)
        era5_data = xr.open_dataset(era5)
        ghrsst_data = xr.open_dataset(ghrsst)
        depth_list = {'surface':[0, 2, 4, 6, 8, 10, 12 ,14, 16, 18, 20, 21], 'deep':[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]}
        latlon = {'5.625deg':(30,64), '2.8125deg':(60,128), '1.40625deg':(121,256)}
        data_23to33 = np.empty((1,48,latlon[resolution][0],latlon[resolution][1]))
        data_23to33[:,0,:,:] = mkt_data['zos'].values
        data_23to33[0,1,:,:] = era5_data['u10'][day*4:(day+1)*4,:,:].data.mean(axis=0)
        data_23to33[0,2,:,:] = era5_data['v10'][day*4:(day+1)*4,:,:].data.mean(axis=0)
        data_23to33[:,3,:,:] = ghrsst_data['analysed_sst'].values - 273.15
        # thetao
        loc = [4*i for i in range(1,len(depth_list['deep'])+1)]
        data_23to33[:,loc,:,:] = mkt_data['thetao'][:, depth_list['deep'], :, :].values
        #so
        loc = [4*i+1 for i in range(1,len(depth_list['deep'])+1)]
        data_23to33[:,loc,:,:] = mkt_data['so'][:, depth_list['deep'], :, :].values
        #uo
        loc = [4*i+2 for i in range(1,len(depth_list['deep'])+1)]
        data_23to33[:,loc,:,:] = mkt_data['uo'][:, depth_list['deep'], :, :].values
        #vo
        loc = [4*i+3 for i in range(1,len(depth_list['deep'])+1)]
        data_23to33[:,loc,:,:] = mkt_data['vo'][:, depth_list['deep'], :, :].values
        result = data_23to33.astype(np.float16)
        train_data = np.concatenate([data_1to22, data_23to33[:,4:,:,:]], axis=1)
        filename = 'mra5_' + cur_date_str + '.npy'
        print('saving ' + os.path.join(save_path, filename) + '......')
        np.save(os.path.join(save_path, filename), train_data)
        mkt_data.close();era5_data.close();ghrsst_data.close()

if __name__ == '__main__':
    #resolutions = ['5.625deg', '2.8125deg', '1.40625deg']
    resolutions = ['2.8125deg', '1.40625deg']
    #resolutions = ['5.625deg'] 
    for resolution in resolutions:
        
        save_path = f'/public/home/wangxiang02/jhm/{resolution}_npy/data'
        os.makedirs(save_path, exist_ok = True)
        generatedata(resolution, save_path)




 
