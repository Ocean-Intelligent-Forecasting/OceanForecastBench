import numpy as np
import xarray as xr
import ipdb
import os
def create_time_climatology_forecast(ds_train):
    return ds_train.mean('time')

def create_weekly_climatology_forecast(ds_train, valid_time):
    ds_train['week'] = ds_train['time.week']
    weekly_averages = ds_train.groupby('week').mean('time')
    valid_time['week'] = valid_time['time.week']
    fc_list = []
    for t in valid_time:
        fc_list.append(weekly_averages.sel(week = t.week))
    return xr.concat(fc_list, dim=valid_time)

def create_monthly_climatology_forecast(ds_train, valid_time):
    ds_train['month'] = ds_train['time.month']
    monthly_averages = ds_train.groupby('month').mean('time')
    valid_time['month'] = valid_time['time.month']
    fc_list = []
    for t in valid_time:
        fc_list.append(monthly_averages.sel(month = t.month))
    return xr.concat(fc_list, dim=valid_time)

res = '1.40625deg'
data_dir = f'xxx/sst_{res}'
baseline = ['time', 'weekly', 'monthly']

data = xr.open_mfdataset(data_dir + '/*.nc', combine= 'by_coords')
for b in baseline:
    baseline_dir = f'xxxx'
    os.makedirs(baseline_dir, exist_ok=True)
    train_data = data.sel(time=slice(None,'2017'))
    if b == 'time':
        climatology = eval(f'create_{b}_climatology_forecast(trian_data)')
        climatology.to_netcdf(f'{baseline_dir}/climatology_{res}.nc')
    else:
        test_data = data.sel(time=slice('2019','2020'))
        period_data = eval(f'create_{b}_climatology_forecast(train_data, test_data.time')
        period_data.to_netcdf(f'{baseline_dir}/{b}_climatology_{res}.nc')