import numpy as np
import h5py
import os
import datetime
import sys

# np.set_printoptions(threshold=sys.maxsize)
def transfer(year):
    first_day = datetime.date(year, 1, 1)
    delta = 1
    data = np.load('../../../jhm_data/1.40625deg_npy/data/val/mra5_'+str(year)+'0101'+'.npy')
    data = np.pad(data, ((0, 0), (0, 0), (0, 7), (0, 0)), 'constant')
    data = np.nan_to_num(data, nan=0)
    print(data.shape)
    while True:
        d = first_day + datetime.timedelta(days=delta)
        if datetime.date(year, 12, 31) < d:
            break
        file_name = 'mra5_'+d.isoformat().replace('-', '')+'.npy'
        if os.path.exists('../../../jhm_data/1.40625deg_npy/data/val/'+file_name):
            data1 = np.load('../../../jhm_data/1.40625deg_npy/data/val/'+file_name)
            data1 = np.pad(data1, ((0, 0), (0, 0), (0, 7), (0, 0)), 'constant')
            data1 = np.nan_to_num(data1, nan=0)
            data = np.vstack((data, data1))
            delta += 1
        else:
            delta += 1
    with h5py.File('./FCN_ERA5_data_v0/val/'+str(year)+'.h5', 'w') as file:
        dset = file.create_dataset('fields', data=data)
    with h5py.File('./FCN_ERA5_data_v0/val/'+str(year)+'.h5', 'r') as file:
        data = file['fields']
        print(data.shape)

for i in range(2018, 2019):
    transfer(i)