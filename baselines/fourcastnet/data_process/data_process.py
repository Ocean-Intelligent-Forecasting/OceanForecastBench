import xarray
import netCDF4 as nc
import h5py
import numpy as np

def mean():
    data = np.load('./normalize_mean_23_1.40625deg.npz')
    d = np.array([])
    dic = data.files
    for i in range(len(dic)):
        d = np.append(d, data[dic[i]])
    d = np.expand_dims(d, axis=1)
    d = np.expand_dims(d, axis=2)
    d = np.expand_dims(d, axis=0)
    np.save('../FCN_ERA5_data_v0/stats/global_means.npy', d)
    print(d.shape)

def std():
    data = np.load('./normalize_std_23_1.40625deg.npz')
    d = np.array([])
    dic = data.files
    for i in range(len(dic)):
        d = np.append(d, data[dic[i]])
    d = np.expand_dims(d, axis=1)
    d = np.expand_dims(d, axis=2)
    d = np.expand_dims(d, axis=0)
    np.save('../FCN_ERA5_data_v0/stats/global_std.npy', d)
    print(d.shape)

# mean()
# std()

data = np.load('normalize_mean_23_1.40625deg.npz')
print(data.files)