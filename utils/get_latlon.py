import xarray as xr
import numpy as np
import os
resolutions = ['5.625deg', '2.8125deg', '1.40625deg']
for resolution in resolutions:
    data_dir = f'/public/home/wangxiang02/jhm/{resolution}'
    save_dir = f'/public/home/wangxiang02/jhm/latlon/{resolution}'
    os.makedirs(save_dir, exist_ok = True)
    data = xr.open_dataset(os.path.join(data_dir,f'mercatorglorys12v1_gl12_mean_19930101_{resolution}.nc'))
    lat = {'arr_0':data.lat.values}
    lon = {'arr_0':data.lon.values}
    np.savez(os.path.join(save_dir, 'lat.npz'), **lat)
    np.savez(os.path.join(save_dir, 'lon.npz'), **lon)



