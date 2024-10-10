import click
import os
import numpy as np
from tqdm import tqdm
import netCDF4 as nc
from netCDF4 import date2num
from datetime import datetime,timedelta
@click.command()
@click.argument('input_npy_path')
@click.argument('output_nc_path')

def npy2nc(input_npy_path, output_nc_path):

    os.makedirs(output_nc_path, exist_ok=True)

    lat_path = '/hpcfs/fhome/yangjh5/jhm_data/latlon/1.40625deg/lat.npz'
    lon_path = '/hpcfs/fhome/yangjh5/jhm_data/latlon/1.40625deg/lon.npz'
    lat, lon = np.load(lat_path), np.load(lon_path)

    levels = 23
    depth = np.array(
        [0.4940, 2.6457, 5.0782, 7.9296, 11.4050,
         15.8101, 21.5988, 29.4447, 40.3441,
         55.7643, 77.8539, 92.3261,109.7293,
         130.6660, 155.8507, 186.1256, 222.4752,
         266.0403, 318.1274, 380.2130, 453.9377,
         541.0889, 643.5668])

    variables = ['_thetao', '_so', '_uo', '_vo', '_zos', '_sst']
    layer_index = [[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90],
                   [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91],
                   [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92],
                   [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93],
                   [0],
                   [1]]

    npy_fp = os.listdir(input_npy_path)
    npy_fp.sort()

    template_fp = '/hpcfs/fhome/yangjh5/jhm_data/baseline_nc_1.40625/merged_climatology/weekly/weekly_climatology_1.40625deg.nc'
    template = nc.Dataset(template_fp)

    for filei in tqdm(npy_fp):
        print(f'执行{filei}拼接任务中......')
        npy_data = np.load(os.path.join(input_npy_path, filei))

        for i in range(len(variables)):
            nc_file_path = os.path.join(output_nc_path, filei[:-4] + variables[i] + '.nc')
            nc_data = nc.Dataset(nc_file_path, 'w', format='NETCDF4')
            if i < 4:
                nc_data.createDimension('time', 1)
                nc_data.createDimension('lat', 121)
                nc_data.createDimension('lon', 256)
                nc_data.createDimension('depth', levels)

                #
                nc_data.createVariable('time', np.float32, ('time'), fill_value=np.nan)
                nc_data.createVariable('lat', np.float64, ('lat'), fill_value = np.nan)
                nc_data.createVariable('lon', np.float64, ('lon'), fill_value = np.nan)
                nc_data.createVariable('depth', np.float32, ('depth'), fill_value = np.nan)

                time_str = filei[-12:-4]
                time_dt = datetime.strptime(time_str, '%Y%m%d')
                nc_data.variables['time'][:] = date2num(time_dt, units='hours since 1950-01-01', calendar='gregorian')
                nc_data.variables['depth'][:] = depth
                nc_data.variables['lat'][:] = lat['arr_0']
                nc_data.variables['lon'][:] = lon['arr_0']

                nc_data.createVariable('prediction', np.float32, ('time', 'depth', 'lat', 'lon'), fill_value=np.nan)
                nc_data.variables['prediction'][:] = npy_data[:, layer_index[i], :, :]
            else:
                nc_data.createDimension('time', 1)
                nc_data.createDimension('lat', 121)
                nc_data.createDimension('lon', 256)

                nc_data.createVariable('time', np.float32, ('time'), fill_value=np.nan)
                nc_data.createVariable('lat', np.float64, ('lat'), fill_value=np.nan)
                nc_data.createVariable('lon', np.float64, ('lon'), fill_value=np.nan)

                time_str = filei[-12:-4]
                time_dt = datetime.strptime(time_str, "%Y%m%d")
                nc_data.variables['time'][:] = date2num(time_dt, units='hours since 1950-01-01', calendar='gregorian')
                nc_data.variables['lat'][:] = lat['arr_0']
                nc_data.variables['lon'][:] = lon['arr_0']
                nc_data.createVariable('prediction', np.float32, ('time', 'lat', 'lon'), fill_value=np.nan)
                nc_data.variables['prediction'][:] = npy_data[:, layer_index[i], :, :]

            for var_name in nc_data.variables:
                if var_name == 'prediction':
                    continue
                template_var = template.variables[var_name]
                nc_data_var = nc_data.variables[var_name]

                for attr_name in template_var.ncattrs():
                    if attr_name != '_FillValue':
                        nc_data_var.setncattr(attr_name, template_var.getncattr(attr_name))

            nc_data.close()
    template.close()
if __name__ == '__main__':
    npy2nc()
