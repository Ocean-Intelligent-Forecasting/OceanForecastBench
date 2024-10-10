# Copyright (c) 2024 College of Meteorology and Oceanography, National University of Defense Technology. All rights reserved
# Code adapted from:
# https://github.com/beijixiong1/OceanForecastBench/
import os
import numpy as np
import netCDF4 as nc
from netCDF4 import date2num
import xarray as xr
import argparse
import pathlib
import time
import pyinstrument
from tqdm import tqdm
from datetime import datetime, timedelta
import glob

from eval_metrics import *
from interp import *
import sys
import gzip
import pickle
from pathlib import Path
import ipdb
import warnings

warnings.filterwarnings("ignore")


def evaluate_TSP(t_grid, t_depth, en4_data_filtered, baseline_data_interp, hys, clm):
    if en4_data_filtered != None:
        tsp_gmetrics = cal_global_metrics_TSP(en4_data_filtered, baseline_data_interp, hys, clm)
        print('tsp_gmetrics: ', tsp_gmetrics)
        # sys.exit()
        tsp_hmetrics = cal_horizontal_metrics_TSP(en4_data_filtered, baseline_data_interp, hys, clm, t_grid)
        tsp_smetrics = cal_subocean_metrics_TSP(bounds, en4_data_filtered, baseline_data_interp, t_grid, hys, clm)
        tsp_vmetrics = cal_vertical_metrics_TSP(t_depth, en4_data_filtered, baseline_data_interp, hys, clm, blev=blev,
                                                ulev=ulev, levs=levs)
    else:
        if list(baselines.keys())[0] == 'climatology':
            tsp_gmetrics = np.full((3, 2, 5), np.nan, dtype=np.float32)
            tsp_hmetrics = np.full((3, 2, 5, 1920), np.nan, dtype=np.float32)
            tsp_smetrics = np.full((3, 2, 5, 9), np.nan, dtype=np.float32)
            tsp_vmetrics = np.full((3, 2, 5, 26), np.nan, dtype=np.float32)
        else:
            tsp_gmetrics = np.full((2, 5), np.nan, dtype=np.float32)
            tsp_hmetrics = np.full((2, 5, 1920), np.nan, dtype=np.float32)
            tsp_smetrics = np.full((2, 5, 9), np.nan, dtype=np.float32)
            tsp_vmetrics = np.full((2, 5, 26), np.nan, dtype=np.float32)

    return [tsp_gmetrics, tsp_hmetrics, tsp_smetrics, tsp_vmetrics]


def evaluate_ZOS(t_grid, zos_data_filtered, baseline_data_interp, hys, clm):
    if zos_data_filtered != None:
        zos_gmetrics = cal_global_metrics_ZOS(zos_data_filtered, baseline_data_interp, hys, clm)
        print('zos_gmetrics: ', zos_gmetrics)
        zos_hmetrics = cal_horizontal_metrics_ZOS(zos_data_filtered, baseline_data_interp, hys, clm, t_grid)
        zos_smetrics = cal_subocean_metrics_ZOS(bounds, zos_data_filtered, baseline_data_interp, t_grid, hys, clm)
    else:
        if list(baselines.keys())[0] == 'climatology':
            zos_gmetrics = np.full((3, 1, 5), np.nan, dtype=np.float32)
            zos_hmetrics = np.full((3, 1, 5, 1920), np.nan, dtype=np.float32)
            zos_smetrics = np.full((3, 1, 5, 9), np.nan, dtype=np.float32)
        else:
            zos_gmetrics = np.full((1, 5), np.nan, dtype=np.float32)
            zos_hmetrics = np.full((1, 5, 1920), np.nan, dtype=np.float32)
            zos_smetrics = np.full((1, 5, 9), np.nan, dtype=np.float32)

    return [zos_gmetrics, zos_hmetrics, zos_smetrics]


def evaluate_GDP(t_grid, gdp_data_filtered, baseline_data_interp, hys, clm):
    if gdp_data_filtered != None:
        gdp_gmetrics = cal_global_metrics_GDP(gdp_data_filtered, baseline_data_interp, hys, clm)
        print('gdp_gmetrics: ', gdp_gmetrics)
        gdp_hmetrics = cal_horizontal_metrics_GDP(gdp_data_filtered, baseline_data_interp, hys, clm, t_grid)
        gdp_smetrics = cal_subocean_metrics_GDP(bounds, gdp_data_filtered, baseline_data_interp, t_grid, hys, clm)
    else:
        if list(baselines.keys())[0] == 'climatology':
            gdp_gmetrics = np.full((3, 3, 5), np.nan, dtype=np.float32)
            gdp_hmetrics = np.full((3, 3, 5, 1920), np.nan, dtype=np.float32)
            gdp_smetrics = np.full((3, 3, 5, 9), np.nan, dtype=np.float32)
        else:
            gdp_gmetrics = np.full((3, 5), np.nan, dtype=np.float32)
            gdp_hmetrics = np.full((3, 5, 1920), np.nan, dtype=np.float32)
            gdp_smetrics = np.full((3, 5, 9), np.nan, dtype=np.float32)
    return [gdp_gmetrics, gdp_hmetrics, gdp_smetrics]


class collect_data():

    def __init__(self, obs_fp, start_date, end_date, variable_name=None, baselines=None, ildy=None):
        self.en4_fp = os.path.join(obs_fp, 'temperature_salinity')
        self.zos_fp = os.path.join(obs_fp, 'zos')

        self.gdp_fp = os.path.join(obs_fp, 'current_sst')
        self.gdp_data = xr.load_dataset(os.path.join(self.gdp_fp, 'drifter_6hour_qc_c35b_4db8_1f03_U1718152782438.nc'))
        self.gdp_group = self.gdp_data.sortby('time').groupby('time.date')

        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')
        self.current_date = self.start_date
        self.variable_name = variable_name

        self.baselines = baselines
        self.baselines_dataset = {}
        self.ildy = ildy

        self.mkt = {'2D': 'zos', '3D-so': 'so', '3D-thetao': 'thetao', '3D-uovo': 'uovo'}
        # self.mkt_level = {'thetao':[0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32],
        #                   'so':[0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32],
        #                   'uo':[0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32],
        #                   'vo':[0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32],
        #                   }
        self.mkt_level = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_date > self.end_date:
            raise StopIteration
        en4_data = self.load_EN4_data()
        gdp_data = self.load_GDP_data()
        zos_data = self.load_ZOS_data()
        fct_data, fct_hys = self.load_fct_data()
        self.current_date += timedelta(days=1)

        return en4_data, gdp_data, zos_data, fct_data, fct_hys

    def load_EN4_data(self):
        try:
            year = self.current_date.strftime("%Y")
            month = self.current_date.strftime("%m")
            day = self.current_date.strftime("%d")
            fp = os.path.join(self.en4_fp, f'EN.4.2.2.profiles.g10.{year}', f'EN.4.2.2.f.profiles.g10.{year}{month}.nc')
            en4_data = xr.load_dataset(fp)
            en4_group = en4_data.sortby('JULD').groupby('JULD.date')
            data = en4_group[self.current_date.date()].copy()
            en4_data.close()
            return data
            # with xr.load_dataset(fp) as en4_data:
            #     en4_group = en4_data.sortby('JULD').groupby('JULD.date')
            #     return en4_group[self.current_date.date()]
        except:
            return None

    def load_GDP_data(self):
        try:
            return self.gdp_group[self.current_date.date()]
        except:
            return None

    def load_ZOS_data(self):
        try:
            ym = self.current_date.strftime('%Y%m')
            ymd = self.current_date.strftime('%Y%m%d')
            fp = os.path.join(self.zos_fp, ym, f'*{ymd}*.nc')
            fp = glob.glob(fp)[0]
            zos_data = xr.load_dataset(fp)
            data = zos_data.copy()
            zos_data.close()
            return data
            # with xr.load_dataset(fp) as zos_data:
            #     return zos_data
        except:
            return None

    def load_fct_data(self):

        if self.baselines is None:
            return None, None
        for b_name in self.baselines.keys():
            if b_name == 'climatology':
                start_date_2019 = datetime(2019, 1, 1)
                end_date_2020 = datetime(2020, 12, 31)
                ndays = (end_date_2020 - start_date_2019).days + 1
                for i in range(ndays):
                    if self.current_date.isocalendar()[1] == (start_date_2019 + timedelta(days=i)).isocalendar()[1]:
                        clm_week = start_date_2019 + timedelta(days=i)
                        break

                self.baselines_dataset.setdefault(b_name, {})
                with xr.load_dataset(os.path.join(self.baselines[b_name], 'time', 'climatology_1.40625deg.nc')) as ds:
                    self.baselines_dataset[b_name]['time'] = ds
                with xr.load_dataset(os.path.join(self.baselines[b_name], 'weekly', 'weekly_climatology_1.40625deg.nc')) as ds:
                    self.baselines_dataset[b_name]['weekly'] = ds.sel(time=clm_week.replace(hour=12, minute=0, second=0))
                with xr.load_dataset(os.path.join(self.baselines[b_name], 'monthly', 'monthly_climatology_1.40625deg.nc')) as ds:
                    self.baselines_dataset[b_name]['monthly'] = ds.sel(time=self.current_date.replace(year=2019, hour=12, minute=0, second=0))
                # if b_name not in self.baselines_dataset.keys():
                #     dataset = {}
                #     dataset['time'] = xr.load_dataset(os.path.join(self.baselines[b_name], 'time', 'climatology_1.40625deg.nc'))
                #     dataset['weekly'] = xr.load_dataset(os.path.join(self.baselines[b_name], 'weekly', 'weekly_climatology_1.40625deg.nc')).sel(time = self.current_date.replace(hour=12,minute=0,second=0))
                #     dataset['monthly'] = xr.load_dataset(os.path.join(self.baselines[b_name], 'monthly', 'monthly_climatology_1.40625deg.nc')).sel(time = self.current_date.replace(hour=12,minute=0,second=0))
                #     self.baselines_dataset[b_name] = dataset
                fct_hys = {}
                for var_type in self.variable_name:
                    if var_type == 'zos' or var_type == 'sst':
                        fct_hys[var_type] = np.full((1,121,256), np.nan, dtype=np.float32)
                    else:
                        fct_hys[var_type] = np.full((1,23,121,256), np.nan, dtype=np.float32)
            elif b_name == 'hysteresis':
                # parameters

                ymd = (self.current_date - timedelta(days=self.ildy)).strftime('%Y%m%d')
                fct_fp = os.path.join(self.baselines[b_name], f'*{ymd}*.nc')
                fct_list = glob.glob(fct_fp)
                var_list = dict.fromkeys(self.variable_name, None)
                for fn in fct_list:
                    var_type = fn.split('_')[-1].split('.')[0]
                    if var_type in var_list:
                        var_list[var_type] = fn

                dataset = {}
                for var_type in self.variable_name:
                    try:
                        data = xr.load_dataset(var_list[var_type])
                        if var_type == 'zos' or var_type == 'sst':
                            if data.prediction.values.ndim == 4:
                                tempdata = data.prediction.values[:,0,:,:]
                            else:
                                tempdata = data.prediction.values
                        else:
                            tempdata = data.prediction.values
                        dataset[var_type] = tempdata
                        data.close()
                    except Exception as e:
                        print(e)
                        if var_type == 'zos' or var_type == 'sst':
                            dataset[var_type] = np.full((1,121,256), np.nan, dtype=np.float32)
                        else:
                            dataset[var_type] = np.full((1,23,121,256), np.nan, dtype=np.float32)
                self.baselines_dataset[b_name] = dataset

                if self.ildy == 1:
                    fct_hys = None
                else:
                    ymd_hys = (self.current_date - timedelta(days=self.ildy + 1)).strftime('%Y%m%d')
                    fct_hys_fp = os.path.join('/'.join(self.baselines[b_name].split('/')[:-1]) + '/' + str(self.ildy - 1),
                                          f'*{ymd_hys}*.nc')
                    fct_list = glob.glob(fct_hys_fp)
                    var_list = dict.fromkeys(self.variable_name, None)
                    for fn in fct_list:
                        var_type = fn.split('_')[-1].split('.')[0]
                        if var_type in var_list:
                            var_list[var_type] = fn
                    fct_hys = {}
                    for var_type in self.variable_name:
                        try:
                            data = xr.load_dataset(var_list[var_type])
                            if var_type == 'zos' or var_type == 'sst':
                                if data.prediction.values.ndim == 4:
                                    tempdata = data.prediction.values[:,0,:,:]
                                else:
                                    tempdata = data.prediction.values
                            else:
                                tempdata = data.prediction.values
                            fct_hys[var_type] = tempdata
                            data.close()
                        except Exception as e:
                            print(e)
                            if var_type == 'zos' or var_type == 'sst':
                                fct_hys[var_type] = np.full((1,121,256), np.nan, dtype=np.float32)
                            else:
                                fct_hys[var_type] = np.full((1,23,121,256), np.nan, dtype=np.float32)
            elif b_name == 'PSY4':
                ymd = self.current_date.strftime('%Y%m%d')
                fct_fp = os.path.join(self.baselines[b_name], ymd, f'*{ymd}-{ymd}*')
                fct_list = glob.glob(fct_fp)
                var_list = dict.fromkeys(self.variable_name, None)
                for fn in fct_list:
                    file_type = fn.split('/')[-1].split('_')[4]
                    if self.mkt[file_type] == 'uovo':
                        var_list['uo'] = fn
                        var_list['vo'] = fn
                    elif self.mkt[file_type] == 'thetao':
                        var_list[self.mkt[file_type]] = fn
                        var_list['sst'] = fn
                    else:
                        var_list[self.mkt[file_type]] = fn
                dataset = {}
                for var_type in self.variable_name:
                    try:
                        data = xr.load_dataset(var_list[var_type])
                        if var_type == 'zos':
                            dataset[var_type] = data[var_type].values
                        elif var_type == 'sst':
                            dataset[var_type] = data['thetao'].values[:,0,:,:]
                        else:
                            dataset[var_type] = data[var_type].values[:, self.mkt_level, :, :]

                        data.close()
                    except:
                        if var_type == 'zos' or var_type == 'sst':
                            dataset[var_type] = np.full((1, 121, 256), np.nan, dtype=np.float32)
                        else:
                            dataset[var_type] = np.full((1, 23, 121, 256), np.nan, dtype=np.float32)
                self.baselines_dataset[b_name] = dataset

                ymd_hys = (self.current_date - timedelta(days=1)).strftime('%Y%m%d')
                fct_hys_fp = os.path.join('/'.join(self.baselines[b_name].split('/')[:-1]), str(self.ildy - 1), ymd_hys,
                                          f'*{ymd_hys}-{ymd_hys}*')
                fct_list = glob.glob(fct_hys_fp)
                var_list = dict.fromkeys(self.variable_name, None)
                for fn in fct_list:
                    file_type = fn.split('/')[-1].split('_')[4]
                    if self.mkt[file_type] == 'uovo':
                        var_list['uo'] = fn
                        var_list['vo'] = fn
                    elif self.mkt[file_type] == 'thetao':
                        var_list[self.mkt[file_type]] = fn
                        var_list['sst'] = fn
                    else:
                        var_list[self.mkt[file_type]] = fn
                fct_hys = {}
                for var_type in self.variable_name:
                    try:
                        data = xr.load_dataset(var_list[var_type])
                        if var_type == 'zos':
                            fct_hys[var_type] = data[var_type].values
                        elif var_type == 'sst':
                            fct_hys[var_type] = data['thetao'].values[:,0,:,:]
                        else:
                            fct_hys[var_type] = data[var_type].values[:, self.mkt_level, :, :]
                        data.close()
                    except:
                        if var_type == 'zos' or var_type == 'sst':
                            fct_hys[var_type] = np.full((1, 121, 256), np.nan, dtype=np.float32)
                        else:
                            fct_hys[var_type] = np.full((1, 23, 121, 256), np.nan, dtype=np.float32)


            else:

                ymd = self.current_date.strftime('%Y%m%d')
                fct_fp = os.path.join(self.baselines[b_name], f'*{ymd}*.nc')
                fct_list = glob.glob(fct_fp)
                var_list = dict.fromkeys(self.variable_name, None)
                for fn in fct_list:
                    var_type = fn.split('_')[-1].split('.')[0]
                    if var_type in var_list:
                        var_list[var_type] = fn
                # zos,sst,thetao,so,uo,vo
                # sorted_list = sorted(fct_list, key=self.sort_fct)
                # var_list = dict(zip(self.variable_name, sorted_list))
                dataset = {}
                for var_type in self.variable_name:
                    try:
                        data = xr.load_dataset(var_list[var_type])
                        if var_type == 'zos' or var_type == 'sst':
                            if data.prediction.values.ndim == 4:
                                tempdata = data.prediction.values[:, 0, :, :]
                            else:
                                tempdata = data.prediction.values
                        else:
                            tempdata = data.prediction.values
                        dataset[var_type] = tempdata
                        data.close()
                    except Exception as e:
                        print(e)
                        if var_type == 'zos' or var_type == 'sst':
                            dataset[var_type] = np.full((1, 121, 256), np.nan, dtype=np.float32)
                        else:
                            dataset[var_type] = np.full((1, 23, 121, 256), np.nan, dtype=np.float32)
                self.baselines_dataset[b_name] = dataset

                ymd_hys = (self.current_date - timedelta(days=1)).strftime('%Y%m%d')
                fct_hys_fp = os.path.join('/'.join(self.baselines[b_name].split('/')[:-1]), str(self.ildy - 1),
                                          f'*{ymd_hys}*.nc')
                fct_list = glob.glob(fct_hys_fp)
                var_list = dict.fromkeys(self.variable_name, None)
                for fn in fct_list:
                    var_type = fn.split('_')[-1].split('.')[0]
                    if var_type in var_list:
                        var_list[var_type] = fn
                fct_hys = {}
                for var_type in self.variable_name:
                    try:
                        data = xr.load_dataset(var_list[var_type])
                        if var_type == 'zos' or var_type == 'sst':
                            if data.prediction.values.ndim == 4:
                                tempdata = data.prediction.values[:, 0, :, :]
                            else:
                                tempdata = data.prediction.values
                        else:
                            tempdata = data.prediction.values
                        fct_hys[var_type] = tempdata
                        data.close()
                    except Exception as e:
                        print(e)
                        if var_type == 'zos' or var_type == 'sst':
                            fct_hys[var_type] = np.full((1, 121, 256), np.nan, dtype=np.float32)
                        else:
                            fct_hys[var_type] = np.full((1, 23, 121, 256), np.nan, dtype=np.float32)
        return self.baselines_dataset, {b_name+'_hys': fct_hys}

    def __del__(self):
        self.gdp_data.close()
        # for v in self.baselines_dataset.values():
        #     for data in v.values():
        #         data.close()


if __name__ == '__main__':
    '''
    Params: start_date,
            end_date,
            issave_interp: True/False,
            ildy: lead_time
            baselines: correspond to the lead time, {b_name: file_path_list}

    '''
    is_en4 = True
    is_gdp = True
    is_zos = True
    start_date = '20220101'
    end_date = '20231231'
    issave_interp = True
    iscal_metrics = True
    ildy = 10
    baselines = {'XiHe': '~/xihe_prediction/10'}

    variable_name = ['zos', 'sst', 'thetao', 'so', 'uo', 'vo']

    obs_dir = '/hpcfs/fhome/yangjh5/jhm_data/ground_truth'
    db = collect_data(obs_dir, start_date=start_date, end_date=end_date, variable_name=variable_name,
                      baselines=baselines, ildy=ildy)

    strday = datetime(2022, 1, 1)
    endday = datetime(2023, 12, 31)

    nldy = 13

    bounds = [[-90, 90, -180, 180],
              [0, 70, -100, 31],
              [-20, 20, -70, 30],
              [-60, 0, -70, 30],
              [0, 65, 100, -77],
              [-20, 20, 90, -70],
              [-60, 0, 100, -70],
              [-40, 31, 20, 120],
              [-70, 20, 90, 180]]

    nsub = len(bounds)
    blev = [0, 3, 5.5, 8.5, 12.5, 17.5, 30, 50, 75, 95, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5, 287.5, 325,
            375, 425, 475, 525, 575, 622, 0]
    ulev = [3, 5.5, 8.5, 12.5, 17.5, 30, 50, 75, 95, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5, 287.5, 325, 375,
            425, 475, 525, 575, 622, 644, 644]
    levs = [i for i in range(len(blev))]

    lons = np.arange(-179, 180, 2)
    lats = np.arange(-89, 90, 2)
    nlat = len(lats)
    nlon = len(lons)

    s_lat = np.load('/hpcfs/fhome/yangjh5/jhm_data/latlon/1.40625deg/lat.npz')['arr_0']
    s_lon = np.load('/hpcfs/fhome/yangjh5/jhm_data/latlon/1.40625deg/lon.npz')['arr_0']
    s_depth = np.array(
        [0.4940, 2.6457, 5.0782, 7.9296, 11.4050,
         15.8101, 21.5988, 29.4447, 40.3441,
         55.7643, 77.8539, 92.3261, 109.7293,
         130.6660, 155.8507, 186.1256, 222.4752,
         266.0403, 318.1274, 380.2130, 453.9377,
         541.0889, 643.5668])

    if list(baselines.keys())[0] == 'climatology':
        now = datetime.now()
        now_str = now.strftime('%Y%m%d')
        save_path = f'./data/{list(baselines.keys())[0]+"_"+now_str}/interpresults'
        save_path_metrics = f'./data/{list(baselines.keys())[0]+"_"+now_str}/metricsresults'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_metrics, exist_ok=True)

        gmetrics_en4 = np.empty((3, 2, 5, 0))
        hmetrics_en4 = np.empty((3, 2, 5, 30 * 64, 0))
        smetrics_en4 = np.empty((3, 2, 5, 9, 0))
        vmetrics_en4 = np.empty((3, 2, 5, len(levs), 0))

        gmetrics_gdp = np.empty((3, 3, 5, 0))
        hmetrics_gdp = np.empty((3, 3, 5, 30 * 64, 0))
        smetrics_gdp = np.empty((3, 3, 5, 9, 0))

        gmetrics_zos = np.empty((3, 1, 5, 0))
        hmetrics_zos = np.empty((3, 1, 5, 30 * 64, 0))
        smetrics_zos = np.empty((3, 1, 5, 9, 0))
    else:
        now = datetime.now()
        now_str = now.strftime('%Y%m%d')
        save_path = f'./data/{list(baselines.keys())[0]+"_"+now_str}/{ildy}/interpresults'
        save_path_metrics = f'./data/{list(baselines.keys())[0]+"_"+now_str}/{ildy}/metricsresults'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_metrics, exist_ok=True)

        gmetrics_en4 = np.empty((2, 5, 0))
        hmetrics_en4 = np.empty((2, 5, 30 * 64, 0))
        smetrics_en4 = np.empty((2, 5, 9, 0))
        vmetrics_en4 = np.empty((2, 5, len(levs), 0))

        gmetrics_gdp = np.empty((3, 5, 0))
        hmetrics_gdp = np.empty((3, 5, 30 * 64, 0))
        smetrics_gdp = np.empty((3, 5, 9, 0))

        gmetrics_zos = np.empty((1, 5, 0))
        hmetrics_zos = np.empty((1, 5, 30 * 64, 0))
        smetrics_zos = np.empty((1, 5, 9, 0))

    for en4_data, gdp_data, zos_data, fct_data, fct_hys in tqdm(db):
        ymd = (db.current_date - timedelta(1)).strftime('%Y%m%d')
        if is_en4:
            file_path = os.path.join(save_path, f'{ymd}_en4.pkl.gz')
            file_path = Path(file_path)
            if file_path.exists():
                with gzip.open(str(file_path), 'rb') as f:
                    load_data = pickle.load(f)
                    en4_data_filtered = load_data['en4_data_filtered']
                    en4_baseline_data_interp = load_data['en4_baseline_data_interp']
                    t_grid = load_data['t_grid']
                    t_depth = load_data['t_depth']
            else:
                if en4_data != None:
                    en4_data_filtered, en4_baseline_data_interp, t_grid, t_depth = interp_TSP(en4_data, fct_data, s_lat,
                                                                                              s_lon, s_depth)
                else:
                    en4_data_filtered, en4_baseline_data_interp, t_grid, t_depth = None, None, None, None
                data_to_save = {
                    'en4_data_filtered': en4_data_filtered,
                    'en4_baseline_data_interp': en4_baseline_data_interp,
                    't_grid': t_grid,
                    't_depth': t_depth
                }
                if issave_interp:
                    with gzip.open(str(file_path), 'wb') as f:
                        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            if iscal_metrics:
                if ildy == 1 and list(baselines.keys())[0] != 'climatology':
                    try:
                        hys_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/hysteresis/{ildy}/interpresults/{ymd}_en4.pkl.gz'
                        with gzip.open(hys_fp, 'rb') as f:
                            load_data = pickle.load(f)
                            hys = load_data['en4_baseline_data_interp']['hysteresis']
                    except:
                        if en4_data != None:
                            shapes = en4_baseline_data_interp[list(baselines.keys())[0]]['potm'].shape
                            hys = {'potm': np.full(shapes, np.nan), 'psal': np.full(shapes, np.nan)}
                        else:
                            hys = None
                else:
                    if en4_data != None:
                        _, hys, _, _ = interp_TSP(en4_data, fct_hys, s_lat, s_lon, s_depth)
                        hys = hys[list(baselines.keys())[0]+'_hys']
                    else:
                        hys = None
                try:
                    clm_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/climatology/interpresults/{ymd}_en4.pkl.gz'
                    with gzip.open(clm_fp, 'rb') as f:
                        load_data = pickle.load(f)
                        clm = load_data['en4_baseline_data_interp']['climatology']['time']
                except:
                    if list(baselines.keys())[0] == 'climatology':
                        if en4_data != None:
                            _, clm, _, _ = interp_TSP(en4_data, fct_data, s_lat, s_lon, s_depth)
                            clm = clm['climatology']['time']
                        else:
                            clm = None
                    else:
                        if en4_data != None:
                            shapes = en4_baseline_data_interp[list(baselines.keys())[0]]['potm'].shape
                            clm = {'potm': np.full(shapes, np.nan), 'psal': np.full(shapes, np.nan)}
                        else:
                            clm = None
                tsp_metrics = evaluate_TSP(t_grid, t_depth, en4_data_filtered, en4_baseline_data_interp, hys, clm)

                gmetrics_en4 = np.concatenate([gmetrics_en4, np.expand_dims(tsp_metrics[0], axis=-1)], axis=-1)
                hmetrics_en4 = np.concatenate([hmetrics_en4, np.expand_dims(tsp_metrics[1], axis=-1)], axis=-1)
                smetrics_en4 = np.concatenate([smetrics_en4, np.expand_dims(tsp_metrics[2], axis=-1)], axis=-1)
                vmetrics_en4 = np.concatenate([vmetrics_en4, np.expand_dims(tsp_metrics[3], axis=-1)], axis=-1)
        if is_gdp:
            file_path = os.path.join(save_path, f'{ymd}_gdp.pkl.gz')
            file_path = Path(file_path)
            if file_path.exists():
                with gzip.open(str(file_path), 'rb') as f:
                    load_data = pickle.load(f)
                    gdp_data_filtered = load_data['gdp_data_filtered']
                    gdp_baseline_data_interp = load_data['gdp_baseline_data_interp']
                    t_grid = load_data['t_grid']

            else:
                if gdp_data != None:
                    gdp_data.ve[gdp_data.ve < -9000] = np.nan
                    gdp_data.vn[gdp_data.vn < -9000] = np.nan
                    gdp_data_filtered, gdp_baseline_data_interp, t_grid = interp_GDP(gdp_data, fct_data, s_lat, s_lon,
                                                                                     s_depth)
                else:
                    gdp_data_filtered, gdp_baseline_data_interp, t_grid = None, None, None
                #
                data_to_save = {
                    'gdp_data_filtered': gdp_data_filtered,
                    'gdp_baseline_data_interp': gdp_baseline_data_interp,
                    't_grid': t_grid
                }
                if issave_interp:
                    with gzip.open(str(file_path), 'wb') as f:
                        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            if iscal_metrics:
                if ildy == 1 and list(baselines.keys())[0] != 'climatology':
                    try:
                        hys_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/hysteresis/{ildy}/interpresults/{ymd}_gdp.pkl.gz'
                        with gzip.open(hys_fp, 'rb') as f:
                            load_data = pickle.load(f)
                            hys = load_data['gdp_baseline_data_interp']['hysteresis']
                    except:
                        if gdp_data != None:
                            shapes_sst = gdp_baseline_data_interp[list(baselines.keys())[0]]['sst'].shape
                            shapes = gdp_baseline_data_interp[list(baselines.keys())[0]]['uo'].shape
                            hys = {'sst': np.full(shapes_sst, np.nan), 'uo': np.full(shapes, np.nan),
                                   'vo': np.full(shapes, np.nan)}
                        else:
                            hys = None
                else:
                    if gdp_data != None:
                        _, hys, _ = interp_GDP(gdp_data, fct_hys, s_lat, s_lon, s_depth)
                        hys = hys[list(baselines.keys())[0]+'_hys']
                    else:
                        hys = None

                try:
                    clm_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/climatology/interpresults/{ymd}_gdp.pkl.gz'
                    with gzip.open(clm_fp, 'rb') as f:
                        load_data = pickle.load(f)
                        clm = load_data['gdp_baseline_data_interp']['climatology']['time']
                except:
                    if list(baselines.keys())[0] == 'climatology':
                        if gdp_data != None:
                            _, clm, _ = interp_GDP(gdp_data, fct_data, s_lat, s_lon, s_depth)
                            clm = clm['climatology']['time']
                        else:
                            clm = None
                    else:
                        if gdp_data != None:
                            shapes_sst = gdp_baseline_data_interp[list(baselines.keys())[0]]['sst'].shape
                            shapes = gdp_baseline_data_interp[list(baselines.keys())[0]]['uo'].shape
                            clm = {'sst': np.full(shapes_sst, np.nan), 'uo': np.full(shapes, np.nan), 'vo': np.full(shapes, np.nan)}
                        else:
                            clm = None
                gdp_metrics = evaluate_GDP(t_grid, gdp_data_filtered, gdp_baseline_data_interp, hys, clm)
                gmetrics_gdp = np.concatenate([gmetrics_gdp, np.expand_dims(gdp_metrics[0], axis=-1)], axis=-1)
                hmetrics_gdp = np.concatenate([hmetrics_gdp, np.expand_dims(gdp_metrics[1], axis=-1)], axis=-1)
                smetrics_gdp = np.concatenate([smetrics_gdp, np.expand_dims(gdp_metrics[2], axis=-1)], axis=-1)
        if is_zos:
            file_path = os.path.join(save_path, f'{ymd}_zos.pkl.gz')
            file_path = Path(file_path)
            if file_path.exists():
                with gzip.open(str(file_path), 'rb') as f:
                    load_data = pickle.load(f)
                    zos_data_filtered = load_data['zos_data_filtered']
                    zos_baseline_data_interp = load_data['zos_baseline_data_interp']
                    t_grid = load_data['t_grid']
            else:
                if zos_data != None:
                    zos_data_filtered, zos_baseline_data_interp, t_grid = interp_ZOS(zos_data, fct_data, s_lat, s_lon)
                else:
                    zos_data_filtered, zos_baseline_data_interp, t_grid = None, None, None
                data_to_save = {
                    'zos_data_filtered': zos_data_filtered,
                    'zos_baseline_data_interp': zos_baseline_data_interp,
                    't_grid': t_grid
                }
                if issave_interp:
                    with gzip.open(str(file_path), 'wb') as f:
                        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            if iscal_metrics:
                if ildy == 1 and list(baselines.keys())[0] != 'climatology':
                    try:
                        hys_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/hysteresis/{ildy}/interpresults/{ymd}_zos.pkl.gz'
                        with gzip.open(hys_fp, 'rb') as f:
                            load_data = pickle.load(f)
                            hys = load_data['zos_baseline_data_interp']['hysteresis']
                    except:
                        if zos_data != None:
                            shapes = zos_baseline_data_interp[list(baselines.keys())[0]]['zos'].shape
                            hys = {'zos': np.full(shapes, np.nan)}
                        else:
                            hys = None
                else:
                    if zos_data != None:
                        _, hys, _ = interp_ZOS(zos_data, fct_hys, s_lat, s_lon)
                        hys = hys[list(baselines.keys())[0]+'_hys']
                    else:
                        hys = None

                try:
                    clm_fp = f'/hpcfs/fhome/yangjh5/jhm_data/evaluation_oceanbench/evaluate/data/climatology/interpresults/{ymd}_zos.pkl.gz'
                    with gzip.open(clm_fp, 'rb') as f:
                        load_data = pickle.load(f)
                        clm = load_data['zos_baseline_data_interp']['climatology']['time']
                except:
                    if list(baselines.keys())[0] == 'climatology':
                        if zos_data != None:
                            _, clm, _ = interp_ZOS(zos_data, fct_data, s_lat, s_lon)
                            clm = clm['climatology']['time']
                        else:
                            clm = None
                    else:
                        if zos_data != None:
                            shapes = zos_baseline_data_interp[list(baselines.keys())[0]]['zos'].shape
                            clm = {'zos': np.full(shapes, np.nan)}
                        else:
                            clm = None
                zos_metrics = evaluate_ZOS(t_grid, zos_data_filtered, zos_baseline_data_interp, hys, clm)
                gmetrics_zos = np.concatenate([gmetrics_zos, np.expand_dims(zos_metrics[0], axis=-1)], axis=-1)
                hmetrics_zos = np.concatenate([hmetrics_zos, np.expand_dims(zos_metrics[1], axis=-1)], axis=-1)
                smetrics_zos = np.concatenate([smetrics_zos, np.expand_dims(zos_metrics[2], axis=-1)], axis=-1)
    if iscal_metrics:
        if is_en4:
            results_fp = os.path.join(save_path_metrics, 'EN4_DAYS.npz')
            np.savez(results_fp, gmetrics_en4=gmetrics_en4, hmetrics_en4=hmetrics_en4, smetrics_en4=smetrics_en4,
                     vmetrics_en4=vmetrics_en4)
        if is_gdp:
            results_fp = os.path.join(save_path_metrics, 'GDP_DAYS.npz')
            np.savez(results_fp, gmetrics_gdp=gmetrics_gdp, hmetrics_gdp=hmetrics_gdp, smetrics_gdp=smetrics_gdp)
        if is_zos:
            results_fp = os.path.join(save_path_metrics, 'ZOS_DAYS.npz')
            np.savez(results_fp, gmetrics_zos=gmetrics_zos, hmetrics_zos=hmetrics_zos, smetrics_zos=smetrics_zos)
        if is_en4:
            gmetrics_en4 = np.nanmean(gmetrics_en4, axis=-1)
            hmetrics_en4 = np.nanmean(hmetrics_en4, axis=-1)
            smetrics_en4 = np.nanmean(smetrics_en4, axis=-1)
            vmetrics_en4 = np.nanmean(vmetrics_en4, axis=-1)
            results_fp = os.path.join(save_path_metrics, 'EN4.npz')
            np.savez(results_fp, gmetrics_en4=gmetrics_en4, hmetrics_en4=hmetrics_en4, smetrics_en4=smetrics_en4,vmetrics_en4=vmetrics_en4)
        if is_gdp:
            gmetrics_gdp = np.nanmean(gmetrics_gdp, axis=-1)
            hmetrics_gdp = np.nanmean(hmetrics_gdp, axis=-1)
            smetrics_gdp = np.nanmean(smetrics_gdp, axis=-1)
            results_fp = os.path.join(save_path_metrics, 'GDP.npz')
            np.savez(results_fp, gmetrics_gdp=gmetrics_gdp, hmetrics_gdp=hmetrics_gdp, smetrics_gdp=smetrics_gdp)
        if is_zos:
            gmetrics_zos = np.nanmean(gmetrics_zos, axis=-1)
            hmetrics_zos = np.nanmean(hmetrics_zos, axis=-1)
            smetrics_zos = np.nanmean(smetrics_zos, axis=-1)
            results_fp = os.path.join(save_path_metrics, 'ZOS.npz')
            np.savez(results_fp, gmetrics_zos=gmetrics_zos, hmetrics_zos=hmetrics_zos, smetrics_zos=smetrics_zos)






