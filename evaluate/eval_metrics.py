# Copyright (c) 2024 College of Meteorology and Oceanography, National University of Defense Technology. All rights reserved
# Code adapted from:
# https://github.com/beijixiong1/OceanForecastBench/

import ipdb
import numpy as np
from cal_metrics import *
import builtins
def cal_global_metrics_ZOS(zos_data_filtered, baseline_data_interp, hys, clm):
    grmse = {}; gbias = {}; gacc = {}; gpss={}; gcss = {}
    hys = hys['zos']
    clm = clm['zos']
    for b_name in baseline_data_interp:
        if b_name == 'climatology':

            fct_t = baseline_data_interp['climatology']['time']['zos']
            fct_w = baseline_data_interp['climatology']['weekly']['zos']
            fct_m = baseline_data_interp['climatology']['monthly']['zos']
            obs = zos_data_filtered['zos']

            grmse[b_name] = {}
            grmse[b_name]['time'] = {}
            grmse[b_name]['time']['zos'] = crmse(obs, fct_t)

            grmse[b_name]['weekly'] = {}
            grmse[b_name]['weekly']['zos'] = crmse(obs, fct_w)

            grmse[b_name]['monthly'] = {}
            grmse[b_name]['monthly']['zos'] = crmse(obs, fct_m)

            gbias[b_name] = {}
            gbias[b_name]['time'] = {}
            gbias[b_name]['time']['zos'] = cbias(obs, fct_t)

            gbias[b_name]['weekly'] = {}
            gbias[b_name]['weekly']['zos'] = cbias(obs, fct_w)

            gbias[b_name]['monthly'] = {}
            gbias[b_name]['monthly']['zos'] = cbias(obs, fct_m)

            gacc[b_name] = {}
            gacc[b_name]['time'] = {}
            gacc[b_name]['time']['zos'] = cacc(obs, fct_t, clm)

            gacc[b_name]['weekly'] = {}
            gacc[b_name]['weekly']['zos'] = cacc(obs, fct_w, clm)

            gacc[b_name]['monthly'] = {}
            gacc[b_name]['monthly']['zos'] = cacc(obs, fct_m, clm)

            gpss[b_name] = {}
            gpss[b_name]['time'] = {}
            gpss[b_name]['time']['zos'] = cpss(obs, fct_t, hys)

            gpss[b_name]['weekly'] = {}
            gpss[b_name]['weekly']['zos'] = cpss(obs, fct_w, hys)

            gpss[b_name]['monthly'] = {}
            gpss[b_name]['monthly']['zos'] = cpss(obs, fct_m, hys)

            gcss[b_name] = {}
            gcss[b_name]['time'] = {}
            gcss[b_name]['time']['zos'] = ccss(obs, fct_t, clm)

            gcss[b_name]['weekly'] = {}
            gcss[b_name]['weekly']['zos'] = ccss(obs, fct_w, clm)

            gcss[b_name]['monthly'] = {}
            gcss[b_name]['monthly']['zos'] = ccss(obs, fct_m, clm)

            gmetrics = np.zeros((3,1,5))
        # elif b_name == 'PSY4':
        #     pass
        else:
            fct = baseline_data_interp[b_name]['zos']
            obs = zos_data_filtered['zos']
            grmse[b_name] = {}
            grmse[b_name]['zos'] = crmse(obs, fct)
            gbias[b_name] = {}
            gbias[b_name]['zos'] = cbias(obs, fct)
            gacc[b_name] = {}
            gacc[b_name]['zos'] = cacc(obs, fct, clm)
            gpss[b_name] = {}
            gpss[b_name]['zos'] = cpss(obs, fct, hys)
            gcss[b_name] = {}
            gcss[b_name]['zos'] = ccss(obs, fct, clm)
            gmetrics = np.zeros((1,5))
    if builtins.list(grmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        for m, t in enumerate(type):
            gmetrics[m, 0, 0] = grmse['climatology'][t]['zos']
            gmetrics[m, 0, 1] = gacc['climatology'][t]['zos']
            gmetrics[m, 0, 2] = gbias['climatology'][t]['zos']
            gmetrics[m, 0, 3] = gpss['climatology'][t]['zos']
            gmetrics[m, 0, 4] = gcss['climatology'][t]['zos']
    else:
        gmetrics[0, 0] = builtins.list(grmse.values())[0]['zos']
        gmetrics[0, 1] = builtins.list(gacc.values())[0]['zos']
        gmetrics[0, 2] = builtins.list(gbias.values())[0]['zos']
        gmetrics[0, 3] = builtins.list(gpss.values())[0]['zos']
        gmetrics[0, 4] = builtins.list(gcss.values())[0]['zos']

    return gmetrics
def cal_global_metrics_GDP(gdp_data_filtered, baseline_data_interp, hys, clm):
    grmse = {};
    gbias = {};
    gacc = {};
    gpss = {};
    gcss = {}

    for b_name in baseline_data_interp:
        if b_name == 'climatology':
            hys_sst = hys['sst']
            clm_sst = clm['sst']
            fct_t = baseline_data_interp['climatology']['time']['sst']
            fct_w = baseline_data_interp['climatology']['weekly']['sst']
            fct_m = baseline_data_interp['climatology']['monthly']['sst']
            obs = gdp_data_filtered['sst']

            grmse[b_name] = {}
            grmse[b_name]['time'] = {}
            grmse[b_name]['time']['sst'] = crmse(obs, fct_t)

            grmse[b_name]['weekly'] = {}
            grmse[b_name]['weekly']['sst'] = crmse(obs, fct_w)

            grmse[b_name]['monthly'] = {}
            grmse[b_name]['monthly']['sst'] = crmse(obs, fct_m)

            gbias[b_name] = {}
            gbias[b_name]['time'] = {}
            gbias[b_name]['time']['sst'] = cbias(obs, fct_t)

            gbias[b_name]['weekly'] = {}
            gbias[b_name]['weekly']['sst'] = cbias(obs, fct_w)

            gbias[b_name]['monthly'] = {}
            gbias[b_name]['monthly']['sst'] = cbias(obs, fct_m)

            gacc[b_name] = {}
            gacc[b_name]['time'] = {}
            gacc[b_name]['time']['sst'] = cacc(obs, fct_t, clm_sst)

            gacc[b_name]['weekly'] = {}
            gacc[b_name]['weekly']['sst'] = cacc(obs, fct_w, clm_sst)

            gacc[b_name]['monthly'] = {}
            gacc[b_name]['monthly']['sst'] = cacc(obs, fct_m, clm_sst)

            gpss[b_name] = {}
            gpss[b_name]['time'] = {}
            gpss[b_name]['time']['sst'] = cpss(obs, fct_t, hys_sst)

            gpss[b_name]['weekly'] = {}
            gpss[b_name]['weekly']['sst'] = cpss(obs, fct_w, hys_sst)

            gpss[b_name]['monthly'] = {}
            gpss[b_name]['monthly']['sst'] = cpss(obs, fct_m, hys_sst)

            gcss[b_name] = {}
            gcss[b_name]['time'] = {}
            gcss[b_name]['time']['sst'] = ccss(obs, fct_t, clm_sst)

            gcss[b_name]['weekly'] = {}
            gcss[b_name]['weekly']['sst'] = ccss(obs, fct_w, clm_sst)

            gcss[b_name]['monthly'] = {}
            gcss[b_name]['monthly']['sst'] = ccss(obs, fct_m, clm_sst)

            hys_uo = hys['uo'][0,:]
            clm_uo = clm['uo'][0,:]
            fct_t = baseline_data_interp['climatology']['time']['uo'][0,:]
            fct_w = baseline_data_interp['climatology']['weekly']['uo'][0,:]
            fct_m = baseline_data_interp['climatology']['monthly']['uo'][0,:]
            obs = gdp_data_filtered['uo']
            grmse[b_name]['time']['uo'] = crmse(obs, fct_t)
            grmse[b_name]['weekly']['uo'] = crmse(obs, fct_w)
            grmse[b_name]['monthly']['uo'] = crmse(obs, fct_m)
            gbias[b_name]['time']['uo'] = cbias(obs, fct_t)
            gbias[b_name]['weekly']['uo'] = cbias(obs, fct_w)
            gbias[b_name]['monthly']['uo'] = cbias(obs, fct_m)
            gacc[b_name]['time']['uo'] = cacc(obs, fct_t, clm_uo)
            gacc[b_name]['weekly']['uo'] = cacc(obs, fct_w, clm_uo)
            gacc[b_name]['monthly']['uo'] = cacc(obs, fct_m, clm_uo)
            gpss[b_name]['time']['uo'] = cpss(obs, fct_t, hys_uo)
            gpss[b_name]['weekly']['uo'] = cpss(obs, fct_w, hys_uo)
            gpss[b_name]['monthly']['uo'] = cpss(obs, fct_m, hys_uo)
            gcss[b_name]['time']['uo'] = ccss(obs, fct_t, clm_uo)
            gcss[b_name]['weekly']['uo'] = ccss(obs, fct_w, clm_uo)
            gcss[b_name]['monthly']['uo'] = ccss(obs, fct_m, clm_uo)

            hys_vo = hys['vo'][0,:]
            clm_vo = clm['vo'][0,:]
            fct_t = baseline_data_interp['climatology']['time']['vo'][0,:]
            fct_w = baseline_data_interp['climatology']['weekly']['vo'][0,:]
            fct_m = baseline_data_interp['climatology']['monthly']['vo'][0,:]
            obs = gdp_data_filtered['vo']
            grmse[b_name]['time']['vo'] = crmse(obs, fct_t)
            grmse[b_name]['weekly']['vo'] = crmse(obs, fct_w)
            grmse[b_name]['monthly']['vo'] = crmse(obs, fct_m)
            gbias[b_name]['time']['vo'] = cbias(obs, fct_t)
            gbias[b_name]['weekly']['vo'] = cbias(obs, fct_w)
            gbias[b_name]['monthly']['vo'] = cbias(obs, fct_m)
            gacc[b_name]['time']['vo'] = cacc(obs, fct_t, clm_vo)
            gacc[b_name]['weekly']['vo'] = cacc(obs, fct_w, clm_vo)
            gacc[b_name]['monthly']['vo'] = cacc(obs, fct_m, clm_vo)
            gpss[b_name]['time']['vo'] = cpss(obs, fct_t, hys_vo)
            gpss[b_name]['weekly']['vo'] = cpss(obs, fct_w, hys_vo)
            gpss[b_name]['monthly']['vo'] = cpss(obs, fct_m, hys_vo)
            gcss[b_name]['time']['vo'] = ccss(obs, fct_t, clm_vo)
            gcss[b_name]['weekly']['vo'] = ccss(obs, fct_w, clm_vo)
            gcss[b_name]['monthly']['vo'] = ccss(obs, fct_m, clm_vo)

            gmetrics = np.zeros((3, 3, 5))
        # elif b_name == 'PSY4':
        #     pass
        else:
            hys_sst = hys['sst']
            clm_sst = clm['sst']
            fct = baseline_data_interp[b_name]['sst']
            obs = gdp_data_filtered['sst']
            grmse[b_name] = {}
            grmse[b_name]['sst'] = crmse(obs, fct)
            gbias[b_name] = {}
            gbias[b_name]['sst'] = cbias(obs, fct)
            gacc[b_name] = {}
            gacc[b_name]['sst'] = cacc(obs, fct, clm_sst)
            gpss[b_name] = {}
            gpss[b_name]['sst'] = cpss(obs, fct, hys_sst)
            gcss[b_name] = {}
            gcss[b_name]['sst'] = ccss(obs, fct, clm_sst)


            hys_uo = hys['uo'][0,:]
            clm_uo = clm['uo'][0,:]
            fct = baseline_data_interp[b_name]['uo'][0,:]
            obs = gdp_data_filtered['uo']
            grmse[b_name]['uo'] = crmse(obs, fct)
            gbias[b_name]['uo'] = cbias(obs, fct)
            gacc[b_name]['uo'] = cacc(obs, fct, clm_uo)
            gpss[b_name]['uo'] = cpss(obs, fct, hys_uo)
            gcss[b_name]['uo'] = ccss(obs, fct, clm_uo)

            hys_vo = hys['vo'][0,:]
            clm_vo = clm['vo'][0,:]
            fct = baseline_data_interp[b_name]['vo'][0,:]
            obs = gdp_data_filtered['vo']
            grmse[b_name]['vo'] = crmse(obs, fct)
            gbias[b_name]['vo'] = cbias(obs, fct)
            gacc[b_name]['vo'] = cacc(obs, fct, clm_vo)
            gpss[b_name]['vo'] = cpss(obs, fct, hys_vo)
            gcss[b_name]['vo'] = ccss(obs, fct, clm_vo)

            gmetrics = np.zeros((3, 5))
    if builtins.list(grmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        var_type = ['sst', 'uo', 'vo']
        for m, t in enumerate(type):
            for n, vt in enumerate(var_type):
                gmetrics[m, n, 0] = grmse['climatology'][t][vt]
                gmetrics[m, n, 1] = gacc['climatology'][t][vt]
                gmetrics[m, n, 2] = gbias['climatology'][t][vt]
                gmetrics[m, n, 3] = gpss['climatology'][t][vt]
                gmetrics[m, n, 4] = gcss['climatology'][t][vt]
    else:
        var_type = ['sst', 'uo', 'vo']
        for n, vt in enumerate(var_type):
            gmetrics[n, 0] = builtins.list(grmse.values())[0][vt]
            gmetrics[n, 1] = builtins.list(gacc.values())[0][vt]
            gmetrics[n, 2] = builtins.list(gbias.values())[0][vt]
            gmetrics[n, 3] = builtins.list(gpss.values())[0][vt]
            gmetrics[n, 4] = builtins.list(gcss.values())[0][vt]

    return gmetrics
def cal_global_metrics_TSP(en4_data_filtered, baseline_data_interp, hys, clm):
    grmse = {};
    gbias = {};
    gacc = {};
    gpss = {};
    gcss = {}
    for b_name in baseline_data_interp:
        if b_name == 'climatology':
            hys_potm = hys['potm']
            clm_potm = clm['potm']
            fct_t = baseline_data_interp['climatology']['time']['potm']
            fct_w = baseline_data_interp['climatology']['weekly']['potm']
            fct_m = baseline_data_interp['climatology']['monthly']['potm']
            obs = en4_data_filtered['potm'].T

            grmse[b_name] = {}
            grmse[b_name]['time'] = {}
            grmse[b_name]['time']['potm'] = crmse(obs, fct_t)

            grmse[b_name]['weekly'] = {}
            grmse[b_name]['weekly']['potm'] = crmse(obs, fct_w)

            grmse[b_name]['monthly'] = {}
            grmse[b_name]['monthly']['potm'] = crmse(obs, fct_m)

            gbias[b_name] = {}
            gbias[b_name]['time'] = {}
            gbias[b_name]['time']['potm'] = cbias(obs, fct_t)

            gbias[b_name]['weekly'] = {}
            gbias[b_name]['weekly']['potm'] = cbias(obs, fct_w)

            gbias[b_name]['monthly'] = {}
            gbias[b_name]['monthly']['potm'] = cbias(obs, fct_m)

            gacc[b_name] = {}
            gacc[b_name]['time'] = {}
            gacc[b_name]['time']['potm'] = cacc(obs, fct_t, clm_potm)

            gacc[b_name]['weekly'] = {}
            gacc[b_name]['weekly']['potm'] = cacc(obs, fct_w, clm_potm)

            gacc[b_name]['monthly'] = {}
            gacc[b_name]['monthly']['potm'] = cacc(obs, fct_m, clm_potm)

            gpss[b_name] = {}
            gpss[b_name]['time'] = {}
            gpss[b_name]['time']['potm'] = cpss(obs, fct_t, hys_potm)

            gpss[b_name]['weekly'] = {}
            gpss[b_name]['weekly']['potm'] = cpss(obs, fct_w, hys_potm)

            gpss[b_name]['monthly'] = {}
            gpss[b_name]['monthly']['potm'] = cpss(obs, fct_m, hys_potm)

            gcss[b_name] = {}
            gcss[b_name]['time'] = {}
            gcss[b_name]['time']['potm'] = ccss(obs, fct_t, clm_potm)

            gcss[b_name]['weekly'] = {}
            gcss[b_name]['weekly']['potm'] = ccss(obs, fct_w, clm_potm)

            gcss[b_name]['monthly'] = {}
            gcss[b_name]['monthly']['potm'] = ccss(obs, fct_m, clm_potm)

            #盐度
            hys_psal = hys['psal']
            clm_psal = clm['psal']
            fct_t = baseline_data_interp['climatology']['time']['psal']
            fct_w = baseline_data_interp['climatology']['weekly']['psal']
            fct_m = baseline_data_interp['climatology']['monthly']['psal']
            obs = en4_data_filtered['psal'].T


            grmse[b_name]['time']['psal'] = crmse(obs, fct_t)
            grmse[b_name]['weekly']['psal'] = crmse(obs, fct_w)
            grmse[b_name]['monthly']['psal'] = crmse(obs, fct_m)

            gbias[b_name]['time']['psal'] = cbias(obs, fct_t)
            gbias[b_name]['weekly']['psal'] = cbias(obs, fct_w)
            gbias[b_name]['monthly']['psal'] = cbias(obs, fct_m)

            gacc[b_name]['time']['psal'] = cacc(obs, fct_t, clm_psal)
            gacc[b_name]['weekly']['psal'] = cacc(obs, fct_w, clm_psal)
            gacc[b_name]['monthly']['psal'] = cacc(obs, fct_m, clm_psal)

            gpss[b_name]['time']['psal'] = cpss(obs, fct_t, hys_psal)
            gpss[b_name]['weekly']['psal'] = cpss(obs, fct_w, hys_psal)
            gpss[b_name]['monthly']['psal'] = cpss(obs, fct_m, hys_psal)

            gcss[b_name]['time']['psal'] = ccss(obs, fct_t, clm_psal)
            gcss[b_name]['weekly']['psal'] = ccss(obs, fct_w, clm_psal)
            gcss[b_name]['monthly']['psal'] = ccss(obs, fct_m, clm_psal)

            gmetrics = np.zeros((3, 2, 5))
        # elif b_name == 'PSY4':
        #     pass
        else:
            hys_potm = hys['potm']
            clm_potm = clm['potm']
            fct = baseline_data_interp[b_name]['potm']
            obs = en4_data_filtered['potm'].T
            numeval = (~np.isnan(fct -obs)).sum()
            print('numeval:',numeval)
            grmse[b_name] = {}
            grmse[b_name]['potm'], mask = crmse_mask(obs, fct)
            print('mask:',mask.sum())
            gbias[b_name] = {}
            gbias[b_name]['potm'] = cbias(obs, fct)
            gacc[b_name] = {}
            gacc[b_name]['potm'] = cacc_mask(obs, fct, clm_potm, mask)
            gpss[b_name] = {}
            gpss[b_name]['potm'] = cpss(obs, fct, hys_potm)
            gcss[b_name] = {}
            gcss[b_name]['potm'] = ccss(obs, fct, clm_potm)

            hys_psal = hys['psal']
            clm_psal = clm['psal']
            fct = baseline_data_interp[b_name]['psal']
            obs = en4_data_filtered['psal'].T
            grmse[b_name]['psal'], mask = crmse_psal(obs, fct)
            gbias[b_name]['psal'] = cbias(obs, fct)
            gacc[b_name]['psal'] = cacc_mask(obs, fct, clm_psal, mask)
            gpss[b_name]['psal'] = cpss(obs, fct, hys_psal)
            gcss[b_name]['psal'] = ccss(obs, fct, clm_psal)
            gmetrics = np.zeros((2, 5))
    if builtins.list(grmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        var_type = ['potm', 'psal']
        for m,t in enumerate(type):
            for n,vt in enumerate(var_type):
                gmetrics[m, n, 0] = grmse['climatology'][t][vt]
                gmetrics[m, n, 1] = gacc['climatology'][t][vt]
                gmetrics[m, n, 2] = gbias['climatology'][t][vt]
                gmetrics[m, n, 3] = gpss['climatology'][t][vt]
                gmetrics[m, n, 4] = gcss['climatology'][t][vt]
    else:
        var_type = ['potm', 'psal']
        for n, vt in enumerate(var_type):
            gmetrics[n, 0] = builtins.list(grmse.values())[0][vt]
            gmetrics[n, 1] = builtins.list(gacc.values())[0][vt]
            gmetrics[n, 2] = builtins.list(gbias.values())[0][vt]
            gmetrics[n, 3] = builtins.list(gpss.values())[0][vt]
            gmetrics[n, 4] = builtins.list(gcss.values())[0][vt]

    return gmetrics
#global指标计算应该没问题， acc计算的是时间序列的相关性(错的）空间平均
def cal_global_rmse():
    pass
def cal_horizontal_metrics_ZOS(zos_data_filtered, baseline_data_interp, hys, clm, t_grid):
    hrmse = {};hbias = {};hacc = {};hpss = {};hcss = {}
    lat = np.arange(-80 + 5.625/2, 90, 5.625)
    lon = np.arange(-180, 180, 5.625)
    grid = np.meshgrid(lon, lat)
    grid = np.column_stack((grid[1].ravel(), grid[0].ravel()))

    for b_name in baseline_data_interp:
        if b_name == 'climatology':
            hys = hys['zos']
            clm = clm['zos']
            fct_t = baseline_data_interp['climatology']['time']['zos']
            fct_w = baseline_data_interp['climatology']['weekly']['zos']
            fct_m = baseline_data_interp['climatology']['monthly']['zos']
            obs = zos_data_filtered['zos']

            hrmse[b_name] = {}
            hrmse[b_name]['time'] = {}
            hrmse[b_name]['time']['zos'] = np.abs(obs - fct_t)

            hrmse[b_name]['weekly'] = {}
            hrmse[b_name]['weekly']['zos'] = np.abs(obs - fct_w)

            hrmse[b_name]['monthly'] = {}
            hrmse[b_name]['monthly']['zos'] = np.abs(obs - fct_m)

            hbias[b_name] = {}
            hbias[b_name]['time'] = {}
            hbias[b_name]['time']['zos'] = obs - fct_t

            hbias[b_name]['weekly'] = {}
            hbias[b_name]['weekly']['zos'] = obs - fct_w

            hbias[b_name]['monthly'] = {}
            hbias[b_name]['monthly']['zos'] = obs - fct_m

            # hacc[b_name] = {}
            # hacc[b_name]['time'] = {}
            # hacc[b_name]['time']['zos'] = (obs-clm)*(fct-clm)/
            #
            # gacc[b_name]['weekly'] = {}
            # gacc[b_name]['weekly']['zos'] = cacc(obs, fct_w, clm)
            #
            # gacc[b_name]['monthly'] = {}
            # gacc[b_name]['monthly']['zos'] = cacc(obs, fct_m, clm)

            hpss[b_name] = {}
            hpss[b_name]['time'] = {}
            # hpss[b_name]['time']['zos'] = cpss(obs, fct_t, hys)
            #1 应该会广播吧
            hpss[b_name]['time']['zos'] = 1 - np.abs(obs-fct_t)/np.abs(obs-hys)

            hpss[b_name]['weekly'] = {}
            hpss[b_name]['weekly']['zos'] = 1 - np.abs(obs-fct_w)/np.abs(obs-hys)

            hpss[b_name]['monthly'] = {}
            hpss[b_name]['monthly']['zos'] = 1 - np.abs(obs-fct_m)/np.abs(obs-hys)

            hcss[b_name] = {}
            hcss[b_name]['time'] = {}
            hcss[b_name]['time']['zos'] = 1 - np.abs(obs-fct_t)/np.abs(obs-clm)

            hcss[b_name]['weekly'] = {}
            hcss[b_name]['weekly']['zos'] = 1 - np.abs(obs-fct_w)/np.abs(obs-clm)

            hcss[b_name]['monthly'] = {}
            hcss[b_name]['monthly']['zos'] = 1 - np.abs(obs-fct_m)/np.abs(obs-clm)

            hmetrics = np.zeros((3, 1, 5, grid.shape[0]))
            count_hmetrics = np.zeros((3, 1, 5, grid.shape[0]))
        # elif b_name == 'PSY4':
        #     pass
        else:
            hys = hys['zos']
            clm = clm['zos']
            fct = baseline_data_interp[b_name]['zos']
            obs = zos_data_filtered['zos']
            hrmse[b_name] = {}
            hrmse[b_name]['zos'] = np.abs(obs - fct)
            hbias[b_name] = {}
            hbias[b_name]['zos'] = obs - fct
            # hacc[b_name] = {}
            # hacc[b_name]['zos'] = cacc(obs, fct, clm)
            hpss[b_name] = {}
            hpss[b_name]['zos'] = 1 - np.abs(obs-fct)/np.abs(obs-hys)
            hcss[b_name] = {}
            hcss[b_name]['zos'] = 1 - np.abs(obs-fct)/np.abs(obs-clm)

            hmetrics = np.zeros((1, 5, grid.shape[0]))
            count_hmetrics = np.zeros((1, 5, grid.shape[0]))
    if builtins.list(hrmse.keys())[0] == 'climatology':
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:,0]-iloc[0])**2 + (grid[:,1]-iloc[1])**2).argmin()
            type = ['time', 'weekly', 'monthly']
            for m, t in enumerate(type):
                hmetrics[m, 0, 0, index] += np.nansum(hrmse['climatology'][t]['zos'][i])
                # hmetrics[m, 0, 1, index] += np.nansum(hacc['climatology'][t]['zos'][i])
                hmetrics[m, 0, 2, index] += np.nansum(hbias['climatology'][t]['zos'][i])
                hmetrics[m, 0, 3, index] += np.nansum(hpss['climatology'][t]['zos'][i])
                hmetrics[m, 0, 4, index] += np.nansum(hcss['climatology'][t]['zos'][i])
                count_hmetrics[m, 0, 0, index] += np.isfinite(hrmse['climatology'][t]['zos'][i])
                # count_hmetrics[m, 0, 1, index] += np.isfinite(hacc['climatology'][t]['zos'][i])
                count_hmetrics[m, 0, 2, index] += np.isfinite(hbias['climatology'][t]['zos'][i])
                count_hmetrics[m, 0, 3, index] += np.isfinite(hpss['climatology'][t]['zos'][i])
                count_hmetrics[m, 0, 4, index] += np.isfinite(hcss['climatology'][t]['zos'][i])
        hmetrics = np.where(count_hmetrics>0, hmetrics / count_hmetrics, np.nan)
    else:
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:, 0] - iloc[0]) ** 2 + (grid[:, 1] - iloc[1]) ** 2).argmin()
            hmetrics[0, 0, index] += np.nansum(builtins.list(hrmse.values())[0]['zos'][i])
            # hmetrics[0, 1, index] += np.nansum(builtins.list(hacc.values())[0]['zos'][i])
            hmetrics[0, 2, index] += np.nansum(builtins.list(hbias.values())[0]['zos'][i])
            hmetrics[0, 3, index] += np.nansum(builtins.list(hpss.values())[0]['zos'][i])
            hmetrics[0, 4, index] += np.nansum(builtins.list(hcss.values())[0]['zos'][i])
            count_hmetrics[0, 0, index] += np.isfinite(builtins.list(hrmse.values())[0]['zos'][i])
            # count_hmetrics[0, 1, index] += np.isfinite(builtins.list(hacc.values())[0]['zos'][i])
            count_hmetrics[0, 2, index] += np.isfinite(builtins.list(hbias.values())[0]['zos'][i])
            count_hmetrics[0, 3, index] += np.isfinite(builtins.list(hpss.values())[0]['zos'][i])
            count_hmetrics[0, 4, index] += np.isfinite(builtins.list(hcss.values())[0]['zos'][i])
        hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    return hmetrics
def cal_horizontal_metrics_GDP(gdp_data_filtered, baseline_data_interp, hys, clm, t_grid):
    hrmse = {};
    hbias = {};
    hacc = {};
    hpss = {};
    hcss = {}
    lat = np.arange(-80 + 5.625 / 2, 90, 5.625)
    lon = np.arange(-180, 180, 5.625)
    grid = np.meshgrid(lon, lat)
    grid = np.column_stack((grid[1].ravel(), grid[0].ravel()))
    for b_name in baseline_data_interp:
        if b_name == 'climatology':
            hys_sst = hys['sst']
            clm_sst = clm['sst']
            fct_t = baseline_data_interp['climatology']['time']['sst']
            fct_w = baseline_data_interp['climatology']['weekly']['sst']
            fct_m = baseline_data_interp['climatology']['monthly']['sst']
            obs = gdp_data_filtered['sst']

            hrmse[b_name] = {}
            hrmse[b_name]['time'] = {}
            hrmse[b_name]['time']['sst'] = np.abs(obs - fct_t)

            hrmse[b_name]['weekly'] = {}
            hrmse[b_name]['weekly']['sst'] = np.abs(obs - fct_w)

            hrmse[b_name]['monthly'] = {}
            hrmse[b_name]['monthly']['sst'] = np.abs(obs - fct_m)

            hbias[b_name] = {}
            hbias[b_name]['time'] = {}
            hbias[b_name]['time']['sst'] = obs - fct_t

            hbias[b_name]['weekly'] = {}
            hbias[b_name]['weekly']['sst'] = obs - fct_w

            hbias[b_name]['monthly'] = {}
            hbias[b_name]['monthly']['sst'] = obs - fct_m

            hpss[b_name] = {}
            hpss[b_name]['time'] = {}
            # hpss[b_name]['time']['sst'] = cpss(obs, fct_t, hys_sst)
            # 1 应该会广播吧
            hpss[b_name]['time']['sst'] = 1 - np.abs(obs - fct_t) / np.abs(obs - hys_sst)

            hpss[b_name]['weekly'] = {}
            hpss[b_name]['weekly']['sst'] = 1 - np.abs(obs - fct_w) / np.abs(obs - hys_sst)

            hpss[b_name]['monthly'] = {}
            hpss[b_name]['monthly']['sst'] = 1 - np.abs(obs - fct_m) / np.abs(obs - hys_sst)

            hcss[b_name] = {}
            hcss[b_name]['time'] = {}
            hcss[b_name]['time']['sst'] = 1 - np.abs(obs - fct_t) / np.abs(obs - clm_sst)

            hcss[b_name]['weekly'] = {}
            hcss[b_name]['weekly']['sst'] = 1 - np.abs(obs - fct_w) / np.abs(obs - clm_sst)

            hcss[b_name]['monthly'] = {}
            hcss[b_name]['monthly']['sst'] = 1 - np.abs(obs - fct_m) / np.abs(obs - clm_sst)

            hys_uo = hys['uo'][0,:]
            clm_uo = clm['uo'][0,:]
            fct_t = baseline_data_interp['climatology']['time']['uo'][0,:]
            fct_w = baseline_data_interp['climatology']['weekly']['uo'][0,:]
            fct_m = baseline_data_interp['climatology']['monthly']['uo'][0,:]
            obs = gdp_data_filtered['uo']
            hrmse[b_name]['time']['uo'] = np.abs(obs - fct_t)
            hrmse[b_name]['weekly']['uo'] = np.abs(obs - fct_w)
            hrmse[b_name]['monthly']['uo'] = np.abs(obs - fct_m)
            hbias[b_name]['time']['uo'] = obs - fct_t
            hbias[b_name]['weekly']['uo'] = obs - fct_w
            hbias[b_name]['monthly']['uo'] = obs - fct_m
            hpss[b_name]['time']['uo'] = 1 - np.abs(obs - fct_t) / np.abs(obs - hys_uo)
            hpss[b_name]['weekly']['uo'] = 1 - np.abs(obs - fct_w) / np.abs(obs - hys_uo)
            hpss[b_name]['monthly']['uo'] = 1 - np.abs(obs - fct_m) / np.abs(obs - hys_uo)
            hcss[b_name]['time']['uo'] = 1 - np.abs(obs - fct_t) / np.abs(obs - clm_uo)
            hcss[b_name]['weekly']['uo'] = 1 - np.abs(obs - fct_w) / np.abs(obs - clm_uo)
            hcss[b_name]['monthly']['uo'] = 1 - np.abs(obs - fct_m) / np.abs(obs - clm_uo)

            hys_vo = hys['vo'][0,:]
            clm_vo = clm['vo'][0,:]
            fct_t = baseline_data_interp['climatology']['time']['vo'][0,:]
            fct_w = baseline_data_interp['climatology']['weekly']['vo'][0,:]
            fct_m = baseline_data_interp['climatology']['monthly']['vo'][0,:]
            obs = gdp_data_filtered['vo']
            hrmse[b_name]['time']['vo'] = np.abs(obs - fct_t)
            hrmse[b_name]['weekly']['vo'] = np.abs(obs - fct_w)
            hrmse[b_name]['monthly']['vo'] = np.abs(obs - fct_m)
            hbias[b_name]['time']['vo'] = obs - fct_t
            hbias[b_name]['weekly']['vo'] = obs - fct_w
            hbias[b_name]['monthly']['vo'] = obs - fct_m
            hpss[b_name]['time']['vo'] = 1 - np.abs(obs - fct_t) / np.abs(obs - hys_vo)
            hpss[b_name]['weekly']['vo'] = 1 - np.abs(obs - fct_w) / np.abs(obs - hys_vo)
            hpss[b_name]['monthly']['vo'] = 1 - np.abs(obs - fct_m) / np.abs(obs - hys_vo)
            hcss[b_name]['time']['vo'] = 1 - np.abs(obs - fct_t) / np.abs(obs - clm_vo)
            hcss[b_name]['weekly']['vo'] = 1 - np.abs(obs - fct_w) / np.abs(obs - clm_vo)
            hcss[b_name]['monthly']['vo'] = 1 - np.abs(obs - fct_m) / np.abs(obs - clm_vo)

            hmetrics = np.zeros((3, 3, 5, grid.shape[0]))
            count_hmetrics = np.zeros((3, 3, 5, grid.shape[0]))
        # elif b_name == 'PSY4':
        #     pass
        else:
            hys_sst = hys['sst']
            clm_sst = clm['sst']
            fct = baseline_data_interp[b_name]['sst']
            obs = gdp_data_filtered['sst']
            hrmse[b_name] = {}
            hrmse[b_name]['sst'] = np.abs(obs - fct)
            hbias[b_name] = {}
            hbias[b_name]['sst'] = obs - fct
            # hacc[b_name] = {}
            # hacc[b_name]['sst'] = cacc(obs, fct, clm_sst)
            hpss[b_name] = {}
            hpss[b_name]['sst'] = 1 - np.abs(obs - fct) / np.abs(obs - hys_sst)
            hcss[b_name] = {}
            hcss[b_name]['sst'] = 1 - np.abs(obs - fct) / np.abs(obs - clm_sst)

            hys_uo = hys['uo'][0,:]
            clm_uo = clm['uo'][0,:]
            fct = baseline_data_interp[b_name]['uo'][0,:]
            obs = gdp_data_filtered['uo']
            hrmse[b_name]['uo'] = np.abs(obs - fct)
            hbias[b_name]['uo'] = obs - fct
            hpss[b_name]['uo'] = 1 - np.abs(obs - fct) / np.abs(obs - hys_uo)
            hcss[b_name]['uo'] = 1 - np.abs(obs - fct) / np.abs(obs - clm_uo)

            hys_vo = hys['vo'][0,:]
            clm_vo = clm['vo'][0,:]
            fct = baseline_data_interp[b_name]['vo'][0,:]
            obs = gdp_data_filtered['vo']
            hrmse[b_name]['vo'] = np.abs(obs - fct)
            hbias[b_name]['vo'] = obs - fct
            hpss[b_name]['vo'] = 1 - np.abs(obs - fct) / np.abs(obs - hys_vo)
            hcss[b_name]['vo'] = 1 - np.abs(obs - fct) / np.abs(obs - clm_vo)

            hmetrics = np.zeros((3, 5, grid.shape[0]))
            count_hmetrics = np.zeros((3, 5, grid.shape[0]))
    if builtins.list(hrmse.keys())[0] == 'climatology':
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:,0]-iloc[0])**2+(grid[:,1]-iloc[1])**2).argmin()

            type = ['time', 'weekly', 'monthly']
            var_type = ['sst']
            for m, t in enumerate(type):
                for n, vt in enumerate(var_type):
                    hmetrics[m, n, 0, index] += np.nansum(hrmse['climatology'][t][vt][i])
                    # hmetrics[m, n, 1, index] += np.nansum(hacc['climatology'][t][vt][i])
                    hmetrics[m, n, 2, index] += np.nansum(hbias['climatology'][t][vt][i])
                    hmetrics[m, n, 3, index] += np.nansum(hpss['climatology'][t][vt][i])
                    hmetrics[m, n, 4, index] += np.nansum(hcss['climatology'][t][vt][i])
                    count_hmetrics[m, n, 0, index] += np.isfinite(hrmse['climatology'][t][vt][i])
                    # count_hmetrics[m, n, 1, index] += np.isfinite(hacc['climatology'][t][vt][i])
                    count_hmetrics[m, n, 2, index] += np.isfinite(hbias['climatology'][t][vt][i])
                    count_hmetrics[m, n, 3, index] += np.isfinite(hpss['climatology'][t][vt][i])
                    count_hmetrics[m, n, 4, index] += np.isfinite(hcss['climatology'][t][vt][i])
        # hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    else:
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:, 0] - iloc[0]) ** 2 + (grid[:, 1] - iloc[1]) ** 2).argmin()
            var_type = ['sst']
            for n, vt in enumerate(var_type):
                
                hmetrics[n, 0, index] += np.nansum(builtins.list(hrmse.values())[0][vt][i])
                # hmetrics[n, 1, index] += np.nansum(builtins.list(hacc.values())[0][vt][i])
                hmetrics[n, 2, index] += np.nansum(builtins.list(hbias.values())[0][vt][i])
                hmetrics[n, 3, index] += np.nansum(builtins.list(hpss.values())[0][vt][i])
                hmetrics[n, 4, index] += np.nansum(builtins.list(hcss.values())[0][vt][i])
                count_hmetrics[n, 0, index] += np.isfinite(builtins.list(hrmse.values())[0][vt][i])
                # count_hmetrics[n, 1, index] += np.isfinite(builtins.list(hacc.values())[0][vt][i])
                count_hmetrics[n, 2, index] += np.isfinite(builtins.list(hbias.values())[0][vt][i])
                count_hmetrics[n, 3, index] += np.isfinite(builtins.list(hpss.values())[0][vt][i])
                
                count_hmetrics[n, 4, index] += np.isfinite(builtins.list(hcss.values())[0][vt][i])
        # hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    if builtins.list(hrmse.keys())[0] == 'climatology':
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:,0]-iloc[0])**2+(grid[:,1]-iloc[1])**2).argmin()

            type = ['time', 'weekly', 'monthly']
            var_type = ['uo', 'vo']
            for m, t in enumerate(type):
                for n, vt in enumerate(var_type):
                    hmetrics[m, n+1, 0, index] += np.nansum(hrmse['climatology'][t][vt][i])
                    # hmetrics[m, n, 1, index] += np.nansum(hacc['climatology'][t][vt][i])
                    hmetrics[m, n+1, 2, index] += np.nansum(hbias['climatology'][t][vt][i])
                    hmetrics[m, n+1, 3, index] += np.nansum(hpss['climatology'][t][vt][i])
                    hmetrics[m, n+1, 4, index] += np.nansum(hcss['climatology'][t][vt][i])
                    count_hmetrics[m, n+1, 0, index] += np.isfinite(hrmse['climatology'][t][vt][i])
                    # count_hmetrics[m, n, 1, index] += np.isfinite(hacc['climatology'][t][vt][i])
                    count_hmetrics[m, n+1, 2, index] += np.isfinite(hbias['climatology'][t][vt][i])
                    count_hmetrics[m, n+1, 3, index] += np.isfinite(hpss['climatology'][t][vt][i])
                    count_hmetrics[m, n+1, 4, index] += np.isfinite(hcss['climatology'][t][vt][i])
        hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    else:
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:, 0] - iloc[0]) ** 2 + (grid[:, 1] - iloc[1]) ** 2).argmin()
            var_type = ['uo', 'vo']
            for n, vt in enumerate(var_type):
                hmetrics[n+1, 0, index] += np.nansum(builtins.list(hrmse.values())[0][vt][i])
                # hmetrics[n, 1, index] += np.nansum(builtins.list(hacc.values())[0][vt][i])
                hmetrics[n+1, 2, index] += np.nansum(builtins.list(hbias.values())[0][vt][i])
                hmetrics[n+1, 3, index] += np.nansum(builtins.list(hpss.values())[0][vt][i])
                hmetrics[n+1, 4, index] += np.nansum(builtins.list(hcss.values())[0][vt][i])
                count_hmetrics[n+1, 0, index] += np.isfinite(builtins.list(hrmse.values())[0][vt][i])
                # count_hmetrics[n, 1, index] += np.isfinite(builtins.list(hacc.values())[0][vt][i])
                count_hmetrics[n+1, 2, index] += np.isfinite(builtins.list(hbias.values())[0][vt][i])
                count_hmetrics[n+1, 3, index] += np.isfinite(builtins.list(hpss.values())[0][vt][i])
                count_hmetrics[n+1, 4, index] += np.isfinite(builtins.list(hcss.values())[0][vt][i])
        hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    return hmetrics
def cal_horizontal_metrics_TSP(en4_data_filtered, baseline_data_interp, hys, clm, t_grid):
    hrmse = {};
    hbias = {};
    hacc = {};
    hpss = {};
    hcss = {}
    lat = np.arange(-80 + 5.625/2, 90, 5.625)
    lon = np.arange(-180, 180, 5.625)
    grid = np.meshgrid(lon, lat)
    grid = np.column_stack((grid[1].ravel(), grid[0].ravel()))

    # t_grid : [lat,
    #             lon]
    # hrmse :
    # nvar, nmetrics, ngrid

    for b_name in baseline_data_interp:
        if b_name == 'climatology':
            hys_potm = hys['potm']
            clm_potm = clm['potm']
            fct_t = baseline_data_interp['climatology']['time']['potm']
            fct_w = baseline_data_interp['climatology']['weekly']['potm']
            fct_m = baseline_data_interp['climatology']['monthly']['potm']
            obs = en4_data_filtered['potm'].T

            hrmse[b_name] = {}
            hrmse[b_name]['time'] = {}
            hrmse[b_name]['time']['potm'] = np.sqrt(np.nanmean((obs - fct_t)**2, axis=0))

            hrmse[b_name]['weekly'] = {}
            hrmse[b_name]['weekly']['potm'] = np.sqrt(np.nanmean((obs - fct_w)**2, axis=0))

            hrmse[b_name]['monthly'] = {}
            hrmse[b_name]['monthly']['potm'] = np.sqrt(np.nanmean((obs - fct_m)**2, axis=0))

            hacc[b_name] = {}
            hacc[b_name]['time'] = {}
            hacc[b_name]['time']['potm'] = choracc(obs, fct_t, clm_potm)

            hacc[b_name]['weekly'] = {}
            hacc[b_name]['weekly']['potm'] = choracc(obs, fct_w, clm_potm)

            hacc[b_name]['monthly'] = {}
            hacc[b_name]['monthly']['potm'] = choracc(obs, fct_m, clm_potm)

            hbias[b_name] = {}
            hbias[b_name]['time'] = {}
            hbias[b_name]['time']['potm'] = np.nanmean((obs - fct_t), axis=0)

            hbias[b_name]['weekly'] = {}
            hbias[b_name]['weekly']['potm'] = np.nanmean((obs - fct_w), axis=0)

            hbias[b_name]['monthly'] = {}
            hbias[b_name]['monthly']['potm'] = np.nanmean((obs - fct_m), axis=0)

            hpss[b_name] = {}
            hpss[b_name]['time'] = {}
            hpss[b_name]['time']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_t)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_potm)**2, axis=0))

            hpss[b_name]['weekly'] = {}
            hpss[b_name]['weekly']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_w)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_potm)**2, axis=0))

            hpss[b_name]['monthly'] = {}
            hpss[b_name]['monthly']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_m)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_potm)**2, axis=0))

            hcss[b_name] = {}
            hcss[b_name]['time'] = {}
            hcss[b_name]['time']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_t)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_potm)**2, axis=0))

            hcss[b_name]['weekly'] = {}
            hcss[b_name]['weekly']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_w)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_potm)**2, axis=0))

            hcss[b_name]['monthly'] = {}
            hcss[b_name]['monthly']['potm'] = 1 - np.sqrt(np.nanmean((obs - fct_m)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_potm)**2, axis=0))

            # 盐度
            hys_psal = hys['psal']
            clm_psal = clm['psal']
            fct_t = baseline_data_interp['climatology']['time']['psal']
            fct_w = baseline_data_interp['climatology']['weekly']['psal']
            fct_m = baseline_data_interp['climatology']['monthly']['psal']
            obs = en4_data_filtered['psal'].T
            hrmse[b_name]['time']['psal'] = np.sqrt(np.nanmean((obs - fct_t)**2, axis=0))
            hrmse[b_name]['weekly']['psal'] = np.sqrt(np.nanmean((obs - fct_w)**2, axis=0))
            hrmse[b_name]['monthly']['psal'] = np.sqrt(np.nanmean((obs - fct_m)**2, axis=0))

            hacc[b_name]['time']['psal'] = choracc(obs, fct_t, clm_psal)
            hacc[b_name]['weekly']['psal'] = choracc(obs, fct_w, clm_psal)
            hacc[b_name]['monthly']['psal'] = choracc(obs, fct_m, clm_psal)

            hbias[b_name]['time']['psal'] = np.nanmean((obs - fct_t), axis=0)
            hbias[b_name]['weekly']['psal'] = np.nanmean((obs - fct_w), axis=0)
            hbias[b_name]['monthly']['psal'] = np.nanmean((obs - fct_m), axis=0)

            hpss[b_name]['time']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_t)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_psal)**2, axis=0))
            hpss[b_name]['weekly']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_w)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_psal)**2, axis=0))
            hpss[b_name]['monthly']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_m)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_psal)**2, axis=0))

            hcss[b_name]['time']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_t)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_psal)**2, axis=0))
            hcss[b_name]['weekly']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_w)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_psal)**2, axis=0))
            hcss[b_name]['monthly']['psal'] = 1 - np.sqrt(np.nanmean((obs - fct_m)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_psal)**2, axis=0))

            hmetrics = np.zeros((3, 2, 5, grid.shape[0]))
            count_hmetrics = np.zeros((3, 2, 5, grid.shape[0]))
        # elif b_name == 'PSY4':
        #     pass
        else:
            hys_potm = hys['potm']
            clm_potm = clm['potm']
            fct = baseline_data_interp[b_name]['potm']
            obs = en4_data_filtered['potm'].T
            hrmse[b_name] = {}
            hrmse[b_name]['potm'] = np.sqrt(np.nanmean((obs - fct)**2, axis=0))
            hacc[b_name] = {}
            hacc[b_name]['potm'] = choracc(obs, fct, clm_potm)
            hbias[b_name] = {}
            hbias[b_name]['potm'] = np.nanmean((obs - fct), axis=0)
            hpss[b_name] = {}
            hpss[b_name]['potm'] = 1 - np.sqrt(np.nanmean((obs - fct)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_potm)**2, axis=0))
            hcss[b_name] = {}
            hcss[b_name]['potm'] = 1 - np.sqrt(np.nanmean((obs - fct)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_potm)**2, axis=0))

            hys_psal = hys['psal']
            clm_psal = clm['psal']
            fct = baseline_data_interp[b_name]['psal']
            obs = en4_data_filtered['psal'].T
            hrmse[b_name]['psal'] = np.sqrt(np.nanmean((obs - fct)**2, axis=0))
            hacc[b_name]['psal'] = choracc(obs, fct, clm_psal)
            hbias[b_name]['psal'] = np.nanmean((obs - fct), axis=0)
            hpss[b_name]['psal'] = 1 - np.sqrt(np.nanmean((obs - fct)**2, axis=0)) / np.sqrt(np.nanmean((obs - hys_psal)**2, axis=0))
            hcss[b_name]['psal'] = 1 - np.sqrt(np.nanmean((obs - fct)**2, axis=0)) / np.sqrt(np.nanmean((obs - clm_potm)**2, axis=0))

            hmetrics = np.zeros((2, 5, grid.shape[0]))
            count_hmetrics = np.zeros((2, 5, grid.shape[0]))
    # nvar, nmetrics, ngrid
    if builtins.list(hrmse.keys())[0] == 'climatology':
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:,0]-iloc[0])**2+(grid[:,1]-iloc[1])**2).argmin()

            type = ['time', 'weekly', 'monthly']
            var_type = ['potm', 'psal']
            for m, t in enumerate(type):
                for n, vt in enumerate(var_type):
                    hmetrics[m, n, 0, index] += np.nansum(hrmse['climatology'][t][vt][i])
                    hmetrics[m, n, 1, index] += np.nansum(hacc['climatology'][t][vt][i])
                    hmetrics[m, n, 2, index] += np.nansum(hbias['climatology'][t][vt][i])
                    hmetrics[m, n, 3, index] += np.nansum(hpss['climatology'][t][vt][i])
                    hmetrics[m, n, 4, index] += np.nansum(hcss['climatology'][t][vt][i])
                    count_hmetrics[m, n, 0, index] += np.isfinite(hrmse['climatology'][t][vt][i])
                    count_hmetrics[m, n, 1, index] += np.isfinite(hacc['climatology'][t][vt][i])
                    count_hmetrics[m, n, 2, index] += np.isfinite(hbias['climatology'][t][vt][i])
                    count_hmetrics[m, n, 3, index] += np.isfinite(hpss['climatology'][t][vt][i])
                    count_hmetrics[m, n, 4, index] += np.isfinite(hcss['climatology'][t][vt][i])
        hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    else:
        for i, iloc in enumerate(t_grid):
            index = np.sqrt((grid[:, 0] - iloc[0]) ** 2 + (grid[:, 1] - iloc[1]) ** 2).argmin()
            var_type = ['potm', 'psal']
            for n, vt in enumerate(var_type):
                hmetrics[n, 0, index] += np.nansum(builtins.list(hrmse.values())[0][vt][i])
                hmetrics[n, 1, index] += np.nansum(builtins.list(hacc.values())[0][vt][i])
                hmetrics[n, 2, index] += np.nansum(builtins.list(hbias.values())[0][vt][i])
                hmetrics[n, 3, index] += np.nansum(builtins.list(hpss.values())[0][vt][i])
                hmetrics[n, 4, index] += np.nansum(builtins.list(hcss.values())[0][vt][i])
                count_hmetrics[n, 0, index] += np.isfinite(builtins.list(hrmse.values())[0][vt][i])
                count_hmetrics[n, 1, index] += np.isfinite(builtins.list(hacc.values())[0][vt][i])
                count_hmetrics[n, 2, index] += np.isfinite(builtins.list(hbias.values())[0][vt][i])
                count_hmetrics[n, 3, index] += np.isfinite(builtins.list(hpss.values())[0][vt][i])
                count_hmetrics[n, 4, index] += np.isfinite(builtins.list(hcss.values())[0][vt][i])
        hmetrics = np.where(count_hmetrics > 0, hmetrics / count_hmetrics, np.nan)
    return hmetrics
def cal_subocean_metrics_ZOS(bounds, zos_data_filtered, baseline_data_interp, grid, hys, clm):
    srmse = {};
    sbias = {};
    sacc = {};
    spss = {};
    scss = {}

    s0 = ((grid[:, 0] >= bounds[0][0]) & (grid[:, 0] <= bounds[0][1]) & (grid[:, 1] >= bounds[0][2]) & (
                grid[:, 1] <= bounds[0][3]))
    s1 = ((grid[:, 0] >= bounds[1][0]) & (grid[:, 0] <= bounds[1][1]) & (grid[:, 1] >= bounds[1][2]) & (
                grid[:, 1] <= bounds[1][3]))
    s2 = ((grid[:, 0] >= bounds[2][0]) & (grid[:, 0] <= bounds[2][1]) & (grid[:, 1] >= bounds[2][2]) & (
                grid[:, 1] <= bounds[2][3]))
    s3 = ((grid[:, 0] >= bounds[3][0]) & (grid[:, 0] <= bounds[3][1]) & (grid[:, 1] >= bounds[3][2]) & (
                grid[:, 1] <= bounds[3][3]))
    s4 = ((grid[:, 0] >= bounds[4][0]) & (grid[:, 0] <= bounds[4][1]) & ((grid[:, 1] >= bounds[4][2]) | (
                grid[:, 1] <= bounds[4][3])))
    s5 = ((grid[:, 0] >= bounds[5][0]) & (grid[:, 0] <= bounds[5][1]) & ((grid[:, 1] >= bounds[5][2]) | (
                grid[:, 1] <= bounds[5][3])))
    s6 = ((grid[:, 0] >= bounds[6][0]) & (grid[:, 0] <= bounds[6][1]) & ((grid[:, 1] >= bounds[6][2]) | (
                grid[:, 1] <= bounds[6][3])))
    s7 = ((grid[:, 0] >= bounds[7][0]) & (grid[:, 0] <= bounds[7][1]) & (grid[:, 1] >= bounds[7][2]) & (
                grid[:, 1] <= bounds[7][3]))
    s8 = ((grid[:, 0] >= bounds[8][0]) & (grid[:, 0] <= bounds[8][1]) & (grid[:, 1] >= bounds[8][2]) & (
                grid[:, 1] <= bounds[8][3]))

    s_en = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    s_grid = [grid[s] for s in s_en]

    for s in s_en:
        for b_name in baseline_data_interp:
            if b_name == 'climatology':
                hys_s = hys['zos'][s]
                clm_s = clm['zos'][s]
                fct_t = baseline_data_interp['climatology']['time']['zos'][s]
                fct_w = baseline_data_interp['climatology']['weekly']['zos'][s]
                fct_m = baseline_data_interp['climatology']['monthly']['zos'][s]
                obs = zos_data_filtered['zos'][s]

                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('time',{})
                ## 00:00
                srmse[b_name]['time'].setdefault('zos', [])
                srmse[b_name]['time']['zos'].append(crmse(obs, fct_t))

                srmse[b_name].setdefault('weekly',{})
                srmse[b_name]['weekly'].setdefault('zos', [])
                srmse[b_name]['weekly']['zos'].append(crmse(obs, fct_w))

                srmse[b_name].setdefault('monthly',{})
                srmse[b_name]['monthly'].setdefault('zos', [])
                srmse[b_name]['monthly']['zos'].append(crmse(obs, fct_m))

                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('time',{})
                sbias[b_name]['time'].setdefault('zos', [])
                sbias[b_name]['time']['zos'].append(cbias(obs, fct_t))

                sbias[b_name].setdefault('weekly', {})
                sbias[b_name]['weekly'].setdefault('zos', [])
                sbias[b_name]['weekly']['zos'].append(cbias(obs, fct_w))

                sbias[b_name].setdefault('monthly',{})
                sbias[b_name]['monthly'].setdefault('zos', [])
                sbias[b_name]['monthly']['zos'].append(cbias(obs, fct_m))

                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('time',{})
                sacc[b_name]['time'].setdefault('zos', [])
                sacc[b_name]['time']['zos'].append(cacc(obs, fct_t, clm_s))

                sacc[b_name].setdefault('weekly',{})
                sacc[b_name]['weekly'].setdefault('zos', [])
                sacc[b_name]['weekly']['zos'].append(cacc(obs, fct_w, clm_s))

                sacc[b_name].setdefault('monthly',{})
                sacc[b_name]['monthly'].setdefault('zos', [])
                sacc[b_name]['monthly']['zos'].append(cacc(obs, fct_m, clm_s))

                spss.setdefault(b_name, {})
                spss[b_name].setdefault('time',{})
                spss[b_name]['time'].setdefault('zos', [])
                spss[b_name]['time']['zos'].append(cpss(obs, fct_t, hys_s))

                spss[b_name].setdefault('weekly',{})
                spss[b_name]['weekly'].setdefault('zos', [])
                spss[b_name]['weekly']['zos'].append(cpss(obs, fct_w, hys_s))

                spss[b_name].setdefault('monthly',{})
                spss[b_name]['monthly'].setdefault('zos', [])
                spss[b_name]['monthly']['zos'].append(cpss(obs, fct_m, hys_s))

                scss.setdefault(b_name, {})
                scss[b_name].setdefault('time',{})
                scss[b_name]['time'].setdefault('zos', [])
                scss[b_name]['time']['zos'].append(ccss(obs, fct_t, clm_s))

                scss[b_name].setdefault('weekly',{})
                scss[b_name]['weekly'].setdefault('zos', [])
                scss[b_name]['weekly']['zos'].append(ccss(obs, fct_w, clm_s))

                scss[b_name].setdefault('monthly',{})
                scss[b_name]['monthly'].setdefault('zos', [])
                scss[b_name]['monthly']['zos'].append(ccss(obs, fct_m, clm_s))

                smetrics = np.zeros((3, 1, 5, len(s_en)))
            # elif b_name == 'PSY4':
            #     pass
            else:
                hys_s = hys['zos'][s]
                clm_s = clm['zos'][s]
                fct = baseline_data_interp[b_name]['zos'][s]
                obs = zos_data_filtered['zos'][s]
                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('zos', [])
                srmse[b_name]['zos'].append(crmse(obs, fct))
                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('zos', [])
                sbias[b_name]['zos'].append(cbias(obs, fct))
                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('zos', [])
                sacc[b_name]['zos'].append(cacc(obs, fct, clm_s))
                spss.setdefault(b_name, {})
                spss[b_name].setdefault('zos', [])
                spss[b_name]['zos'].append(cpss(obs, fct, hys_s))
                scss.setdefault(b_name, {})
                scss[b_name].setdefault('zos', [])
                scss[b_name]['zos'].append(ccss(obs, fct, clm_s))

                smetrics = np.zeros((1, 5, len(s_en)))
    if builtins.list(srmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        for m, t in enumerate(type):
            smetrics[m, 0, 0, :] = np.array(srmse['climatology'][t]['zos'])
            smetrics[m, 0, 1, :] = np.array(sacc['climatology'][t]['zos'])
            smetrics[m, 0, 2, :] = np.array(sbias['climatology'][t]['zos'])
            smetrics[m, 0, 3, :] = np.array(spss['climatology'][t]['zos'])
            smetrics[m, 0, 4, :] = np.array(scss['climatology'][t]['zos'])
    else:
        smetrics[0, 0, :] = np.array(builtins.list(srmse.values())[0]['zos'])
        smetrics[0, 1, :] = np.array(builtins.list(sacc.values())[0]['zos'])
        smetrics[0, 2, :] = np.array(builtins.list(sbias.values())[0]['zos'])
        smetrics[0, 3, :] = np.array(builtins.list(spss.values())[0]['zos'])
        smetrics[0, 4, :] = np.array(builtins.list(scss.values())[0]['zos'])
    return smetrics
def cal_subocean_metrics_GDP(bounds, gdp_data_filtered, baseline_data_interp, grid, hys, clm):
    srmse = {};
    sbias = {};
    sacc = {};
    spss = {};
    scss = {}

    s0 = ((grid[:, 0] >= bounds[0][0]) & (grid[:, 0] <= bounds[0][1]) & (grid[:, 1] >= bounds[0][2]) & (
                grid[:, 1] <= bounds[0][3]))
    s1 = ((grid[:, 0] >= bounds[1][0]) & (grid[:, 0] <= bounds[1][1]) & (grid[:, 1] >= bounds[1][2]) & (
                grid[:, 1] <= bounds[1][3]))
    s2 = ((grid[:, 0] >= bounds[2][0]) & (grid[:, 0] <= bounds[2][1]) & (grid[:, 1] >= bounds[2][2]) & (
                grid[:, 1] <= bounds[2][3]))
    s3 = ((grid[:, 0] >= bounds[3][0]) & (grid[:, 0] <= bounds[3][1]) & (grid[:, 1] >= bounds[3][2]) & (
                grid[:, 1] <= bounds[3][3]))
    s4 = ((grid[:, 0] >= bounds[4][0]) & (grid[:, 0] <= bounds[4][1]) & ((grid[:, 1] >= bounds[4][2]) | (
                grid[:, 1] <= bounds[4][3])))
    s5 = ((grid[:, 0] >= bounds[5][0]) & (grid[:, 0] <= bounds[5][1]) & ((grid[:, 1] >= bounds[5][2]) | (
                grid[:, 1] <= bounds[5][3])))
    s6 = ((grid[:, 0] >= bounds[6][0]) & (grid[:, 0] <= bounds[6][1]) & ((grid[:, 1] >= bounds[6][2]) | (
                grid[:, 1] <= bounds[6][3])))
    s7 = ((grid[:, 0] >= bounds[7][0]) & (grid[:, 0] <= bounds[7][1]) & (grid[:, 1] >= bounds[7][2]) & (
                grid[:, 1] <= bounds[7][3]))
    s8 = ((grid[:, 0] >= bounds[8][0]) & (grid[:, 0] <= bounds[8][1]) & (grid[:, 1] >= bounds[8][2]) & (
                grid[:, 1] <= bounds[8][3]))

    s_en = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    s_grid = [grid[s] for s in s_en]

    s0 = ((grid[:, 0] >= bounds[0][0]) & (grid[:, 0] <= bounds[0][1]) & (grid[:, 1] >= bounds[0][2]) & (
            grid[:, 1] <= bounds[0][3]))
    s1 = ((grid[:, 0] >= bounds[1][0]) & (grid[:, 0] <= bounds[1][1]) & (grid[:, 1] >= bounds[1][2]) & (
            grid[:, 1] <= bounds[1][3]))
    s2 = ((grid[:, 0] >= bounds[2][0]) & (grid[:, 0] <= bounds[2][1]) & (grid[:, 1] >= bounds[2][2]) & (
            grid[:, 1] <= bounds[2][3]))
    s3 = ((grid[:, 0] >= bounds[3][0]) & (grid[:, 0] <= bounds[3][1]) & (grid[:, 1] >= bounds[3][2]) & (
            grid[:, 1] <= bounds[3][3]))
    s4 = ((grid[:, 0] >= bounds[4][0]) & (grid[:, 0] <= bounds[4][1]) & (grid[:, 1] >= bounds[4][2]) & (
            grid[:, 1] <= bounds[4][3]))
    s5 = ((grid[:, 0] >= bounds[5][0]) & (grid[:, 0] <= bounds[5][1]) & (grid[:, 1] >= bounds[5][2]) & (
            grid[:, 1] <= bounds[5][3]))
    s6 = ((grid[:, 0] >= bounds[6][0]) & (grid[:, 0] <= bounds[6][1]) & (grid[:, 1] >= bounds[6][2]) & (
            grid[:, 1] <= bounds[6][3]))
    s7 = ((grid[:, 0] >= bounds[7][0]) & (grid[:, 0] <= bounds[7][1]) & (grid[:, 1] >= bounds[7][2]) & (
            grid[:, 1] <= bounds[7][3]))
    s8 = ((grid[:, 0] >= bounds[8][0]) & (grid[:, 0] <= bounds[8][1]) & (grid[:, 1] >= bounds[8][2]) & (
            grid[:, 1] <= bounds[8][3]))

    s_curen = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    s_curgrid = [grid[s] for s in s_curen]
    for s,s_cur in zip(s_en, s_curen):
        for b_name in baseline_data_interp:
            if b_name == 'climatology':
                hys_sst = hys['sst'][s]
                clm_sst = clm['sst'][s]
                fct_t = baseline_data_interp['climatology']['time']['sst'][s]
                fct_w = baseline_data_interp['climatology']['weekly']['sst'][s]
                fct_m = baseline_data_interp['climatology']['monthly']['sst'][s]
                obs = gdp_data_filtered['sst'][s]

                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('time',{})
                srmse[b_name]['time'].setdefault('sst', [])
                srmse[b_name]['time']['sst'].append(crmse(obs, fct_t))

                srmse[b_name].setdefault('weekly', {})
                srmse[b_name]['weekly'].setdefault('sst', [])
                srmse[b_name]['weekly']['sst'].append(crmse(obs, fct_w))

                srmse[b_name].setdefault('monthly', {})
                srmse[b_name]['monthly'].setdefault('sst', [])
                srmse[b_name]['monthly']['sst'].append(crmse(obs, fct_m))

                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('time', {})
                sbias[b_name]['time'].setdefault('sst', [])
                sbias[b_name]['time']['sst'].append(cbias(obs, fct_t))

                sbias[b_name].setdefault('weekly', {})
                sbias[b_name]['weekly'].setdefault('sst', [])
                sbias[b_name]['weekly']['sst'].append(cbias(obs, fct_w))

                sbias[b_name].setdefault('monthly', {})
                sbias[b_name]['monthly'].setdefault('sst', [])
                sbias[b_name]['monthly']['sst'].append(cbias(obs, fct_m))

                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('time', {})
                sacc[b_name]['time'].setdefault('sst', [])
                sacc[b_name]['time']['sst'].append(cacc(obs, fct_t, clm_sst))

                sacc[b_name].setdefault('weekly', {})
                sacc[b_name]['weekly'].setdefault('sst', [])
                sacc[b_name]['weekly']['sst'].append(cacc(obs, fct_w, clm_sst))

                sacc[b_name].setdefault('monthly', {})
                sacc[b_name]['monthly'].setdefault('sst', [])
                sacc[b_name]['monthly']['sst'].append(cacc(obs, fct_m, clm_sst))

                spss.setdefault(b_name, {})
                spss[b_name].setdefault('time', {})
                spss[b_name]['time'].setdefault('sst', [])
                spss[b_name]['time']['sst'].append(cpss(obs, fct_t, hys_sst))

                spss[b_name].setdefault('weekly', {})
                spss[b_name]['weekly'].setdefault('sst', [])
                spss[b_name]['weekly']['sst'].append(cpss(obs, fct_w, hys_sst))

                spss[b_name].setdefault('monthly', {})
                spss[b_name]['monthly'].setdefault('sst', [])
                spss[b_name]['monthly']['sst'].append(cpss(obs, fct_m, hys_sst))

                scss.setdefault(b_name, {})
                scss[b_name].setdefault('time', {})
                scss[b_name]['time'].setdefault('sst', [])
                scss[b_name]['time']['sst'].append(ccss(obs, fct_t, clm_sst))

                scss[b_name].setdefault('weekly', {})
                scss[b_name]['weekly'].setdefault('sst', [])
                scss[b_name]['weekly']['sst'].append(ccss(obs, fct_w, clm_sst))

                scss[b_name].setdefault('monthly', {})
                scss[b_name]['monthly'].setdefault('sst', [])
                scss[b_name]['monthly']['sst'].append(ccss(obs, fct_m, clm_sst))


                hys_uo = hys['uo'][0,s_cur]
                clm_uo = clm['uo'][0,s_cur]
                fct_t = baseline_data_interp['climatology']['time']['uo'][0,s_cur]
                fct_w = baseline_data_interp['climatology']['weekly']['uo'][0,s_cur]
                fct_m = baseline_data_interp['climatology']['monthly']['uo'][0,s_cur]
                obs = gdp_data_filtered['uo'][s_cur]

                srmse[b_name]['time'].setdefault('uo', [])
                srmse[b_name]['time']['uo'].append(crmse(obs, fct_t))

                srmse[b_name]['weekly'].setdefault('uo', [])
                srmse[b_name]['weekly']['uo'].append(crmse(obs, fct_w))

                srmse[b_name]['monthly'].setdefault('uo', [])
                srmse[b_name]['monthly']['uo'].append(crmse(obs, fct_m))

                sbias[b_name]['time'].setdefault('uo', [])
                sbias[b_name]['time']['uo'].append(cbias(obs, fct_t))

                sbias[b_name]['weekly'].setdefault('uo', [])
                sbias[b_name]['weekly']['uo'].append(cbias(obs, fct_w))

                sbias[b_name]['monthly'].setdefault('uo', [])
                sbias[b_name]['monthly']['uo'].append(cbias(obs, fct_m))

                sacc[b_name]['time'].setdefault('uo', [])
                sacc[b_name]['time']['uo'].append(cacc(obs, fct_t, clm_uo))

                sacc[b_name]['weekly'].setdefault('uo', [])
                sacc[b_name]['weekly']['uo'].append(cacc(obs, fct_w, clm_uo))

                sacc[b_name]['monthly'].setdefault('uo', [])
                sacc[b_name]['monthly']['uo'].append(cacc(obs, fct_m, clm_uo))

                spss[b_name]['time'].setdefault('uo', [])
                spss[b_name]['time']['uo'].append(cpss(obs, fct_t, hys_uo))

                spss[b_name]['weekly'].setdefault('uo', [])
                spss[b_name]['weekly']['uo'].append(cpss(obs, fct_w, hys_uo))

                spss[b_name]['monthly'].setdefault('uo', [])
                spss[b_name]['monthly']['uo'].append(cpss(obs, fct_m, hys_uo))

                scss[b_name]['time'].setdefault('uo', [])
                scss[b_name]['time']['uo'].append(ccss(obs, fct_t, clm_uo))

                scss[b_name]['weekly'].setdefault('uo', [])
                scss[b_name]['weekly']['uo'].append(ccss(obs, fct_w, clm_uo))

                scss[b_name]['monthly'].setdefault('uo', [])
                scss[b_name]['monthly']['uo'].append(ccss(obs, fct_m, clm_uo))


                hys_vo = hys['vo'][0,s_cur]
                clm_vo = clm['vo'][0,s_cur]
                fct_t = baseline_data_interp['climatology']['time']['vo'][0,s_cur]
                fct_w = baseline_data_interp['climatology']['weekly']['vo'][0,s_cur]
                fct_m = baseline_data_interp['climatology']['monthly']['vo'][0,s_cur]
                obs = gdp_data_filtered['vo'][s_cur]

                srmse[b_name]['time'].setdefault('vo', [])
                srmse[b_name]['time']['vo'].append(crmse(obs, fct_t))

                srmse[b_name]['weekly'].setdefault('vo', [])
                srmse[b_name]['weekly']['vo'].append(crmse(obs, fct_w))

                srmse[b_name]['monthly'].setdefault('vo', [])
                srmse[b_name]['monthly']['vo'].append(crmse(obs, fct_m))

                sbias[b_name]['time'].setdefault('vo', [])
                sbias[b_name]['time']['vo'].append(cbias(obs, fct_t))

                sbias[b_name]['weekly'].setdefault('vo', [])
                sbias[b_name]['weekly']['vo'].append(cbias(obs, fct_w))

                sbias[b_name]['monthly'].setdefault('vo', [])
                sbias[b_name]['monthly']['vo'].append(cbias(obs, fct_m))

                sacc[b_name]['time'].setdefault('vo', [])
                sacc[b_name]['time']['vo'].append(cacc(obs, fct_t, clm_vo))

                sacc[b_name]['weekly'].setdefault('vo', [])
                sacc[b_name]['weekly']['vo'].append(cacc(obs, fct_w, clm_vo))

                sacc[b_name]['monthly'].setdefault('vo', [])
                sacc[b_name]['monthly']['vo'].append(cacc(obs, fct_m, clm_vo))

                spss[b_name]['time'].setdefault('vo', [])
                spss[b_name]['time']['vo'].append(cpss(obs, fct_t, hys_vo))

                spss[b_name]['weekly'].setdefault('vo', [])
                spss[b_name]['weekly']['vo'].append(cpss(obs, fct_w, hys_vo))

                spss[b_name]['monthly'].setdefault('vo', [])
                spss[b_name]['monthly']['vo'].append(cpss(obs, fct_m, hys_vo))

                scss[b_name]['time'].setdefault('vo', [])
                scss[b_name]['time']['vo'].append(ccss(obs, fct_t, clm_vo))

                scss[b_name]['weekly'].setdefault('vo', [])
                scss[b_name]['weekly']['vo'].append(ccss(obs, fct_w, clm_vo))

                scss[b_name]['monthly'].setdefault('vo', [])
                scss[b_name]['monthly']['vo'].append(ccss(obs, fct_m, clm_vo))

                smetrics = np.zeros((3, 3, 5, len(s_en)))
            # elif b_name == 'PSY4':
            #     pass
            else:
                hys_sst = hys['sst'][s]
                clm_sst = clm['sst'][s]
                fct = baseline_data_interp[b_name]['sst'][s]
                obs = gdp_data_filtered['sst'][s]
                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('sst', [])
                srmse[b_name]['sst'].append(crmse(obs, fct))
                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('sst', [])
                sbias[b_name]['sst'].append(cbias(obs, fct))
                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('sst', [])
                sacc[b_name]['sst'].append(cacc(obs, fct, clm_sst))
                spss.setdefault(b_name, {})
                spss[b_name].setdefault('sst', [])
                spss[b_name]['sst'].append(cpss(obs, fct, hys_sst))
                scss.setdefault(b_name, {})
                scss[b_name].setdefault('sst', [])
                scss[b_name]['sst'].append(ccss(obs, fct, clm_sst))


                hys_uo = hys['uo'][0,s_cur]
                clm_uo = clm['uo'][0,s_cur]
                fct = baseline_data_interp[b_name]['uo'][0,s_cur]
                obs = gdp_data_filtered['uo'][s_cur]
                srmse[b_name].setdefault('uo', [])
                srmse[b_name]['uo'].append(crmse(obs, fct))
                sbias[b_name].setdefault('uo', [])
                sbias[b_name]['uo'].append(cbias(obs, fct))
                sacc[b_name].setdefault('uo', [])
                sacc[b_name]['uo'].append(cacc(obs, fct, clm_uo))
                spss[b_name].setdefault('uo', [])
                spss[b_name]['uo'].append(cpss(obs, fct, hys_uo))
                scss[b_name].setdefault('uo', [])
                scss[b_name]['uo'].append(ccss(obs, fct, clm_uo))


                hys_vo = hys['vo'][0,s_cur]
                clm_vo = clm['vo'][0,s_cur]
                fct = baseline_data_interp[b_name]['vo'][0,s_cur]
                obs = gdp_data_filtered['vo'][s_cur]
                srmse[b_name].setdefault('vo', [])
                srmse[b_name]['vo'].append(crmse(obs, fct))
                sbias[b_name].setdefault('vo', [])
                sbias[b_name]['vo'].append(cbias(obs, fct))
                sacc[b_name].setdefault('vo', [])
                sacc[b_name]['vo'].append(cacc(obs, fct, clm_vo))
                spss[b_name].setdefault('vo', [])
                spss[b_name]['vo'].append(cpss(obs, fct, hys_vo))
                scss[b_name].setdefault('vo', [])
                scss[b_name]['vo'].append(ccss(obs, fct, clm_vo))
                smetrics = np.zeros((3, 5, len(s_en)))
    if builtins.list(srmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        var_type = ['sst', 'uo', 'vo']
        for m, t in enumerate(type):
            for n, vt in enumerate(var_type):
                smetrics[m, n, 0, :] = np.array(srmse['climatology'][t][vt])
                smetrics[m, n, 1, :] = np.array(sacc['climatology'][t][vt])
                smetrics[m, n, 2, :] = np.array(sbias['climatology'][t][vt])
                smetrics[m, n, 3, :] = np.array(spss['climatology'][t][vt])
                smetrics[m, n, 4, :] = np.array(scss['climatology'][t][vt])

    else:
        var_type = ['sst', 'uo', 'vo']
        for n, vt in enumerate(var_type):
            smetrics[n, 0, :] = np.array(builtins.list(srmse.values())[0][vt])
            smetrics[n, 1, :] = np.array(builtins.list(sacc.values())[0][vt])
            smetrics[n, 2, :] = np.array(builtins.list(sbias.values())[0][vt])
            smetrics[n, 3, :] = np.array(builtins.list(spss.values())[0][vt])
            smetrics[n, 4, :] = np.array(builtins.list(scss.values())[0][vt])
    return smetrics
def cal_subocean_metrics_TSP(bounds, en4_data_filtered, baseline_data_interp, grid, hys, clm):
    srmse = {};
    sbias = {};
    sacc = {};
    spss = {};
    scss = {}

    s0 = ((grid[:, 0] >= bounds[0][0]) & (grid[:, 0] <= bounds[0][1]) & (grid[:, 1] >= bounds[0][2]) & (grid[:, 1] <= bounds[0][3]))
    s1 = ((grid[:, 0] >= bounds[1][0]) & (grid[:, 0] <= bounds[1][1]) & (grid[:, 1] >= bounds[1][2]) & (grid[:, 1] <= bounds[1][3]))
    s2 = ((grid[:, 0] >= bounds[2][0]) & (grid[:, 0] <= bounds[2][1]) & (grid[:, 1] >= bounds[2][2]) & (grid[:, 1] <= bounds[2][3]))
    s3 = ((grid[:, 0] >= bounds[3][0]) & (grid[:, 0] <= bounds[3][1]) & (grid[:, 1] >= bounds[3][2]) & (grid[:, 1] <= bounds[3][3]))
    s4 = ((grid[:, 0] >= bounds[4][0]) & (grid[:, 0] <= bounds[4][1]) & ((grid[:, 1] >= bounds[4][2]) | (grid[:, 1] <= bounds[4][3])))
    s5 = ((grid[:, 0] >= bounds[5][0]) & (grid[:, 0] <= bounds[5][1]) & ((grid[:, 1] >= bounds[5][2]) | (grid[:, 1] <= bounds[5][3])))
    s6 = ((grid[:, 0] >= bounds[6][0]) & (grid[:, 0] <= bounds[6][1]) & ((grid[:, 1] >= bounds[6][2]) | (grid[:, 1] <= bounds[6][3])))
    s7 = ((grid[:, 0] >= bounds[7][0]) & (grid[:, 0] <= bounds[7][1]) & (grid[:, 1] >= bounds[7][2]) & (grid[:, 1] <= bounds[7][3]))
    s8 = ((grid[:, 0] >= bounds[8][0]) & (grid[:, 0] <= bounds[8][1]) & (grid[:, 1] >= bounds[8][2]) & (grid[:, 1] <= bounds[8][3]))

    s_en = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    s_grid = [grid[s] for s in s_en]

    for s in s_en:
        for b_name in baseline_data_interp:
            if b_name == 'climatology':
                hys_potm = hys['potm'][:, s]
                clm_potm = clm['potm'][:, s]
                fct_t = baseline_data_interp['climatology']['time']['potm'][:, s]
                fct_w = baseline_data_interp['climatology']['weekly']['potm'][:, s]
                fct_m = baseline_data_interp['climatology']['monthly']['potm'][:, s]
                obs = en4_data_filtered['potm'][s, :].T

                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('time',{})
                srmse[b_name]['time'].setdefault('potm', [])
                srmse[b_name]['time']['potm'].append(crmse(obs, fct_t))

                srmse[b_name].setdefault('weekly', {})
                srmse[b_name]['weekly'].setdefault('potm', [])
                srmse[b_name]['weekly']['potm'].append(crmse(obs, fct_w))

                srmse[b_name].setdefault('monthly', {})
                srmse[b_name]['monthly'].setdefault('potm', [])
                srmse[b_name]['monthly']['potm'].append(crmse(obs, fct_m))

                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('time', {})
                sbias[b_name]['time'].setdefault('potm', [])
                sbias[b_name]['time']['potm'].append(cbias(obs, fct_t))

                sbias[b_name].setdefault('weekly', {})
                sbias[b_name]['weekly'].setdefault('potm', [])
                sbias[b_name]['weekly']['potm'].append(cbias(obs, fct_w))

                sbias[b_name].setdefault('monthly', {})
                sbias[b_name]['monthly'].setdefault('potm', [])
                sbias[b_name]['monthly']['potm'].append(cbias(obs, fct_m))

                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('time', {})
                sacc[b_name]['time'].setdefault('potm', [])
                sacc[b_name]['time']['potm'].append(cacc(obs, fct_t, clm_potm))

                sacc[b_name].setdefault('weekly', {})
                sacc[b_name]['weekly'].setdefault('potm', [])
                sacc[b_name]['weekly']['potm'].append(cacc(obs, fct_w, clm_potm))

                sacc[b_name].setdefault('monthly', {})
                sacc[b_name]['monthly'].setdefault('potm', [])
                sacc[b_name]['monthly']['potm'].append(cacc(obs, fct_m, clm_potm))

                spss.setdefault(b_name, {})
                spss[b_name].setdefault('time', {})
                spss[b_name]['time'].setdefault('potm', [])
                spss[b_name]['time']['potm'].append(cpss(obs, fct_t, hys_potm))

                spss[b_name].setdefault('weekly', {})
                spss[b_name]['weekly'].setdefault('potm', [])
                spss[b_name]['weekly']['potm'].append(cpss(obs, fct_w, hys_potm))

                spss[b_name].setdefault('monthly', {})
                spss[b_name]['monthly'].setdefault('potm', [])
                spss[b_name]['monthly']['potm'].append(cpss(obs, fct_m, hys_potm))

                scss.setdefault(b_name, {})
                scss[b_name].setdefault('time', {})
                scss[b_name]['time'].setdefault('potm', [])
                scss[b_name]['time']['potm'].append(ccss(obs, fct_t, clm_potm))

                scss[b_name].setdefault('weekly', {})
                scss[b_name]['weekly'].setdefault('potm', [])
                scss[b_name]['weekly']['potm'].append(ccss(obs, fct_w, clm_potm))

                scss[b_name].setdefault('monthly', {})
                scss[b_name]['monthly'].setdefault('potm', [])
                scss[b_name]['monthly']['potm'].append(ccss(obs, fct_m, clm_potm))


                hys_psal = hys['psal'][:, s]
                clm_psal = clm['psal'][:, s]
                fct_t = baseline_data_interp['climatology']['time']['psal'][:, s]
                fct_w = baseline_data_interp['climatology']['weekly']['psal'][:, s]
                fct_m = baseline_data_interp['climatology']['monthly']['psal'][:, s]
                obs = en4_data_filtered['psal'][s, :].T

                srmse[b_name]['time'].setdefault('psal', [])
                srmse[b_name]['time']['psal'].append(crmse(obs, fct_t))

                srmse[b_name]['weekly'].setdefault('psal', [])
                srmse[b_name]['weekly']['psal'].append(crmse(obs, fct_w))

                srmse[b_name]['monthly'].setdefault('psal', [])
                srmse[b_name]['monthly']['psal'].append(crmse(obs, fct_m))

                sbias[b_name]['time'].setdefault('psal', [])
                sbias[b_name]['time']['psal'].append(cbias(obs, fct_t))

                sbias[b_name]['weekly'].setdefault('psal', [])
                sbias[b_name]['weekly']['psal'].append(cbias(obs, fct_w))

                sbias[b_name]['monthly'].setdefault('psal', [])
                sbias[b_name]['monthly']['psal'].append(cbias(obs, fct_m))

                sacc[b_name]['time'].setdefault('psal', [])
                sacc[b_name]['time']['psal'].append(cacc(obs, fct_t, clm_psal))

                sacc[b_name]['weekly'].setdefault('psal', [])
                sacc[b_name]['weekly']['psal'].append(cacc(obs, fct_w, clm_psal))

                sacc[b_name]['monthly'].setdefault('psal', [])
                sacc[b_name]['monthly']['psal'].append(cacc(obs, fct_m, clm_psal))

                spss[b_name]['time'].setdefault('psal', [])
                spss[b_name]['time']['psal'].append(cpss(obs, fct_t, hys_psal))

                spss[b_name]['weekly'].setdefault('psal', [])
                spss[b_name]['weekly']['psal'].append(cpss(obs, fct_w, hys_psal))

                spss[b_name]['monthly'].setdefault('psal', [])
                spss[b_name]['monthly']['psal'].append(cpss(obs, fct_m, hys_psal))

                scss[b_name]['time'].setdefault('psal', [])
                scss[b_name]['time']['psal'].append(ccss(obs, fct_t, clm_psal))

                scss[b_name]['weekly'].setdefault('psal', [])
                scss[b_name]['weekly']['psal'].append(ccss(obs, fct_w, clm_psal))

                scss[b_name]['monthly'].setdefault('psal', [])
                scss[b_name]['monthly']['psal'].append(ccss(obs, fct_m, clm_psal))

                smetrics = np.zeros((3, 2, 5, len(s_en)))

            # elif b_name == 'PSY4':
            #     pass
            else:
                hys_potm = hys['potm'][:, s]
                clm_potm = clm['potm'][:, s]
                fct = baseline_data_interp[b_name]['potm'][:, s]
                obs = en4_data_filtered['potm'][s, :].T
                srmse.setdefault(b_name, {})
                srmse[b_name].setdefault('potm', [])
                srmse[b_name]['potm'].append(crmse(obs, fct))
                sbias.setdefault(b_name, {})
                sbias[b_name].setdefault('potm', [])
                sbias[b_name]['potm'].append(cbias(obs, fct))
                sacc.setdefault(b_name, {})
                sacc[b_name].setdefault('potm', [])
                sacc[b_name]['potm'].append(cacc(obs, fct, clm_potm))
                spss.setdefault(b_name, {})
                spss[b_name].setdefault('potm', [])
                spss[b_name]['potm'].append(cpss(obs, fct, hys_potm))
                scss.setdefault(b_name, {})
                scss[b_name].setdefault('potm', [])
                scss[b_name]['potm'].append(ccss(obs, fct, clm_potm))


                hys_psal = hys['psal'][:, s]
                clm_psal = clm['psal'][:, s]
                fct = baseline_data_interp[b_name]['psal'][:, s]
                obs = en4_data_filtered['psal'][s, :].T
                srmse[b_name].setdefault('psal', [])
                srmse[b_name]['psal'].append(crmse(obs, fct))
                sbias[b_name].setdefault('psal', [])
                sbias[b_name]['psal'].append(cbias(obs, fct))
                sacc[b_name].setdefault('psal', [])
                sacc[b_name]['psal'].append(cacc(obs, fct, clm_psal))
                spss[b_name].setdefault('psal', [])
                spss[b_name]['psal'].append(cpss(obs, fct, hys_psal))
                scss[b_name].setdefault('psal', [])
                scss[b_name]['psal'].append(ccss(obs, fct, clm_psal))
                smetrics = np.zeros((2, 5, len(s_en)))
    if builtins.list(srmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        var_type = ['potm', 'psal']
        for m, t in enumerate(type):
            for n, vt in enumerate(var_type):
                smetrics[m, n, 0, :] = np.array(srmse['climatology'][t][vt])
                smetrics[m, n, 1, :] = np.array(sacc['climatology'][t][vt])
                smetrics[m, n, 2, :] = np.array(sbias['climatology'][t][vt])
                smetrics[m, n, 3, :] = np.array(spss['climatology'][t][vt])
                smetrics[m, n, 4, :] = np.array(scss['climatology'][t][vt])
    else:
        var_type = ['potm', 'psal']
        for n, vt in enumerate(var_type):
            smetrics[n ,0, :] = np.array(builtins.list(srmse.values())[0][vt])
            smetrics[n, 1, :] = np.array(builtins.list(sacc.values())[0][vt])
            smetrics[n, 2, :] = np.array(builtins.list(sbias.values())[0][vt])
            smetrics[n, 3, :] = np.array(builtins.list(spss.values())[0][vt])
            smetrics[n, 4, :] = np.array(builtins.list(scss.values())[0][vt])


    return smetrics
def cal_vertical_metrics_TSP(t_depth, en4_data_filtered, baseline_data_interp, hys, clm, **level):

    vrmse = {}
    vbias = {}
    vacc = {}
    vpss = {}
    vcss = {}

    for i in level['levs']:
        blev = level['blev'][i]
        ulev = level['ulev'][i]

        id = ((t_depth >= blev) & (t_depth < ulev)).T
        for b_name in baseline_data_interp:
            if b_name == 'climatology':
                hys_potm = hys['potm'][id]
                clm_potm = clm['potm'][id]
                fct_t = baseline_data_interp['climatology']['time']['potm'][id]
                fct_w = baseline_data_interp['climatology']['weekly']['potm'][id]
                fct_m = baseline_data_interp['climatology']['monthly']['potm'][id]
                obs = en4_data_filtered['potm'].T[id]

                vrmse.setdefault(b_name, {})
                vrmse[b_name].setdefault('time', {})
                vrmse[b_name]['time'].setdefault('potm', [])
                vrmse[b_name]['time']['potm'].append(crmse(obs, fct_t))

                vrmse[b_name].setdefault('weekly', {})
                vrmse[b_name]['weekly'].setdefault('potm', [])
                vrmse[b_name]['weekly']['potm'].append(crmse(obs, fct_w))

                vrmse[b_name].setdefault('monthly', {})
                vrmse[b_name]['monthly'].setdefault('potm', [])
                vrmse[b_name]['monthly']['potm'].append(crmse(obs, fct_m))

                vbias.setdefault(b_name, {})
                vbias[b_name].setdefault('time', {})
                vbias[b_name]['time'].setdefault('potm', [])
                vbias[b_name]['time']['potm'].append(cbias(obs, fct_t))

                vbias[b_name].setdefault('weekly', {})
                vbias[b_name]['weekly'].setdefault('potm', [])
                vbias[b_name]['weekly']['potm'].append(cbias(obs, fct_w))

                vbias[b_name].setdefault('monthly', {})
                vbias[b_name]['monthly'].setdefault('potm', [])
                vbias[b_name]['monthly']['potm'].append(cbias(obs, fct_m))

                vacc.setdefault(b_name, {})
                vacc[b_name].setdefault('time', {})
                vacc[b_name]['time'].setdefault('potm', [])
                vacc[b_name]['time']['potm'].append(cacc(obs, fct_t, clm_potm))

                vacc[b_name].setdefault('weekly', {})
                vacc[b_name]['weekly'].setdefault('potm', [])
                vacc[b_name]['weekly']['potm'].append(cacc(obs, fct_w, clm_potm))

                vacc[b_name].setdefault('monthly', {})
                vacc[b_name]['monthly'].setdefault('potm', [])
                vacc[b_name]['monthly']['potm'].append(cacc(obs, fct_m, clm_potm))

                vpss.setdefault(b_name, {})
                vpss[b_name].setdefault('time', {})
                vpss[b_name]['time'].setdefault('potm', [])
                vpss[b_name]['time']['potm'].append(cpss(obs, fct_t, hys_potm))

                vpss[b_name].setdefault('weekly', {})
                vpss[b_name]['weekly'].setdefault('potm', [])
                vpss[b_name]['weekly']['potm'].append(cpss(obs, fct_w, hys_potm))

                vpss[b_name].setdefault('monthly', {})
                vpss[b_name]['monthly'].setdefault('potm', [])
                vpss[b_name]['monthly']['potm'].append(cpss(obs, fct_m, hys_potm))

                vcss.setdefault(b_name, {})
                vcss[b_name].setdefault('time', {})
                vcss[b_name]['time'].setdefault('potm', [])
                vcss[b_name]['time']['potm'].append(ccss(obs, fct_t, clm_potm))

                vcss[b_name].setdefault('weekly', {})
                vcss[b_name]['weekly'].setdefault('potm', [])
                vcss[b_name]['weekly']['potm'].append(ccss(obs, fct_w, clm_potm))

                vcss[b_name].setdefault('monthly', {})
                vcss[b_name]['monthly'].setdefault('potm', [])
                vcss[b_name]['monthly']['potm'].append(ccss(obs, fct_m, clm_potm))

                # 盐度
                hys_psal = hys['psal'][id]
                clm_psal = clm['psal'][id]
                fct_t = baseline_data_interp['climatology']['time']['psal'][id]
                fct_w = baseline_data_interp['climatology']['weekly']['psal'][id]
                fct_m = baseline_data_interp['climatology']['monthly']['psal'][id]
                obs = en4_data_filtered['psal'].T[id]

                vrmse[b_name]['time'].setdefault('psal', [])
                vrmse[b_name]['weekly'].setdefault('psal', [])
                vrmse[b_name]['monthly'].setdefault('psal', [])
                vrmse[b_name]['time']['psal'].append(crmse(obs, fct_t))
                vrmse[b_name]['weekly']['psal'].append(crmse(obs, fct_w))
                vrmse[b_name]['monthly']['psal'].append(crmse(obs, fct_m))

                vbias[b_name]['time'].setdefault('psal', [])
                vbias[b_name]['weekly'].setdefault('psal', [])
                vbias[b_name]['monthly'].setdefault('psal', [])
                vbias[b_name]['time']['psal'].append(cbias(obs, fct_t))
                vbias[b_name]['weekly']['psal'].append(cbias(obs, fct_w))
                vbias[b_name]['monthly']['psal'].append(cbias(obs, fct_m))

                vacc[b_name]['time'].setdefault('psal', [])
                vacc[b_name]['weekly'].setdefault('psal', [])
                vacc[b_name]['monthly'].setdefault('psal', [])
                vacc[b_name]['time']['psal'].append(cacc(obs, fct_t, clm_psal))
                vacc[b_name]['weekly']['psal'].append(cacc(obs, fct_w, clm_psal))
                vacc[b_name]['monthly']['psal'].append(cacc(obs, fct_m, clm_psal))

                vpss[b_name]['time'].setdefault('psal', [])
                vpss[b_name]['weekly'].setdefault('psal', [])
                vpss[b_name]['monthly'].setdefault('psal', [])
                vpss[b_name]['time']['psal'].append(cpss(obs, fct_t, hys_psal))
                vpss[b_name]['weekly']['psal'].append(cpss(obs, fct_w, hys_psal))
                vpss[b_name]['monthly']['psal'].append(cpss(obs, fct_m, hys_psal))

                vcss[b_name]['time'].setdefault('psal', [])
                vcss[b_name]['weekly'].setdefault('psal', [])
                vcss[b_name]['monthly'].setdefault('psal', [])
                vcss[b_name]['time']['psal'].append(ccss(obs, fct_t, clm_psal))
                vcss[b_name]['weekly']['psal'].append(ccss(obs, fct_w, clm_psal))
                vcss[b_name]['monthly']['psal'].append(ccss(obs, fct_m, clm_psal))

                vmetrics = np.zeros((3, 2, 5, len(level['levs'])))
            # elif b_name == 'PSY4':
            #     pass
            else:
                hys_potm = hys['potm'][id]
                clm_potm = clm['potm'][id]
                fct = baseline_data_interp[b_name]['potm'][id]
                obs = en4_data_filtered['potm'].T[id]
                vrmse.setdefault(b_name, {})
                vrmse[b_name].setdefault('potm', [])
                vrmse[b_name]['potm'].append(crmse(obs, fct))
                vbias.setdefault(b_name, {})
                vbias[b_name].setdefault('potm', [])
                vbias[b_name]['potm'].append(cbias(obs, fct))
                vacc.setdefault(b_name, {})
                vacc[b_name].setdefault('potm', [])
                vacc[b_name]['potm'].append(cacc(obs, fct, clm_potm))
                vpss.setdefault(b_name, {})
                vpss[b_name].setdefault('potm', [])
                vpss[b_name]['potm'].append(cpss(obs, fct, hys_potm))
                vcss.setdefault(b_name, {})
                vcss[b_name].setdefault('potm', [])
                vcss[b_name]['potm'].append(ccss(obs, fct, clm_potm))

                hys_psal = hys['psal'][id]
                clm_psal = clm['psal'][id]
                fct = baseline_data_interp[b_name]['psal'][id]
                obs = en4_data_filtered['psal'].T[id]
                vrmse[b_name].setdefault('psal', [])
                vbias[b_name].setdefault('psal', [])
                vacc[b_name].setdefault('psal', [])
                vpss[b_name].setdefault('psal', [])
                vcss[b_name].setdefault('psal', [])

                vrmse[b_name]['psal'].append(crmse(obs, fct))
                vbias[b_name]['psal'].append(cbias(obs, fct))
                vacc[b_name]['psal'].append(cacc(obs, fct, clm_psal))
                vpss[b_name]['psal'].append(cpss(obs, fct, hys_psal))
                vcss[b_name]['psal'].append(ccss(obs, fct, clm_psal))

                vmetrics = np.zeros((2, 5, len(level['levs'])))
    if builtins.list(vrmse.keys())[0] == 'climatology':
        type = ['time', 'weekly', 'monthly']
        var_type = ['potm', 'psal']
        for m, t in enumerate(type):
            for n, vt in enumerate(var_type):
                vmetrics[m, n, 0, :] = np.array(vrmse['climatology'][t][vt])
                vmetrics[m, n, 1, :] = np.array(vacc['climatology'][t][vt])
                vmetrics[m, n, 2, :] = np.array(vbias['climatology'][t][vt])
                vmetrics[m, n, 3, :] = np.array(vpss['climatology'][t][vt])
                vmetrics[m, n, 4, :] = np.array(vcss['climatology'][t][vt])
    else:
        var_type = ['potm', 'psal']
        for n, vt in enumerate(var_type):
            vmetrics[n, 0] = np.array(builtins.list(vrmse.values())[0][vt])
            vmetrics[n, 1] = np.array(builtins.list(vacc.values())[0][vt])
            vmetrics[n, 2] = np.array(builtins.list(vbias.values())[0][vt])
            vmetrics[n, 3] = np.array(builtins.list(vpss.values())[0][vt])
            vmetrics[n, 4] = np.array(builtins.list(vcss.values())[0][vt])
    return vmetrics
