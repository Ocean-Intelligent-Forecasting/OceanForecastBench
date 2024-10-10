# Copyright (c) 2024 College of Meteorology and Oceanography, National University of Defense Technology. All rights reserved
# Code adapted from:
# https://github.com/beijixiong1/OceanForecastBench/

import ipdb
import numpy as np
import xarray as xr
from scipy.interpolate import griddata, CubicSpline,RegularGridInterpolator
import time
def cal():
    pass

#basline_data
def interp_TSP(en4_data, baseline_data, s_lat, s_lon, s_depth):

    baseline_data_interp = {}
    en4_data_filtered = {}
    position_aval = np.where(en4_data.POSITION_QC.values==b'1')[0]
    lat = en4_data.LATITUDE[position_aval]
    lon = en4_data.LONGITUDE[position_aval]
    depth = en4_data.DEPH_CORRECTED[position_aval]
    #temperature: en4_data_potm->xarray , depth_potm->numpy array
    en4_data_potm = en4_data.POTM_CORRECTED[position_aval].values
    dp_aval_potm = (en4_data.POTM_CORRECTED_QC[position_aval] == b'1').values
    en4_data_potm[~dp_aval_potm] = np.nan

    depth_potm = np.where(dp_aval_potm, depth, np.nan)

    #salinity: en4_data_sal->xarray , depth_sal->numpy array
    en4_data_sal = en4_data.PSAL_CORRECTED[position_aval].values
    dp_aval_sal = (en4_data.PSAL_CORRECTED_QC[position_aval] == b'1').values

    en4_data_sal[~dp_aval_sal] = np.nan


    depth_sal = np.where(dp_aval_sal, depth, np.nan)

    nprof = len(position_aval)

    s_grid = np.meshgrid(s_lon,s_lat)
    s_grid = np.column_stack((s_grid[1].ravel(), s_grid[0].ravel()))
    ndepth = len(s_depth)
    clm_depth = [0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,25,26,27,28,29,30,31,32]

    # ddeg_out = 1/4
    # s_lat_h = np.arange(-80+ddeg_out/2, 90, ddeg_out)
    # s_lon_h = np.arange(-180, 180, ddeg_out)
    # s_grid_h = np.meshgrid(s_lon_h, s_lat_h)
    # s_grid_h = np.column_stack((s_grid_h[1].ravel(), s_grid_h[0].ravel()))

    #potm(nprof, nlevel)  每一个位置代表 温度的值
    #lat(nprof,)lat[0]
    #lon(nprof,) lon[0]
    #depth(nprof, nlevel) 每一个位置，每一个位置的深度
    t_lat = lat.values
    t_lon = lon.values
    t_grid = np.column_stack((t_lat, t_lon))
    t_depth = depth.values
    nt_depth = t_depth.shape[1]
    nt_prof = t_depth.shape[0]


    en4_data_filtered['potm'] = en4_data_potm
    en4_data_filtered['psal'] = en4_data_sal

    for b_name in baseline_data:

        if b_name == 'climatology':
            for t_name in baseline_data[b_name]:
                if t_name == 'time':
                    tem_potm = np.full((ndepth, nprof), np.nan)
                    tem_psal = np.full((ndepth, nprof), np.nan)
                    interp_potm = np.full((nt_depth, nt_prof), np.nan)
                    interp_psal = np.full((nt_depth, nt_prof), np.nan)
                    start_time = time.time()
                    for idep in range(ndepth):
                        potm_pred = baseline_data[b_name]['time'].thetao.values[clm_depth[idep], :, :]
                        interp1 = RegularGridInterpolator((s_lat, s_lon), potm_pred,bounds_error=False)
                        tem_potm[idep, :] = interp1(t_grid)
                        psal_pred = baseline_data[b_name]['time'].so.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), psal_pred,bounds_error=False)
                        tem_psal[idep, :] = interp2(t_grid)
                    end_time = time.time()
                    print('tsp horizontal interp:', end_time-start_time, 's')
                    start_time = time.time()
                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_potm[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_potm[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_potm[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_potm[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_potm[iprof, indp]
                                y_new = cs(x_new)
                                interp_potm[indp, iprof] = y_new

                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_psal[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_sal[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_sal[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_psal[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_sal[iprof, indp]
                                y_new = cs(x_new)
                                interp_psal[indp, iprof] = y_new
                    end_time = time.time()
                    print('tsp vertical interp:', end_time - start_time, 's')
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                    baseline_data_interp[b_name]['time'] = {}
                    baseline_data_interp[b_name]['time']['potm'] = interp_potm.astype(np.float32)
                    baseline_data_interp[b_name]['time']['psal'] = interp_psal.astype(np.float32)
                elif t_name == 'weekly':
                    tem_potm = np.full((ndepth, nprof), np.nan)
                    tem_psal = np.full((ndepth, nprof), np.nan)
                    interp_potm = np.full((nt_depth, nt_prof), np.nan)
                    interp_psal = np.full((nt_depth, nt_prof), np.nan)
                    for idep in range(ndepth):
                        potm_pred = baseline_data[b_name]['weekly'].thetao.values[clm_depth[idep], :, :]
                        interp1 = RegularGridInterpolator((s_lat, s_lon), potm_pred,bounds_error=False)
                        tem_potm[idep, :] = interp1(t_grid)
                        psal_pred = baseline_data[b_name]['weekly'].so.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), psal_pred,bounds_error=False)
                        tem_psal[idep, :] = interp2(t_grid)
                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_potm[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_potm[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_potm[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_potm[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_potm[iprof, indp]
                                y_new = cs(x_new)
                                interp_potm[indp, iprof] = y_new
                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_psal[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_sal[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_sal[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_psal[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_sal[iprof, indp]
                                y_new = cs(x_new)
                                interp_psal[indp, iprof] = y_new
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                    baseline_data_interp[b_name]['weekly'] = {}
                    baseline_data_interp[b_name]['weekly']['potm'] = interp_potm.astype(np.float32)
                    baseline_data_interp[b_name]['weekly']['psal'] = interp_psal.astype(np.float32)
                elif t_name == 'monthly':
                    tem_potm = np.full((ndepth, nprof), np.nan)
                    tem_psal = np.full((ndepth, nprof), np.nan)
                    interp_potm = np.full((nt_depth, nt_prof), np.nan)
                    interp_psal = np.full((nt_depth, nt_prof), np.nan)
                    for idep in range(ndepth):
                        potm_pred = baseline_data[b_name]['monthly'].thetao.values[clm_depth[idep], :, :]
                        interp1 = RegularGridInterpolator((s_lat, s_lon), potm_pred,bounds_error=False)
                        tem_potm[idep, :] = interp1(t_grid)
                        psal_pred = baseline_data[b_name]['monthly'].so.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), psal_pred,bounds_error=False)
                        tem_psal[idep, :] = interp2(t_grid)
                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_potm[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_potm[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_potm[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_potm[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_potm[iprof, indp]
                                y_new = cs(x_new)
                                interp_potm[indp, iprof] = y_new
                    for iprof in range(nprof):
                        indz = np.where(~np.isnan(tem_psal[:, iprof]))[0]
                        if indz.size > 0:
                            up = (depth_sal[iprof, :] >= s_depth[indz[0]]).tolist()
                            bo = (depth_sal[iprof, :] <= s_depth[indz[-1]]).tolist()
                            indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                            if indp.size > 0:
                                x = s_depth[indz]
                                y = tem_psal[indz, iprof]
                                cs = CubicSpline(x, y)
                                x_new = depth_sal[iprof, indp]
                                y_new = cs(x_new)
                                interp_psal[indp, iprof] = y_new
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                    baseline_data_interp[b_name]['monthly'] = {}
                    baseline_data_interp[b_name]['monthly']['potm'] = interp_potm.astype(np.float32)
                    baseline_data_interp[b_name]['monthly']['psal'] = interp_psal.astype(np.float32)
        else:

            tem_potm = np.full((ndepth,nprof), np.nan)
            tem_psal = np.full((ndepth,nprof), np.nan)
            interp_potm = np.full((nt_depth, nt_prof), np.nan)
            interp_psal = np.full((nt_depth, nt_prof), np.nan)
            for idep in range(ndepth):
                potm_pred = baseline_data[b_name]['thetao'][0,idep,:,:]
                interp1 = RegularGridInterpolator((s_lat, s_lon), potm_pred,bounds_error=False)
                tem_potm[idep, :] = interp1(t_grid)

                psal_pred = baseline_data[b_name]['so'][0,idep,:,:]
                interp2 = RegularGridInterpolator((s_lat, s_lon), psal_pred,bounds_error=False)
                tem_psal[idep, :] = interp2(t_grid)

            for iprof in range(nprof):
                indz = np.where(~np.isnan(tem_potm[:,iprof]))[0]
                if indz.size > 0:
                    up = (depth_potm[iprof, :] >= s_depth[indz[0]]).tolist()
                    bo = (depth_potm[iprof, :] <= s_depth[indz[-1]]).tolist()
                    indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                    if indp.size>0:
                        x = s_depth[indz]
                        y = tem_potm[indz,iprof]
                        cs = CubicSpline(x,y)
                        x_new = depth_potm[iprof, indp]
                        y_new = cs(x_new)
                        interp_potm[indp,iprof] = y_new

            for iprof in range(nprof):
                indz = np.where(~np.isnan(tem_psal[:,iprof]))[0]
                if indz.size > 0:
                    up = (depth_sal[iprof, :] >= s_depth[indz[0]]).tolist()
                    bo = (depth_sal[iprof, :] <= s_depth[indz[-1]]).tolist()
                    indp = np.where(list(map(lambda x, y: x and y, up, bo)))[0]
                    if indp.size>0:
                        x = s_depth[indz]
                        y = tem_psal[indz,iprof]
                        cs = CubicSpline(x,y)
                        x_new = depth_sal[iprof, indp]
                        y_new = cs(x_new)
                        interp_psal[indp,iprof] = y_new

            baseline_data_interp[b_name] = {}
            baseline_data_interp[b_name]['potm'] = interp_potm.astype(np.float32)
            baseline_data_interp[b_name]['psal'] = interp_psal.astype(np.float32)

    # en4_data_filtered:dict:'potm'numpyarray 'psal'numpyarray     baseline_data_interp:dict:numpy array
    return en4_data_filtered, baseline_data_interp, t_grid, t_depth
def interp_GDP(gdp_data, baseline_data, s_lat, s_lon, s_depth):

    baseline_data_interp = {}
    gdp_data_filtered = {}

    position_aval = np.where(gdp_data.drogue_lost_date.values > gdp_data.time.values)[0]
    lat = gdp_data.latitude[position_aval]
    lon = gdp_data.longitude[position_aval]
    gdp_data_sst = gdp_data.sst[position_aval]

    s_grid = np.meshgrid(s_lon, s_lat)
    s_grid = np.column_stack((s_grid[1].ravel(), s_grid[0].ravel()))

    ndepth = len(s_depth)
    clm_depth = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    t_lat = lat.values
    t_lon = lon.values
    t_grid = np.column_stack((t_lat, t_lon))
    gdp_data_filtered['sst'] = gdp_data_sst.values
    for b_name in baseline_data:
        if b_name == 'climatology':
            sst_pred = baseline_data['climatology']['time'].analysed_sst.values-273.15
            interp1 = RegularGridInterpolator((s_lat, s_lon), sst_pred,bounds_error=False)
            interp_sst = interp1(t_grid)
            baseline_data_interp[b_name] = {}
            baseline_data_interp[b_name]['time'] = {}
            baseline_data_interp[b_name]['time']['sst'] = interp_sst.astype(np.float32)

            sst_pred = baseline_data['climatology']['weekly'].analysed_sst.values-273.15
            interp1 = RegularGridInterpolator((s_lat, s_lon), sst_pred,bounds_error=False)
            interp_sst = interp1(t_grid)
            baseline_data_interp[b_name]['weekly'] = {}
            baseline_data_interp[b_name]['weekly']['sst'] = interp_sst.astype(np.float32)

            sst_pred = baseline_data['climatology']['monthly'].analysed_sst.values-273.15
            interp1 = RegularGridInterpolator((s_lat, s_lon), sst_pred,bounds_error=False)
            interp_sst = interp1(t_grid)
            baseline_data_interp[b_name]['monthly'] = {}
            baseline_data_interp[b_name]['monthly']['sst'] = interp_sst.astype(np.float32)
        else:
            sst_pred = baseline_data[b_name]['sst'][0]
            interp1 = RegularGridInterpolator((s_lat, s_lon), sst_pred,bounds_error=False)
            interp_sst = interp1(t_grid)
            baseline_data_interp[b_name] = {}
            baseline_data_interp[b_name]['sst'] = interp_sst.astype(np.float32)

    #流速

    gdp_data_filtered['uo'] = gdp_data.ve[position_aval].values
    gdp_data_filtered['vo'] = gdp_data.vn[position_aval].values

    for b_name in baseline_data:
        if b_name == 'climatology':
            for t_name in baseline_data[b_name]:
                if t_name == 'time':
                    tem_uo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    tem_vo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    interp_uo = np.full((1, t_lat.shape[0]), np.nan)
                    interp_vo = np.full((1, t_lat.shape[0]), np.nan)
                    ###.........
                    start_time = time.time()
                    for idep in range(ndepth):
                        uo_pred = baseline_data[b_name]['time'].uo.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), uo_pred,bounds_error=False)
                        tem_uo[idep, :] = interp2(t_grid)
                        vo_pred = baseline_data[b_name]['time'].vo.values[clm_depth[idep], :, :]
                        interp3 = RegularGridInterpolator((s_lat, s_lon), vo_pred,bounds_error=False)
                        tem_vo[idep, :] = interp3(t_grid)
                    for i in range(t_lat.shape[0]):
                        x = s_depth
                        y_uo = tem_uo[:, i]
                        mask = np.where(~np.isnan(y_uo))[0]
                        if mask.size == 0:
                            continue
                        if x[mask[0]] <= 15 and x[mask[-1]]>=15:
                            cs_uo = CubicSpline(x[mask], y_uo[mask])
                            x_new = 15
                            interp_uo[0, i] = cs_uo(x_new)
                        else:
                            interp_uo[0, i] = np.nan


                        y_vo = tem_vo[:, i]
                        mask = np.where(~np.isnan(y_vo))[0]
                        if x[mask[0]] <= 15 and x[mask[-1]]>=15:
                            cs_vo = CubicSpline(x[mask], y_vo[mask])
                            x_new = 15
                            interp_vo[0, i] = cs_vo(x_new)
                        else:
                            interp_vo[0, i] = np.nan

                    end_time = time.time()
                    print('gdp vertical interp:',end_time-start_time, 's')
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                        baseline_data_interp[b_name]['time'] = {}
                    baseline_data_interp[b_name]['time']['uo']= interp_uo.astype(np.float32)
                    baseline_data_interp[b_name]['time']['vo']= interp_vo.astype(np.float32)
                elif t_name == 'weekly':
                    tem_uo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    tem_vo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    interp_uo = np.full((1, t_lat.shape[0]), np.nan)
                    interp_vo = np.full((1, t_lat.shape[0]), np.nan)
                    ###.........
                    for idep in range(ndepth):
                        uo_pred = baseline_data[b_name]['weekly'].uo.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), uo_pred,bounds_error=False)
                        tem_uo[idep, :] = interp2(t_grid)
                        vo_pred = baseline_data[b_name]['weekly'].vo.values[clm_depth[idep], :, :]
                        interp3 = RegularGridInterpolator((s_lat, s_lon), vo_pred,bounds_error=False)
                        tem_vo[idep, :] = interp3(t_grid)
                    for i in range(t_lat.shape[0]):
                        x = s_depth
                        y_uo = tem_uo[:, i]
                        mask = np.where(~np.isnan(y_uo))[0]
                        if mask.size == 0:
                            continue
                        if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                            cs_uo = CubicSpline(x[mask], y_uo[mask])
                            x_new = 15
                            interp_uo[0, i] = cs_uo(x_new)
                        else:
                            interp_uo[0, i] = np.nan


                        y_vo = tem_vo[:, i]
                        mask = np.where(~np.isnan(y_vo))[0]
                        if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                            cs_vo = CubicSpline(x[mask], y_vo[mask])
                            x_new = 15
                            interp_vo[0, i] = cs_vo(x_new)
                        else:
                            interp_vo[0, i] = np.nan
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                        baseline_data_interp[b_name]['weekly'] = {}
                    baseline_data_interp[b_name]['weekly']['uo'] = interp_uo.astype(np.float32)
                    baseline_data_interp[b_name]['weekly']['vo'] = interp_vo.astype(np.float32)
                elif t_name == 'monthly':
                    tem_uo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    tem_vo = np.full((ndepth, t_lat.shape[0]), np.nan)
                    interp_uo = np.full((1, t_lat.shape[0]), np.nan)
                    interp_vo = np.full((1, t_lat.shape[0]), np.nan)
                    ###.........
                    for idep in range(ndepth):
                        uo_pred = baseline_data[b_name]['monthly'].uo.values[clm_depth[idep], :, :]
                        interp2 = RegularGridInterpolator((s_lat, s_lon), uo_pred,bounds_error=False)
                        tem_uo[idep, :] = interp2(t_grid)
                        vo_pred = baseline_data[b_name]['monthly'].vo.values[clm_depth[idep], :, :]
                        interp3 = RegularGridInterpolator((s_lat, s_lon), vo_pred,bounds_error=False)
                        tem_vo[idep, :] = interp3(t_grid)
                    for i in range(t_lat.shape[0]):
                        x = s_depth
                        y_uo = tem_uo[:, i]
                        mask = np.where(~np.isnan(y_uo))[0]
                        if mask.size == 0:
                            continue
                        if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                            cs_uo = CubicSpline(x[mask], y_uo[mask])
                            x_new = 15
                            interp_uo[0, i] = cs_uo(x_new)
                        else:
                            interp_uo[0, i] = np.nan

                        y_vo = tem_vo[:, i]
                        mask = np.where(~np.isnan(y_vo))[0]
                        if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                            cs_vo = CubicSpline(x[mask], y_vo[mask])
                            x_new = 15
                            interp_vo[0, i] = cs_vo(x_new)
                        else:
                            interp_vo[0, i] = np.nan
                    if b_name not in baseline_data_interp.keys():
                        baseline_data_interp[b_name] = {}
                        baseline_data_interp[b_name]['monthly'] = {}
                    baseline_data_interp[b_name]['monthly']['uo'] = interp_uo.astype(np.float32)
                    baseline_data_interp[b_name]['monthly']['vo'] = interp_vo.astype(np.float32)
        else:
            tem_uo = np.full((ndepth, t_lat.shape[0]), np.nan)
            tem_vo = np.full((ndepth, t_lat.shape[0]), np.nan)
            interp_uo = np.full((1, t_lat.shape[0]), np.nan)
            interp_vo = np.full((1, t_lat.shape[0]), np.nan)
            ###.........
            for idep in range(ndepth):
                uo_pred = baseline_data[b_name]['uo'][0, idep, :, :]
                interp2 = RegularGridInterpolator((s_lat, s_lon), uo_pred,bounds_error=False)
                tem_uo[idep, :] = interp2(t_grid)
                vo_pred = baseline_data[b_name]['vo'][0, idep, :, :]
                interp3 = RegularGridInterpolator((s_lat, s_lon), vo_pred,bounds_error=False)
                tem_vo[idep, :] = interp3(t_grid)
            for i in range(t_lat.shape[0]):
                x = s_depth
                y_uo = tem_uo[:, i]
                mask = np.where(~np.isnan(y_uo))[0]
                if mask.size == 0:
                    continue
                if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                    cs_uo = CubicSpline(x[mask], y_uo[mask])
                    x_new = 15
                    interp_uo[0, i] = cs_uo(x_new)
                else:
                    interp_uo[0, i] = np.nan


                y_vo = tem_vo[:, i]
                mask = np.where(~np.isnan(y_vo))[0]
                if x[mask[0]] <= 15 and x[mask[-1]] >= 15:
                    cs_vo = CubicSpline(x[mask], y_vo[mask])
                    x_new = 15
                    interp_vo[0, i] = cs_vo(x_new)
                else:
                    interp_vo[0, i] = np.nan
            if b_name not in baseline_data_interp.keys():
                baseline_data_interp[b_name] = {}
            baseline_data_interp[b_name]['uo'] = interp_uo.astype(np.float32)
            baseline_data_interp[b_name]['vo'] = interp_vo.astype(np.float32)
    #gdp_data_filtered:dict 1d numpy array ; baseline_data_interp: dict 1d numpy array
    return gdp_data_filtered, baseline_data_interp, t_grid
def interp_ZOS(zos_data, baseline_data, s_lat, s_lon):
    baseline_data_interp = {}
    zos_data_filtered= {}

    lat = zos_data.latitude
    lon = zos_data.longitude
    zos_obs = zos_data.sla_filtered

    s_grid = np.meshgrid(s_lon, s_lat)
    s_grid = np.column_stack((s_grid[1].ravel(), s_grid[0].ravel()))

    t_lat = lat.values
    t_lon = (lon.values-180) % 360 -180
    t_grid = np.column_stack((t_lat, t_lon))

    zos_data_filtered['zos'] = zos_obs.values

    mdt = xr.open_dataset('/hpcfs/fhome/yangjh5/jhm_data/ground_truth_rest/MDT_1.40625.nc')

    for b_name in baseline_data:
        if b_name == 'climatology':
            zos_pred = baseline_data['climatology']['time'].zos.values - mdt.mdt.values
            interp = RegularGridInterpolator((s_lat,s_lon), zos_pred,bounds_error=False)
            interp_zos = interp(t_grid)

            baseline_data_interp[b_name]= {}
            baseline_data_interp[b_name]['time'] = {}
            baseline_data_interp[b_name]['time']['zos'] = interp_zos.astype(np.float32)

            zos_pred = baseline_data['climatology']['weekly'].zos.values - mdt.mdt.values
            interp = RegularGridInterpolator((s_lat, s_lon), zos_pred,bounds_error=False)
            interp_zos = interp(t_grid)

            baseline_data_interp[b_name]['weekly'] = {}
            baseline_data_interp[b_name]['weekly']['zos'] = interp_zos.astype(np.float32)

            zos_pred = baseline_data['climatology']['monthly'].zos.values - mdt.mdt.values
            interp = RegularGridInterpolator((s_lat, s_lon), zos_pred,bounds_error=False)
            interp_zos = interp(t_grid)

            baseline_data_interp[b_name]['monthly'] = {}
            baseline_data_interp[b_name]['monthly']['zos'] = interp_zos.astype(np.float32)

        else:
            zos_pred = baseline_data[b_name]['zos'][0] - mdt.mdt.values
            interp = RegularGridInterpolator((s_lat, s_lon), zos_pred,bounds_error=False)
            interp_zos = interp(t_grid)

            baseline_data_interp[b_name] = {}
            baseline_data_interp[b_name]['zos'] = interp_zos.astype(np.float32)

    # zos_data_filtered: dict:numpy array     baseline_data_interp: dict:numpy array
    return zos_data_filtered, baseline_data_interp, t_grid




