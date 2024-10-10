# Copyright (c) 2024 College of Meteorology and Oceanography, National University of Defense Technology. All rights reserved
# Code adapted from:
# https://github.com/beijixiong1/OceanForecastBench/
import numpy as np
def crmse(obs, fct):
    ts = (obs - fct) ** 2
    # ts[ts>9] = np.nan
    numb = (~np.isnan(ts)).sum()
    if numb ==0:
        rmse = np.nan
    else:
        rmse = np.sqrt(np.nansum(ts)/ numb)
    return rmse


def crmse_mask(obs, fct):
    ts = (obs - fct) ** 2
    mask = ts > 9

    #ts[ts > 9] = np.nan

    numb = (~np.isnan(ts)).sum()
    if numb == 0:
        rmse = np.nan
    else:
        rmse = np.sqrt(np.nansum(ts) / numb)
    return rmse, mask
def crmse_psal(obs, fct):
    ts = (obs - fct) ** 2
    mask = ts > 0.25
    #ts[mask] = np.nan

    numb = (~np.isnan(ts)).sum()
    if numb == 0:
        rmse = np.nan
    else:
        rmse = np.sqrt(np.nansum(ts) / numb)
    return rmse, mask
def cbias(obs, fct):
    ts = obs - fct
    numb = (~np.isnan(ts)).sum()
    if numb ==0:
        bias = np.nan
    else:
        bias = np.nansum(ts)/ numb
    return bias
def cacc_mask(obs, fct, clm, mask):
    # obs[mask] = np.nan
    # fct[mask] = np.nan
    ts1 = fct - clm
    ts2 = obs - clm
    temp1 = np.sqrt(np.nansum(ts1 ** 2))
    temp2 = np.sqrt(np.nansum(ts2 ** 2))
    temp3 = np.nansum(ts1 * ts2)
    if temp1 == 0 or temp2 == 0:
        acc = np.nan
    else:
        acc = temp3 / (temp1 * temp2)
    return acc
def cacc(obs, fct, clm):
    ts1 = fct - clm
    ts2 = obs - clm
    temp1 = np.sqrt(np.nansum(ts1 ** 2))
    temp2 = np.sqrt(np.nansum(ts2 ** 2))
    temp3 = np.nansum(ts1 * ts2)
    if temp1 == 0 or temp2 == 0:
        acc = np.nan
    else:
        acc = temp3 / (temp1 * temp2)
    return acc
def choracc(obs, fct, clm):
    ts1 = fct - clm
    ts2 = obs - clm
    temp1 = np.sqrt(np.nansum(ts1 ** 2, axis = 0))
    temp2 = np.sqrt(np.nansum(ts2 ** 2, axis = 0))
    temp3 = np.nansum(ts1 * ts2, axis = 0)
    acc = temp3 / (temp1 * temp2)
    # if (temp1 == 0).all() or (temp2 == 0).all():
    #     acc = np.nan
    # else:
    #     acc = temp3 / (temp1 * temp2)
    return acc
def cmse(obs, fct):
    ts = (obs - fct) ** 2
    numb = (~np.isnan(ts)).sum()
    if numb == 0:
        mse = np.nan
    else:
        mse = np.nansum(ts) / numb
    return mse
def cpss(obs, fct, hys):
    temp = crmse(obs, hys)
    if np.isnan(temp) or temp == 0:
        pss = np.nan
    else:
        pss = 1 - crmse(obs, fct)/temp
    return pss
def ccss(obs, fct, clm):
    temp = crmse(obs, clm)
    if np.isnan(temp) or temp == 0:
        css =np.nan
    else:
        css = 1 - crmse(obs, fct)/temp
    return css
