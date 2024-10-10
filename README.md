# OceanForecastBench


The repo releases the data and code for the paper:
+ OceanForecastBench: A Benchmark Dataset for Data-Driven Medium-Range Global Ocean Forecasting

## Overview
<details open>
<summary>Major Features</summary>

- **Deep Learning-Ready Dataset for Trainging.**
  OceanForecastBench have compiled a comprehensive deep learning-ready dataset including diverse ocean variables necessary for medium-range global ocean forecasting.

- **Testing Method Based Observation.**
  OceanForecastBench constructed a testing dataset by integrating multi-source observation data.
 A standardized model evaluation pipeline is also provided to align forecast data with discrete observation data and calculate evaluation metrics.

- **Baselines.**
  - [x] ResNet
  - [x] SwinTransformer
  - [x] ClimaX
  - [x] FourCastNet
  - [x] XiHe
  - [x] PSY4
  

</details>

## Datasets and Models Download
The data and pre-trained models used in the paper is available in the following links: [Baidu Netdisk](https://pan.baidu.com/s/1THyTC4uzcB2nrhoqyP8eFw?pwd=iuw2 )

**Original Data Source**
1. `GLORYS12`
2. `ERA5`
3. `GHRSST`
4. `EN4`
5. `GDP`
6. `CMEMS L3 track`

**Traing Dataset**

The training data files shall be organized as the following hierarchy:
```plain
├── 1.40625deg_npy
│   ├── train
│   │	├── mra5_19930101.npy
│   │	├── mra5_19930102.npy
│   │	├── ...
│   ├── val
│   │	├── mra5_20180101.npy
│   │	├── mra5_20180102.npy
│   │	├── ...
├── normalize_1.40625deg
│   ├── normalize_mean_23_1.40625deg.npz
│   ├── normalize_std_23_1.40625deg.npz
├── latlon_1.40625deg
│   ├── lat.npz

│   ├── lon.npz

```
The testing data files shall be organized as the following hierarchy:


**Testing Dataset**
```plain
├── ground_truth
│   ├── temperature_salinity
│   │   ├── EN.4.2.2.profiles.g10.2022/
│   │   │	├── EN.4.2.2.f.profiles.g10.202201.nc
│   │   │	├── EN.4.2.2.f.profiles.g10.202202.nc
│   │   │	├── ...
│   ├── current_sst
│   │   ├── drifter_6hour_qc_c35b_4db8_1f03_U1718152782438.nc
│   ├── zos
│   │   ├──  202201
│   │   │	├── dt_global_alg_phy_l3_20220101_20220701.nc
│   │   │	├── dt_global_alg_phy_l3_20220102_20220701.nc
│   │   │	├── ...
│   │   ├──  202202
│   │   │	├── dt_global_alg_phy_l3_20220201_20220701.nc
│   │   │	├── dt_global_alg_phy_l3_20220202_20220701.nc
│   │   │	├── ...

```
**Pre-Trained Models**
```plain
├── ResNet
│   ├── fct_1d.ckpt
│   ├── fct_2d.ckpt
│   ├── ...
│   ├── fct_10d.ckpt
├── SwinTransformer
│   ├── fct_1d.ckpt
│   ├── fct_2d.ckpt
│   ├── ...
│   ├── fct_10d.ckpt
├── ClimaX
│   ├── fct_1d.ckpt
│   ├── fct_2d.ckpt
│   ├── ...
│   ├── fct_10d.ckpt
├── FourCastNet
│   ├── fct_1d.ckpt
│   ├── fct_2d.ckpt
│   ├── ...
│   ├── fct_10d.ckpt
```


