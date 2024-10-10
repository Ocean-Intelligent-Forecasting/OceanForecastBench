#!/bin/bash

for i in {1..10}; do
    /hpcfs/fhome/yangjh5/anaconda3/envs/climax_v2/bin/python3.8 npy2nc_1.40625.py "/hpcfs/fhome/yangjh5/jhm_data/baseline_1.40625/LinearRegression_v1001_sst/${i}" "/hpcfs/fhome/yangjh5/jhm_data/baseline_nc_1.40625/LinearRegression_v1001_sst/${i}"

done
