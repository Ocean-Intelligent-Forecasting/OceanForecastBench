import numpy as np
import os
from tqdm import tqdm

pred_dir = '/hpcfs/fhome/yangjh5/jhm_data/baseline/ClimaX/1'
os.makedirs('/hpcfs/fhome/yangjh5/jhm_data/baseline/ClimaX/1_masked')
mask = np.load('mask.npy')
for ifp in os.listdir(pred_dir):
    fp = os.path.join(pred_dir, ifp)
    x = np.load(fp)
    x[mask] = np.nan
    np.save('/hpcfs/fhome/yangjh5/jhm_data/baseline/ClimaX/1_masked/'+ifp, x)

