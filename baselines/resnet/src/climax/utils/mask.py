import numpy as np
import os
from tqdm import tqdm

pred_dir = '/hpcfs/fhome/yangjh5/jhm_data/baseline/ResNet_nonorm/10'
os.makedirs('/hpcfs/fhome/yangjh5/jhm_data/baseline/ResNet_nonorm/10_masked')
mask = np.load('mask.npy')
for ifp in tqdm(os.listdir(pred_dir)):
    fp = os.path.join(pred_dir, ifp)
    x = np.load(fp)
    x[mask] = np.nan
    np.save('/hpcfs/fhome/yangjh5/jhm_data/baseline/ResNet_nonorm/10_masked/'+ifp, x)

