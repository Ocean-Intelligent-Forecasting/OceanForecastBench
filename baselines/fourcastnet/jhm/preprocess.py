import numpy as np 
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
mask = np.load('/hpcfs/fhome/yangjh5/jhm_data/oceanrongqiyun/climaxswin_1.40625/src/climax/global_forecast/mask.npy')
ildy = 6
start_date = datetime(2019,1,1) + timedelta(days=ildy)
data_dir = f'/hpcfs/fhome/yangjh5/benchproject/FourCastNet-master/inference/result0{ildy}'
fp_list = os.listdir(data_dir)
#fp_list = sorted(fp_list, key=lambda x: int(x.split('.')[0]))
fp_list = sorted(fp_list)
print(fp_list)
sys.exit()
for i, fp in tqdm(enumerate(fp_list)):
    print(fp+'processing....')
    date = start_date + timedelta(days=i)
    date_str = date.strftime('%Y%m%d')
    data = np.load(os.path.join(data_dir,fp))
    save_data = np.concatenate([data[:,[0,3],:,:],data[:,4:,:,:]], axis=1)
    save_data[mask] = np.nan
    os.makedirs(f'/hpcfs/fhome/yangjh5/jhm_data/baseline_1.40625/FourCastNet/{ildy}', exist_ok=True)
    np.save(f'/hpcfs/fhome/yangjh5/jhm_data/baseline_1.40625/FourCastNet/{ildy}/pred_mra5_{date_str}.npy', save_data)

