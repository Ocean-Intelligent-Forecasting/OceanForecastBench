import numpy as np
from datetime import datetime,timedelta
from tqdm import tqdm
start_date = datetime(2019,1,2)

data = np.load('/hpcfs/fhome/yangjh5/benchproject/FourCastNet-master/inference_result/2019.npy')
mask = np.load('/hpcfs/fhome/yangjh5/jhm_data/oceanrongqiyun/climaxswin_1.40625/src/climax/global_forecast/mask.npy')
for i in tqdm(range(data.shape[0])):
    date_str = start_date.strftime('%Y%m%d')
    save_data = np.concatenate([data[i,[0,3],:,:],data[i,4:,:,:]], axis=0)
    save_data[mask[0]] = np.nan
    np.save(f'../inference_result_split/10/pred_mra5_{date_str}.npy', save_data[np.newaxis,:])
    start_date += timedelta(days=1)
