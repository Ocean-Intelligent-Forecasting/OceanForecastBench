import numpy as np

data = np.load('mra5_20171231.npy')
mask = np.where(np.isnan(data), 0, 1)
# mask = np.pad(mask, ((0, 0), (0, 0), (0, 7), (0, 0)), 'constant')
mask = np.tile(mask, (4, 1, 1, 1))
trans_mask = mask.transpose((1, 0, 2, 3))
mask = np.concatenate((trans_mask[0:1], trans_mask[3:]), axis=0).transpose((1, 0, 2, 3))
print(mask.shape)
np.save('land_sea_mask.npy', mask)