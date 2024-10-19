import numpy
import datetime

import numpy as np

file_path = '/hpcfs/fhome/yangjh5/jhm_data/1.40625deg_npy/data'
dat = datetime.date(1993, 1, 1)
d = 0
file_name = 'mra5_'+dat.isoformat().replace('-', '')+'.npy'
matrix = np.zeros((1, 96, 121, 256))
while dat<=datetime.date(2017, 12, 31):
    # file = file_path + '/train/' + file_name
    # data = np.load(file)
    # matrix += data
    dat = dat + datetime.timedelta(days=10)
    file_name = 'mra5_' + dat.isoformat().replace('-', '') + '.npy'
    print(file_name)
    d += 1
while dat<=datetime.date(2018, 12, 31):
    file = file_path + '/val/' + file_name
    data = np.load(file)
    matrix += data
    dat = dat + datetime.timedelta(days=10)
    file_name = 'mra5_' + dat.isoformat().replace('-', '') + '.npy'
    d += 1
matrix = matrix / d
print(matrix.shape)
print(matrix)
np.save('stats_v0/time_means.npy', matrix)