import numpy as np
from math import sqrt
from scipy import stats
from sklearn import metrics

def mae(y, f):
    mae = metrics.mean_absolute_error(y, f)
    return mae

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def r_squared(y, f):
    sse = np.sum((y - f) ** 2)
    ssr = np.sum((f - np.mean(y)) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    return r2

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci