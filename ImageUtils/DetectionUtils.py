import cv2, numpy as np
from scipy.special import erfc
from .CFAUtils import *
from sklearn.mixture import GaussianMixture
import scipy.stats

def pmap_erfc(data, mean=None, stdev=None):
    if stdev is None: stdev = np.std(data)
    if mean is None: mean = np.mean(data),
    pmap = erfc(np.abs(data - mean) / (2 ** .5 * stdev))
    return pmap

def filtered_pmap_erfc(data, mean = 0, stdev = None, B = 1):
    K = get_filter_pattern(m_1001, 2*B+1, 2*B+1)
    pmap = pmap_erfc(data, mean, stdev)
    pmap = cv2.filter2D(pmap, -1, K)
    return pmap

def weighted_pmap_erfc(data, mean = 0, stdev = None):
    pmap = pmap_erfc(data, mean, stdev)
    fdata, nw = data * pmap, np.sum(pmap)
    mean = np.sum(fdata) / nw
    stdev = (np.sum(pmap*(data - mean)**2) / nw) ** .5
    pmap = erfc(np.abs(data - mean) / (2 ** .5 * stdev))
    return pmap

def em_probability(F):
    X = F.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    M = gm.predict(X)
    i = 0 if np.sum(M == 0) < np.sum(M == 1) else 1     # Assume that cluster with fewer elements is tampered
    m0, m1 = gm.means_[i][0], gm.means_[1-i][0]
    s0, s1 = gm.covariances_[i][0][0]**.5, gm.covariances_[1-i][0][0]**.5
    pdf0, pdf1 = scipy.stats.norm(m0, s0).pdf, scipy.stats.norm(m1, s1).pdf
    P0, P1 = pdf0(F), pdf1(F)
    return P0 / (P0 + P1)

''' Gets the biggest N components, if they are at least 100*t% the size of image 
'''
def mask_components(mask, N = np.inf, t = 0):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    N = min(nlabels, N)
    idx = np.argsort(stats[:, -1])[::-1]
    out = np.zeros_like(mask)
    for i in idx[1:]:
        _, _, _, _, a = stats[i]
        if a < t * mask.size:
            break
        out[labels == i] = 1
    return out

def otsu_components(data, N = 3, A = .01, biggest = False):
    data = data - data.min()
    data = (data / data.max() * 255).astype('uint8')
    blur = cv2.GaussianBlur(data, (N, N), 0)
    # find otsu's threshold value with OpenCV function
    _, otsu = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = 1-otsu
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu)
    idx = np.argsort(stats[:,-1])[::-1]
    out = np.zeros_like(otsu)
    for i in idx[1:]:
        _, _, _, _, a = stats[i]
        if a < A * data.size:
            break
        out[labels == i] = i
    return out
