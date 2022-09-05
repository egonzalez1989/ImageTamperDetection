'''
Technique proposed by Edgar Gonzalez et al (Phd dissertation):
Gonzalez Fernandez E. Image tampering detection techniques based on chromatic interpolation and sensor noise algorithms. 2022.
'''

from ImageUtils.BlockUtils import *
from ImageUtils.CFAUtils import *
from ImageUtils.DetectionUtils import *
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)

def SimilarityPatternAnalysis(img, b, c, withabs=False, withpmap = False):
    # Scale image to range [0,1] and denoise with TV denoise
    img = img / 255. if np.max(img) > 1 else img
    F = img - denoise_tv_chambolle(img, weight=.01)

    # If analysis is required on absolute values
    if withabs:
        F = np.abs(F)
    # Probability map can be computed estimating a minimum variance on possible acquired sets
    if withpmap:
        s = estimateSigma(F, c)
        F = pmap_erfc(F, mean=0, stdev=s)        # Consider mean = 0
    pattern = block_mean(F, b)                   # Estimation of pattern as a simple average
    C = block_corr_similarity(F, pattern)        # Correalation martix. This can be computed on overlapping blocks, recommendable with strides of size B
    C = cv2.GaussianBlur(C, (5, 5), 0)
    return em_probability(C)

# Estimation of standard deviation
def estimateSigma(data, S):
    filter = np.zeros((S, S))
    std = np.inf
    for i in range(S):
        for j in range(S):
            filter[i, j] = 1
            std = min(std, np.std(extract_acquired(data, filter)))
            filter[i, j] = 0
    return std