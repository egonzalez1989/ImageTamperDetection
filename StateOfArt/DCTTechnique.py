import cv2
from ImageUtils.BlockUtils import *
from ImageUtils.CFAUtils import *
from ImageUtils.DetectionUtils import *
from scipy.fftpack import dct
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)

def DCTAnalysis(img, B):
    # Scale image to range [0,1] and denoise with TV denoiser
    img = img / 255. if np.max(img) > 1 else img
    e = img - denoise_tv_chambolle(img, weight=.01)
    # Estimate standard deviation as the minimum of possible green patterns in CFA
    std = min(np.std(extract_acquired(e, m_1001)), np.std(extract_acquired(e, m_1001)))
    # probability map updated to use the weight probability map
    pmap = weighted_pmap_erfc(e, 0, std)
    # Computation of DCT over the probability map
    DCT = np.array(block_map(pmap, lambda X: dct(dct(X.T, type=1).T, type=1)[-1, -1] / X.size, B))

    # Probability of tampered
    DCT = cv2.GaussianBlur(DCT, (5, 5), 0)
    return em_probability(DCT)