from ImageUtils.BlockUtils import *
from ImageUtils.CFAUtils import *
from scipy import ndimage
'''
Implementation of tamper detection technique published in:
P. Ferrara, T. Bianchi, A. De Rosa, and A. Piva. Image forgery localization via fine-grained analysis of CFA artifacts. IEEE T Inf Foren Sec, 7(5):1566–1577, October 2012.
'''

def blockSVDResidue(block):
    U, s, V = np.linalg.svd(block)
    S = np.zeros(np.shape(block))
    for i in range(1, block.shape[0]):
        S[i, i] = s[i]
    predict = U @ S @ V
    return predict[s.size // 2, s.size // 2]

def slideSVDResidue(data, Q):
    h, w = data.shape
    S = np.zeros((h, w))
    offset = Q//2
    data = cv2.copyMakeBorder(data, offset, offset, offset, offset, borderType=cv2.BORDER_REFLECT101)
    for i in range(h):
        for j in range(w):
            S[i,j] = blockSVDResidue(data[i: i+Q, j:j+Q])
    return S

'''
    img: 
    B: Block size for detection
    K: 
'''
def ParkAnalysis(I, b, k, q=3):
    I = prepare_image(I, b)
    hh, ww = I.shape[0] // b, I.shape[1] // b
    bb = b//2

    # Extracting 4 images for each green filter position
    II = [np.array(extract_acquired(I, pat)).reshape((hh * bb, ww * bb)) for pat in [m_1000, m_0100, m_0010, m_0001]]

    # Predicted residue and local variance matrices
    E = [slideSVDResidue(I, q) for I in II]
    W = 1. * np.ones((2*k+1, 2*k+1)) / ((2*k+1)**2)
    mu = [cv2.filter2D(e, -1, W) for e in E]
    mu2 = [cv2.filter2D(e ** 2, -1, W) for e in E]
    sigmas2 = [x[1] - x[0]**2 +.000000001 for x in zip(mu, mu2)]

    # Block sum variance
    M = [np.array(block_map(s, np.sum, bb)).reshape(hh, ww) for s in sigmas2]

    # Initial features
    Fl = np.log((M[1]+M[2]) / (M[0]+M[3]))
    Fr = np.log((M[0] + M[3]) / (M[1] + M[2]))

    # Final feature and probability map
    F = Fr if np.sum(Fr) > np.sum(Fl) else Fl
    F = ndimage.median_filter(F, size=5)
    return 1 / (1 + np.e**F)



