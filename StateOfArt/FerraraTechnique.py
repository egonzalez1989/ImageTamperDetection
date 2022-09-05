'''
Implementation of tamper detection technique published in:
P. Ferrara, T. Bianchi, A. De Rosa, and A. Piva. Image forgery localization via fine-grained analysis of CFA artifacts. IEEE T Inf Foren Sec, 7(5):1566–1577, October 2012.
'''

from ImageUtils.BlockUtils import *
from ImageUtils.CFAUtils import *
from sklearn.mixture import GaussianMixture
import scipy.stats

''' Construction of the Gaussian window according to pixel membership (values alpha_ij)'''
def gaussianWindow(n):
    a = np.zeros((6 * n + 1, 6 * n + 1))  # a4.shape[0]
    a[3 * n, 3 * n] = 1
    W = cv2.GaussianBlur(a.astype(float), (2 * n + 1, 2 * n + 1), 1)[2 * n: -2 * n, 2 * n: -2 * n]
    for i in range(2*n+1):
        for j in range((i+1)%2, 2*n+1, 2):
            W[i, j] = 0
    return W / np.sum(W)

''' Computation of the L-feature: quotient of geometric means.
    block: block of size B with error data
    pattern: 0,1-array as mask of green acquired positions in the Bayer filter '''
def LFeature(block, pattern):
    A = extract_acquired(block, pattern)
    I = extract_interpolated(block, pattern)
    GMA = np.sum(np.log(A)) / A.size
    GMI = np.sum(np.log(I)) / I.size
    return GMA - GMI

''' Computation of the L-feature: quotient of geometric means '''
def LMap(featuremap):
    X = featuremap.flatten().reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, means_init=[[0], [1]]).fit(X)
    gm.means_[0] = [0]
    L = gm.predict(X)
    sigma0, sigma1 = np.std(X[L == 0]), np.std(X[L == 1])
    cfd0, cfd1 = scipy.stats.norm(0, sigma0).pdf, scipy.stats.norm(gm.means_[1], sigma1).pdf
    L = cfd0(featuremap) / cfd1(featuremap)             # Likelihood ratio Eq.(17)
    return 1 / (1 + L.reshape(featuremap.shape))        # Probability map #Eq. (16)

''' Implementation of the Ferrara technique with parameters:
    I: grayscale image of the green channel intensities
    B: size of disjoint blocks
    K: size of the Gaussian kernel used for the local variance computation
    pattern: mask for green acquired pixels in the 2X2 Bayer filter
    '''
def FerraraAnalysis(I, b, k, pattern=m_0110):
    I = prepare_image(I, b)
    hh, ww = I.shape[0] // b, I.shape[1] // b

    e = I - filter_green_data(I, 'BICUBIC')             # Interpolation algorithm can be modified
    W = gaussianWindow(k)
    c = 1 - np.sum(W ** 2)

    # Computation of local variances
    mu = cv2.filter2D(e, -1, W)
    mu2 = cv2.filter2D(e ** 2, -1, W)
    sigma2 = (mu2 - mu ** 2 + .0000001) / c     # Small value added to avoid zero values on the np.log computation

    # Computation of L feature and probability Eq.(11) on disjoint blocks
    f = lambda X: LFeature(X, pattern)
    from matplotlib import pyplot as plt
    L = np.array(block_map(sigma2, f, b)).reshape(hh, ww)
    return LMap(L)      # Probability map (16).