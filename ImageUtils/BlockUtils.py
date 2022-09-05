import numpy as np

def prepare_image(I, b):
    I = I / 255. if np.max(I) > 1 else I
    h, w = I.shape
    hh, ww = h // b, w // b
    return I[: hh*b, : ww*b]

''' Iterator of a function over every block of size 'bsize'.
    If size is an integer, size is reassigned as a tuple (bsize, bsize)
    If stride<1, disjoint blocks are used '''
def block_map(data, function, bsize, stride=0):
    h, w = data.shape
    # Verifying values
    if type(bsize) == type(0):
        b0, b1 = (bsize, bsize)
    if type(stride) == type(0):
        s0, s1 = (stride, stride)
    s0 = s0 if s0 > 0 else b0
    s1 = s1 if s1 > 0 else b1

    # Store results in two dimension array
    results = []
    for i in range(0, h - b0 + s0, s0):
        row = []
        for j in range(0, w - b1 + s1, s1):
            D = data[i: i + b0, j : j + b1]
            f = function(D)
            row.append(f)
        results.append(row)
    return results

''' Computes the correlation similarity  '''
def block_corr_similarity(data, P):
    h, w = data.shape
    #for i in range(0, h, B):
    hp, wp = P.shape
    Pf = P.flatten()
    X = [data[i: i + hp, j: j + wp].flatten() for i in range(0, h - hp + 1, hp) for j in range(0, w - wp + 1, wp)]
    S = list(map(lambda x: np.corrcoef(x, Pf)[0,1], X))
    S = np.array(S).reshape((h - hp) // hp  + 1, (w - wp) // wp + 1)
    S[np.isnan(S)] = np.nanmean(S)
    return S

''' Computes the covariance similarity  '''
def block_cov_similarity(data, P):
    h, w = data.shape
    #for i in range(0, h, B):
    hp, wp = P.shape
    Pf = P.flatten()
    X = [data[i: i + hp, j: j + wp].flatten() for i in range(0, h - hp + 1, hp) for j in range(0, w - wp + 1, wp)]
    S = list(map(lambda x: np.cov(x, Pf)[0,1], X))
    S = np.array(S).reshape((h - hp) // hp  + 1, (w - wp) // wp + 1)
    return S

def block_mean(data, B):
    h, w = data.shape
    X = [data[i: i + B, j: j + B] for i in range(0, h - B, B) for j in range(0, w - B, B)]
    return sum(X) / len(X)