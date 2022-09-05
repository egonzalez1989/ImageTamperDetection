import cv2, numpy as np

# Help masks for bayer filter patterns
m_1000 = np.array([[1, 0], [0, 0]])
m_0100 = np.array([[0, 1], [0, 0]])
m_0010 = np.array([[0, 0], [1, 0]])
m_0001 = np.array([[0, 0], [0, 1]])
m_1001 = m_1000 + m_0001
m_0110 = m_0100 + m_0010

def get_filter_pattern(cfa_pattern, height, width):
    h, w = cfa_pattern.shape
    cfa_matrix = np.tile(cfa_pattern, ((height + h - 1) // h, (width + w - 1) // w))
    cfa_matrix = cfa_matrix[: height, : width]
    return cfa_matrix

''' Extracts values from interpolated values. This only works for green band in xGGx or GxxG patterns.
'''
def extract_interpolated(data, cfa_pattern):
    h, w = data.shape
    true_array = get_filter_pattern(np.ones_like(cfa_pattern) - cfa_pattern, h, w)
    return np.extract(true_array, data)

def extract_acquired(data, cfa_pattern):
    h, w = data.shape
    true_array = get_filter_pattern(cfa_pattern, h, w)
    return np.extract(true_array, data)

def zeroize_pattern(img, cfa_pattern):
    # Zeroize interpolated pixels
    h, w = img.shape
    mosaic = get_filter_pattern(cfa_pattern, h, w)
    return mosaic * img

def zeroize_acquired(img, cfa_pattern):
    pattern = np.ones_like(cfa_pattern) - cfa_pattern
    return zeroize_pattern(img, pattern)

def zeroize_interpolated(img, cfa_pattern):
    return zeroize_pattern(img, cfa_pattern)

'''
'''
def filter_green_data(data, algorithm, cfa_pattern = None):
    if algorithm == 'BILINEAR':
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.
        if cfa_pattern is not None:
            kernel[1, 1] = 1
            data = zeroize_interpolated(data, cfa_pattern)
        fdata = cv2.filter2D(data, -1, kernel, cv2.BORDER_REFLECT_101)
    elif algorithm == 'BICUBIC':
        kernel = np.array(
            [[0, 0, 0, 1, 0, 0, 0], [0, 0, -9, 0, -9, 0, 0], [0, -9, 0, 81, 0, -9, 0], [1, 0, 81, 0, 81, 0, 1],
             [0, -9, 0, 81, 0, -9, 0], [0, 0, -9, 0, -9, 0, 0], [0, 0, 0, 1, 0, 0, 0]]) / 256.
        if cfa_pattern is not None:
            kernel[1, 1] = 1
            data = zeroize_interpolated(data, cfa_pattern)
        fdata = cv2.filter2D(data, -1, kernel, cv2.BORDER_REFLECT_101)
    return fdata