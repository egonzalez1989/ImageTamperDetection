from ImageTamperDetection.DetectionTechniques.FerraraTechnique import *
from ImageTamperDetection.DetectionTechniques.DCTTechnique import *
from ImageTamperDetection.DetectionTechniques.ParkTechnique import *
from ImageTamperDetection.DetectionTechniques.SimilarityPatternTechnique import *


import cv2
from matplotlib import pyplot as plt
img = cv2.imread('images/r0f2e6baft.TIF')
b, g, r = cv2.split(img)
B, S, K = 16, 4, 3


# Ferrara's result
R = 1 - FerraraAnalysis(g, B, K, pattern=m_0110)        # Shows the probability of not being tampered and pattern XGGX
plt.imshow(R, cmap='seismic')
plt.axis('off')
plt.show()


# Park's Result
R = ParkAnalysis(g, B, K)
plt.imshow(R, cmap='seismic')
plt.show()

# Gonzalez's DCT Result
R = DCTAnalysis(g, B)
plt.imshow(R, cmap='seismic')
plt.show()

# Gonzalez's Similarity Pattern Result
Rb, Rg, Rr = [SimilarityPatternAnalysis(c, B, S, withpmap = True) for c in [b,g,r]]
plt.imshow(Rb, cmap='seismic')
plt.show()
plt.imshow(Rg, cmap='seismic')
plt.show()
plt.imshow(Rr, cmap='seismic')
plt.show()