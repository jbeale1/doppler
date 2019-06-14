#!/home/john/.virtualenvs/cv/bin/python

# Take as input a noisy doppler-radar spectogram (from SOX)
# find likely moving objects based on ridge features (velocity curve)

import os     # for file basename extraction
import sys    # for command line arguments
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import img_as_ubyte
# ---------------------------------------------------
imgThresh = 15   # (was 20) critical value to extract clean curves, may drift with time/temp/etc
min_size = 100   # minimum size blob (in pixels) to count as a feature

arguments = len(sys.argv) - 1  
if (arguments < 2):
  print ("Usage: %s inFile outFile" % (sys.argv[0]))
  exit()

src_path = sys.argv[1]  # input file is 1st argument on command line
fname_out = sys.argv[2] # output file
fout_base = os.path.basename(fname_out)
iFname = "INT_" + fout_base  # intermediate file name
# print(iFname)  # DEBUG


def d1_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges

def print_minmax(a):
  smallest = a.min(axis=0).min(axis=0)
  biggest = a.max(axis=0).max(axis=0)
  print(smallest, biggest)
  # print(biggest)

# =================================================================
img1 = cv2.imread(src_path, 0) # 0 imports a grayscale
img = ~img1     # invert greyscale, so black becomes white, etc.
if img is None:
    raise(ValueError(f"Image didn\'t load. Check that '{src_path}' exists."))

imgF  = cv2.bilateralFilter(img,7,125,125)  # reduce speckle noise
# cv2.imwrite(iFname,imgF)  # save filtered data

a = d1_ridges(imgF, sigma=2.0)  # do ridge feature detection (eigenvals of Hessian mx)

# print_minmax(a)
# typ range:  -0.04179 ... 0.06309
ret,a = cv2.threshold(a,0,1,cv2.THRESH_TOZERO) # force all neg. values to zero
a = a * (1/0.06309) * 255.0
# print_minmax(a)
ret,a = cv2.threshold(a,255,255,cv2.THRESH_TRUNC) # clamp max values to 255

nImg = np.zeros((3000, 257))
uImg = a.astype(np.uint8)  # convert to unsigned byte
# cv2.imwrite(iFname,uImg)  # save intermediate ridge data
# ===================================================================

Bimg = cv2.bilateralFilter(uImg,5,50,50)  # smooth out the edges

#cv2.imwrite(iFname,Bimg)  # save filtered data

ret,th1 = cv2.threshold(Bimg,imgThresh,255,cv2.THRESH_BINARY)

#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th1, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1  # remove background component
img2 = np.zeros((output.shape))
objects = 0
for i in range(0, nb_components):  # keep only components larger than min_size
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255
        objects += 1
cv2.imwrite(fname_out,img2)  # save image to disk
print("%03d : %s" % (objects,fout_base))
