#!/home/john/.virtualenvs/cv/bin/python

# scan binary image to find columns with non-zero pixels
# print start-index and length of each (non-zero) block of pixels
# J.Beale 14-June-2019

import os     # for file basename extraction
import sys    # for command line arguments
import cv2
import numpy as np
# import matplotlib.pyplot as plt

# -----------------------
# scan through 1D array, find index & length of contiguous non-zero elements
def findObj(suba):
  objnum = 0
  offset = 0  # running sum from past object index offsets
  objs = []   # initialize list of objects

  colCnt = suba.size     # how many columns in original image
  # print("Cols: %d" % (colCnt))

  while True:
    maxT = np.amax(suba)   # overall maximum (0 = nothing detected anywhere)
    if (maxT == 0):
      return objs

    objnum += 1
    imax = np.argmax(suba) # find index of 1st maximum
    suba = suba[imax:]
    s=suba.size
    imin = np.argmin(suba) # find next 0 elements
    if (imin == 0):
        imin = colCnt - (imax+offset)
        eflag = True
    else:
        eflag = False

    # if imin=0 that means the object extends past right-hend edge of data
    # print("%d: x=%d, size=%d" % (objnum, imax+offset, imin), end ="")  #start index, length
    objs += [[imax+offset, imin]]
    if (eflag):  # object went up to RH edge
        # print(", EF")
        return objs
    # else:
    #    print("")
    suba = suba[imin:]
    offset = offset + imax + imin  # index of new 'suba' in original

# -------------------------------------------
# process one image for objects
def doOneImg(src_path):
  print(src_path, end=" : ")
  img1 = cv2.imread(src_path, 0) # 0 imports a grayscale
  nzcols = np.amax(img1,0) # maximum value in each column
  olist = findObj(nzcols)  # find position & size of found objects
  if (len(olist) > 0):
    for x in olist:
      print("%d,%d " % (x[0],x[1]), end="")
  print()

#--------------------------------------------------
# main program starts here

arguments = len(sys.argv) - 1
if (arguments < 1):
  print ("Usage: %s directory" % (sys.argv[0]))
  exit()

src_dir = sys.argv[1]  # input file is 1st argument on command line

directory = os.fsencode(src_dir)

procCount = 0  # how many files processed so far
for file in sorted(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".png") and filename.startswith("Det_"):
         # print(filename)
         doOneImg(filename)
         procCount += 1
         continue
     else:
         continue

print("Total files: %d" % (procCount))
raise SystemExit  # DEBUG quit here

# -------------------------------------------
# print("Done with %s" % src_path)

# print(nzcols)
