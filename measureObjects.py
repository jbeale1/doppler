#!/home/john/.virtualenvs/cv/bin/python

# scan binary image to find columns with non-zero pixels
# print start-index and length of each (non-zero) block of pixels
# J.Beale 14-June-2019

import os     # for file basename extraction
import sys    # for command line arguments
import cv2
import numpy as np
# import matplotlib.pyplot as plt

pixelsPerFile = 3000       # 10 pixels per second
secondsPerFile = 5.0 * 60  # each file = 5 minutes

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
# scan one image for objects
# return count of events found, and total # of x pixels within events
def doOneImg(src_path):
  print(src_path, end=" : ")
  img1 = cv2.imread(src_path, 0) # 0 imports a grayscale
  xsize = 0                # total x pixels in events so far
  nzcols = np.amax(img1,0) # maximum value in each column
  olist = findObj(nzcols)  # find position & size (x-pixelcount) of objects
  eCount = len(olist)      # number of events found
  if (eCount > 0):
    for x in olist:
      print("%d,%d " % (x[0],x[1]), end="")
      xsize += x[1]    # count total x pixels in all events
  print()
  return (eCount, xsize)

#--------------------------------------------------
# main program starts here

arguments = len(sys.argv) - 1
if (arguments < 1):
  print ("Usage: %s directory" % (sys.argv[0]))
  exit()

src_dir = sys.argv[1]  # input file is 1st argument on command line

directory = os.fsencode(src_dir)

fCount = 0  # how many files processed so far
totalEvents = 0            # count of all events so far
totalXPixels = 0           # all x pixels in events so far

for file in sorted(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".png") and filename.startswith("Det_"):
         # print(filename)
         (tE, tX ) = doOneImg(filename)
         totalEvents += tE  # total events seen
         totalXPixels += tX # total x pixels in all events
         fCount += 1
         continue
     else:
         continue

avgXPixels = (1.0 * totalXPixels) / totalEvents
hours = (fCount * secondsPerFile) / (60.0*60.0)
print("Files: %d Hours: %5.3f Events: %d Events/Hr: %5.3f Avg.Seconds: %5.3f" %
   (fCount, hours, totalEvents, totalEvents/hours, avgXPixels/10))
raise SystemExit  # DEBUG quit here

# -------------------------------------------
