#!/home/john/.virtualenvs/cv/bin/python

# scan binary image to find columns with non-zero pixels
# print start-index and length of each (non-zero) block of pixels
# J.Beale 14-June-2019

import os     # for file basename extraction
import sys    # for command line arguments
import cv2
import numpy as np
# import matplotlib.pyplot as plt

# averaged background level of doppler spectrogram
bk257 = np.array(
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 8, 2,14,16,17,17,18,18,18,18,
 18,18,18,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,21,21,21,21,
 21,21,21,21,21,22,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,24,24,24,
 24,24,24,24,24,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,27,27,27,27,
 27,27,27,27,27,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,30,30,30,30,
 30,30,30,30,31,31,31,31,32,31,31,31,32,32,32,32,32,32,32,33,33,33,33,33,
 33,33,33,33,34,34,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,36,36,
 36,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37,37,37,38,37,38,38,38,
 38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,
 38,38,38,38,38,37,37,37,37,37,36,36,36,35,35,34,34,33,33,32,31,30,29,27,
 26,24,22,20,18,15,12, 9, 6, 3, 1, 0, 0, 0, 0, 0, 0] )
bkcol = np.asarray(bk257, dtype="uint8")  # background level, one column


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
    # print("%d: x=%d, size=%d" % (objnum, imax+offset, imin), end ="")
    #start index, length
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
# and 1 column of pixels averaged all rows, of background (non-event columns)
def doOneImg(src_path):
  margin = 10    # buffer pixels after detected object to include in region
  print(src_path, end=" : ")
  raw_path = src_path[4:] # remove first 4 chars
  img1 = cv2.imread(src_path, 0) # detected image 0 imports a grayscale
  raw_img = cv2.imread(raw_path, 0) # original raw doppler spectrogram

  xTotal = 0                # total x pixels in events so far
  nzcols = np.amax(img1,0) # maximum value in each column
  olist = findObj(nzcols)  # list with position & size (x-pixelcount) of objects
  eCount = len(olist)      # number of events found
  if (eCount > 0):
      xstart = 0   # start of current area of interest
      # print(raw_path)
      for x in olist:
        xpos = x[0]  # starting index of this object
        xsize = x[1] # x pixels included in this object
        print("%d,%d " % (xpos, xsize), end="")
        xpos2 = min(xpos+xsize+margin,nzcols.size-1) # end index of region
        if (xTotal == 0):  # for the very first region
          eSum = raw_img[:, xstart:xpos] # empty area before event
          rSum = raw_img[:, xpos:xpos2] # 1st event (raw img)
          rSum1 = img1[:, xpos:xpos2]   # 1st event (detected img)
        else:
          eregion = raw_img[:, xstart:xpos]
          region = raw_img[:, xpos:xpos2]
          eSum = np.concatenate((eSum, eregion), axis=1) # combine non-events
          rSum = np.concatenate((rSum, region), axis=1) # combine events
          rSum1 = np.concatenate((rSum1, img1[:,xpos:xpos2]), axis=1) # combine events
        xstart = xpos2 # advance area of interest past current one
        xTotal += x[1]    # count total x pixels in all events
      eregion = raw_img[:, xstart:] # include remaining unused pixels, if any
      eSum = np.concatenate((eSum, eregion), axis=1) # all non-event area
      eAvg = np.mean(eSum, axis=1) # avg of each row across non-event bkgnd
  else:
      eAvg = np.mean(raw_img, axis=1) # if no events, everything is bkgnd

  print()
  # print(eAvg.astype(int))  # DEBUG print out background average column

  ShowImage = False   # whether to show events from each image
  if (xTotal > 0) and ( ShowImage ):
      (y,x) = rSum.shape  # find dimensions of array
      # print("x:%d  y:%d" % (x,y))
      blur = cv2.blur(rSum,(9,3))
      bk2 = np.transpose(np.tile(bkcol,(x,1)))
      fg = cv2.subtract(blur, bk2)  # subtract background to see foreground
      cv2.imshow('events',rSum)  # DEBUG display regions with events
      cv2.imshow('events_det',rSum1)  # DEBUG display regions with events
      # cv2.imshow('events w/blur',blur)  # DEBUG display regions with events
      # cv2.imshow('background',bk2)  # DEBUG display regions with events
      cv2.imshow('foreground',fg)  # DEBUG display regions with events
      #cv2.imshow('non-event',eSum)  # DEBUG display regions without events
      cv2.waitKey(0)
  return (eCount, xTotal, np.asarray(eAvg, dtype="uint8"))

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

firstImg = True
for file in sorted(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".png") and filename.startswith("Det_"):
         # print(filename)
         (tE, tX, bS ) = doOneImg(filename)
         if (firstImg):
             bSum = bS
             firstImg = False
             cSize = bS.size  # how may elements in this 1D array?
         else:
             bSum = np.append(bSum, bS)
             # generate averaged bkgnd vs.time

         totalEvents += tE  # total events seen
         totalXPixels += tX # total x pixels in all events
         fCount += 1
         continue
     else:
         continue

avgXPixels = (1.0 * totalXPixels) / totalEvents
hours = (fCount * secondsPerFile) / (60.0*60.0)
print("Files:%d Hours:%5.3f Events:%d Events/Hr:%5.3f Avg.Seconds:%5.3f" %
   (fCount, hours, totalEvents, totalEvents/hours, avgXPixels/10))

np.set_printoptions(precision=1, suppress=True)
bImg = np.transpose(bSum.reshape(fCount, cSize))
(ys, xs) = bImg.shape
print(xs, ys, bImg.dtype)
bkAvg = np.mean(bImg, axis=1)

# print(bkAvg)
print()
diff = bkAvg - bk257
print(diff)  # difference in background of this set of files from preset
print("Min:Max of diff: %3.1f %3.1f" % (diff.min(), diff.max()) )
cv2.imwrite("avgBackground.png",bImg)  # save image to disk
cv2.imshow('Background vs. Time',bImg)  # background vs time
cv2.waitKey(0)


raise SystemExit  # DEBUG quit here

# -------------------------------------------
