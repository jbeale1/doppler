#!/usr/bin/octave

# find velocity, distance from doppler-radar spectrogram images
# J.Beale 6-May-2021

# output CSV column headers
# printf("date,start_s,dur_s,mph,feet,Sum,Fdirection\n");

arg_list = argv ();              # command-line inputs to this function
args = length(arg_list);
if (args < 1)
  printf("Must specify input filename\n");
  exit(1);
endif  

pathname = arg_list{1};      # first argument is input filename
                             # eg. "DpA_2021-05-01_16-00-00.png"

[wdir, froot, ext] = fileparts (pathname);  # parts of file name
if (length(wdir) == 0)
  wdir=".";
endif

fname=strcat(froot,ext);               # filename without path
fname2=strcat(wdir,"/Det_",fname);     # input detected mask image 
fvout=strcat(wdir,"/V_",froot,".csv"); # csv out: velocity per pixel
fsumout=strcat(wdir,"/S_",froot,".csv"); # csv out: column sums
fmaskout=strcat(wdir,"/masked_",fname);       # output masked image

#if exist(fmaskout,"file");  # don't run if output file already exists
#  exit(0);
#endif

fdate = froot(5:end);  # format: 2021-05-01_16-00-00
[tmStr, nch] = strptime(fdate, "%Y-%m-%d_%H-%M-%S");  # get time struct
if (nch != 20)
  printf("Could not find date from filename %s\n",froot);
  printf("Expected something like DpA_2021-05-01_16-00-00\n");
  exit(1);
endif
f_epoch = mktime(tmStr);  # file start time, Unix epoch (but local TZ)

mphFac =  2.237; # mph per m/s scaling factor 1 mph  = 0.44704  m/s
kphFac = 3.6;    # 1 km/h = 0.277778 m/s
ftpm = 3.28084;  # feet per meter

vcal = 0.100094; # (m/s) / (pixel) speed calibration
spp = 0.1;       # seconds per pixel sonogram timescale calibration

try
  im1 = imread(pathname);  # original data image
  im2 = imread(fname2); # detected-mask image
catch
  [msg, msgid] = lasterr();  # if a file was missing
  printf("Error: %s\n",msg);
  exit(1);
end_try_catch

imf = double(im1) .* im2;   # continuous image data * noise mask
pixcol = sum(imf);           # sum of each pixel column
sigmar = pixcol';              # transpose cols to rows

imo = uint8(imf);
imwrite(imo,fmaskout);  # save out masked image

imf1 = double(imf);   # use floating-point instead of uint8 values
[xs ys] = size(imf1); # dimensions of input image

[max_values indices] = max(imf1);
mask = (max_values > 2)';
i2 = (xs - indices') .* mask;  # 1 is lower-left instead of upper-l

dX = i2 * vcal; # units of m/s (per pixel)
speedV = dX * mphFac;   # units of mph (per pixel)

Xt = cumsum(dX * spp);        # sum of X distance travelled (m)
mT = Xt(end);                 # total distance (m)
ftT = mT * ftpm;              # convert meters to feet
msMax = max(dX);              # max speed in m/s
mphMax = msMax * mphFac;      # max speed in mph
kphMax = msMax * kphFac;      # max speed in km/h
  
vset = [speedV sigmar];  # 2 cols: velocity, pixel column-sums

# -------------------------------------------

spp = 0.1;  # seconds per pixel (3000 in 5 minutes => 10 pixels/sec)
mask = (sigmar > 0)';  # convert to binary, 1/0
starts = find (diff ([0, mask]) ==  1);  # index of each event start
ends   = find (diff ([mask, 0]) == -1);
ecount = length(starts);


for i = 1:ecount             # step through each event found
  a = starts(i);   # starting and ending index of this event
  b = ends(i);
  eticks = (1 + ends(i) - starts(i)); # length of event in pixels
  eData = sigmar(a:b);                # part of image with this event  
  [pkval pkpos] = max(eData);         # index where signal is strongest
  pkposRatio = pkpos / eticks;       # position ratio R-L, (0...1)
  stime = (a+pkpos-1) * spp;
  atime = f_epoch + stime;        # abs time (seconds since Unix epoch)
  dur = eticks * spp;           # duration in seconds
  eSum = sum(sum(eData))/dur;   # avg pixel value per second
  [max_values indices] = max(imf1(:,a:b)); # find peak speed
  mask = (max_values > 2)';
  i2 = (xs - indices') .* mask;  # 1 is lower-left instead of upper-l
  dX = i2 * vcal; # units of m/s (per pixel) for each pixel in event
  Xt = cumsum(dX * spp);        # sum of X distance travelled (m)
  mT = Xt(end);                 # total distance (m)
  ftT = mT * ftpm;              # convert meters to feet
  [pkV pkVpos] = max(dX);
  maxMPH = pkV * mphFac;   # units of mph (per pixel)

  # Note: raindrops cause records with duration < 2 seconds
  # and speed < 10 mph
  if (dur > 1) || (maxMPH > 10)
    printf("%s, %12.1f,%04.1f, %04.1f, %03.0f,%04.0f,%04.2f\n",
     froot,atime,dur,maxMPH,ftT,eSum,pkposRatio);
  endif
endfor
