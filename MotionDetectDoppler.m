# GNU Octave code to detect motion in raw audio data
# from 24GHz motion sensor output
# J.Beale 07-July-2021

pkg load signal                 # uses the signal library

# arg_list = argv ();              # command-line inputs to this function
# fname = arg_list{1};             # first argument is input filename
# fname = "ChD_2021-07-06_12-50-00.flac";

function doOutput(data,fname)  # find motion events = signal above threshold
  date=fname(5:14);  # eg 2021-07-06
  time=fname(16:23);
  time(3)=':';
  time(6)=':';
  dt=[date " " time];
  sigThresh = 0.0057;         # signal threshold for motion detect
  m = (data > sigThresh);
  efrac = nnz(m) / length(m);  # fraction of elements that are non-zero
  [ts, nchars] = strptime(dt,"%Y-%m-%d %H:%M:%S");
  epoch=mktime(ts);  # seconds since Unix epoch
  printf("%s, %d, %5.3f\n",dt,epoch,efrac);
  # doplot1(data,m,dt,fname);  # display the plot and save as .png
endfunction

function doplot1(data,m,dt,fname)    # show the graph, save and png file
  hf = figure();
  plot(data*100, "linewidth",2);
  axis([-Inf Inf 0 2]);  # set X-axis range
  grid on;
  hold on;

  plot(m/5, "linewidth",3);
  xlabel("(time, 5 minutes total)");
  ylabel("signal strength");
  plot_title = [ "Motion Plot: " dt " PDT"];
  title(plot_title);
  h = legend("raw signal", "event");
  legend(h, "location", "northwest");
  set (h, "fontsize", 12);
  dt(11)='_';  # change space between date/time to underscore for filename
  fout = [ "Plt_" fname(5:23) ".png" ];
  print (hf, fout, "-dpng", "-S1600,900");
  hold off;
  close;
endfunction

function proc(fname)    # downsample, filter and find signal envelope
  dratio = 5;   # decimation ratio
  wsize = 150; # boxcar filter size
  wsize1 = wsize * power(dratio,4);   # boxcar filter window size
  
  [yraw, fs] = audioread(fname);  # load raw sample data
  e = abs(hilbert(yraw));         # find envelope
  s1 = mean(e(1:wsize));
  e1 = [ s1*ones(wsize1,1)' e' ]';
  y = decimate(e1,dratio);
  y1 = decimate(y,dratio);
  y3 = decimate(y1,dratio);  
  y3a = movmean(y3,wsize);

  out1 = filter(ones(wsize,1)/wsize, 1, y3a); # do boxcar filter  
  out2 = decimate(out1,dratio);
  bufsize = wsize+1;
  out3 = out2(bufsize:end);  # remove front buffer added earlier
  doOutput(out3,fname);  # show the results  
endfunction

# main function here, process files in directory
files = dir('ChD_*flac');
for file = files';
  f = file.name;
  proc(f);             # process this file
end  

