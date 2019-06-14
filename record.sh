#!/bin/bash

# use ffmpeg to continually record audio files

# location to store audio files
todir="/run/shm/audio/"

# number of seconds each .mp3 file should last
duration="300"

# which audio device card number to work with
cnum="1"

# set mic input level to ~ 0 dB (which is strangely 15 out of 100)
amixer -c "$cnum" cset numid=3 15

# Record fixed-length MP3 files from audio device continually
ffmpeg -nostdin -loglevel quiet -f alsa -ac 1 -ar 22050 -i plughw:"$cnum" -map 0:0 -acodec libmp3lame \
  -b:a 128k -f segment -strftime 1 -segment_time "$duration" -segment_atclocktime 1 \
   "$todir"DpA_%Y-%m-%d_%H-%M-%S.mp3 &

sudo renice -n -20 $!
