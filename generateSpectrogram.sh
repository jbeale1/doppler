#!/bin/bash

# generate a spectrogram image of a portion of a file

# max 3 kHz, -45 dB upper limit, 35 dB dynamic range
# sox DpA_2019-06-04_21-45-00.mp3 -n remix 1 trim 55 20 rate 6k spectrogram -Z -45 -z 35 -o s45b.png
#  raw, and monochrome:
# sox DpA_2019-06-05_14-40-00.mp3 -n remix 1 trim 70 50 rate 6k spectrogram -m -r -Z -45 -z 35 -x 1500 -o s1440b.png

FILES="/media/john/Seagate4GB/MINIX-John/Doppler1"
REMOTE="john@john-fitlet2:/home/john/Audio/images"

for f in $FILES/*.mp3  # process all the .mp3 files
do
  [ -f "$f" ] || continue
  fout=${f%%.*}.png
  if [ ! -f "$fout" ]; then
    echo "writing $fout"
    fout_base=${fout##*/} # get basename from full path

    sox $f -n rate 8k spectrogram -Z -40 -z 28 -x 3000 -y 257 -m -r -o $fout

    mv $f $FILES/old  # after conversion, archive .mp3 file
    rcp $fout $REMOTE/$fout_base
    mv $fout $FILES/old  # archive .png
  fi
done
