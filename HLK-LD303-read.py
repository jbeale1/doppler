#!/usr/bin/env python3

# parse data stream from "HLK-LD303-24G Millimeter Wave Ranging Radar Sensor Module LD303"
# for example, from https://www.aliexpress.us/item/3256803374930067.html
# maximum observed range for a person is about 2 meters
# j.beale  9-Nov-2022

import serial  # uses pyserial library, 'pip install pyserial'
import datetime

with serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1) as ser:
    while True:
        dat = ser.read(26)        # read up to 26 bytes (timeout)
        bcount = len(dat)         # how many actually got read
        if (bcount==13 and dat[0] == 0x55 and dat[1] == 0xa5 and dat[2] == 0x0a and dat[3] == 0xd3):
            cm = 256*dat[4] + dat[5] # 2-byte distance in cm  (dat[6] always 0)
            target = dat[7]            # target present, 1=yes
            signal = 256*dat[8] + dat[9]  # 2-byte signal strength
            mm = dat[10]                # 1-byte micro-movement flag, 1 = yes
            if (signal > 0):
                now = datetime.datetime.now()
                timeString = now.strftime('%Y-%m-%d %H:%M:%S')
                print("%s dist = %05d, t = %01d, sig= %05d, %01d" % (timeString, cm, target, signal, mm),end=", ")
                print(dat.hex())
        

"""
Example Output for person approaching sensor
2022-11-09 21:27:28 dist = 00000, t = 0, sig= 00554, 0, 55a50ad300000000022a000003
2022-11-09 21:27:28 dist = 00000, t = 0, sig= 01502, 0, 55a50ad30000000005de0000ba
2022-11-09 21:27:28 dist = 00208, t = 1, sig= 01269, 0, 55a50ad300d0000104f50000a1
2022-11-09 21:27:28 dist = 00203, t = 1, sig= 02412, 0, 55a50ad300cb0001096c000018
2022-11-09 21:27:29 dist = 00200, t = 1, sig= 02421, 0, 55a50ad300c80001097500001e
2022-11-09 21:27:29 dist = 00197, t = 1, sig= 03659, 0, 55a50ad300c500010e4b0000f6
2022-11-09 21:27:29 dist = 00191, t = 1, sig= 03310, 0, 55a50ad300bf00010cee000091
2022-11-09 21:27:29 dist = 00185, t = 1, sig= 03893, 0, 55a50ad300b900010f350000d5
2022-11-09 21:27:29 dist = 00175, t = 1, sig= 03740, 0, 55a50ad300af00010e9c000031
2022-11-09 21:27:29 dist = 00161, t = 1, sig= 04441, 0, 55a50ad300a1000111590000e3
2022-11-09 21:27:29 dist = 00148, t = 1, sig= 04269, 0, 55a50ad30094000110ad000029
2022-11-09 21:27:29 dist = 00139, t = 1, sig= 05435, 0, 55a50ad3008b0001153b0000b3
2022-11-09 21:27:29 dist = 00133, t = 1, sig= 10303, 0, 55a50ad300850001283f0000c4
2022-11-09 21:27:30 dist = 00124, t = 1, sig= 10349, 0, 55a50ad3007c0001286d0000e9
2022-11-09 21:27:30 dist = 00108, t = 1, sig= 07399, 0, 55a50ad3006c00011ce7000047
2022-11-09 21:27:30 dist = 00091, t = 1, sig= 09393, 0, 55a50ad3005b000124b1000008
2022-11-09 21:27:30 dist = 00080, t = 1, sig= 09368, 0, 55a50ad30050000124980000e4
2022-11-09 21:27:30 dist = 00071, t = 1, sig= 12038, 0, 55a50ad3004700012f06000054
2022-11-09 21:27:30 dist = 00062, t = 1, sig= 07171, 0, 55a50ad3003e00011c03000035
2022-11-09 21:27:30 dist = 00054, t = 1, sig= 06986, 0, 55a50ad3003600011b4a000073
2022-11-09 21:27:30 dist = 00047, t = 1, sig= 01946, 0, 55a50ad3002f0001079a0000a8
2022-11-09 21:27:30 dist = 00042, t = 1, sig= 01115, 0, 55a50ad3002a0001045b000061
2022-11-09 21:27:31 dist = 00035, t = 1, sig= 01476, 0, 55a50ad30023000105c40000c4
2022-11-09 21:27:31 dist = 00028, t = 1, sig= 02389, 0, 55a50ad3001c00010955000052
2022-11-09 21:27:31 dist = 00023, t = 1, sig= 02240, 0, 55a50ad30017000108c00000b7
2022-11-09 21:27:31 dist = 00022, t = 1, sig= 02142, 0, 55a50ad300160001085e000054
2022-11-09 21:27:31 dist = 00022, t = 1, sig= 02189, 0, 55a50ad300160001088d000083
"""
