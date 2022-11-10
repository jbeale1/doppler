#!/usr/bin/env python

# parse data stream from "HLK-LD303-24G Millimeter Wave Ranging Radar Sensor Module LD303"
# for example, from https://www.aliexpress.us/item/3256803374930067.html
# j.beale  9-Nov-2022

import serial  # uses pyserial library, 'pip install pyserial'
import datetime

with serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1) as ser:
    while True:
        dat = ser.read(26)        # read up to 26 bytes (timeout)
        if (len(dat)>10 and dat[0] == 0x55 and dat[1] == 0xa5 and dat[2] == 0x0a and dat[3] == 0xd3):
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
Example Output:
2022-11-09 21:14:20 dist = 00000, t = 0, sig= 01032, 0, 55a50ad30000000004080000e3
2022-11-09 21:14:20 dist = 00000, t = 0, sig= 00590, 0, 55a50ad300000000024e000027
2022-11-09 21:14:20 dist = 00000, t = 0, sig= 02159, 0, 55a50ad300000000086f00004e
2022-11-09 21:14:20 dist = 00124, t = 1, sig= 04393, 0, 55a50ad3007c0001112900008e
2022-11-09 21:14:20 dist = 00123, t = 1, sig= 07230, 0, 55a50ad3007b00011c3e0000ad
2022-11-09 21:14:20 dist = 00122, t = 1, sig= 04498, 0, 55a50ad3007a000111920000f5
2022-11-09 21:14:21 dist = 00121, t = 1, sig= 06876, 0, 55a50ad3007900011adc000047
2022-11-09 21:14:21 dist = 00119, t = 1, sig= 07178, 0, 55a50ad3007700011c0a000075
2022-11-09 21:14:21 dist = 00114, t = 1, sig= 08485, 0, 55a50ad3007200012125000090
2022-11-09 21:14:21 dist = 00099, t = 1, sig= 08753, 0, 55a50ad300630001223100008e
2022-11-09 21:14:21 dist = 00081, t = 1, sig= 04683, 0, 55a50ad300510001124b000086
2022-11-09 21:14:21 dist = 00069, t = 1, sig= 07623, 0, 55a50ad3004500011dc7000001
2022-11-09 21:14:21 dist = 00063, t = 1, sig= 06150, 0, 55a50ad3003f00011806000035
2022-11-09 21:14:21 dist = 00060, t = 1, sig= 06227, 0, 55a50ad3003c0001185300007f
2022-11-09 21:14:21 dist = 00058, t = 1, sig= 01735, 0, 55a50ad3003a000106c70000df
2022-11-09 21:14:22 dist = 00056, t = 1, sig= 01643, 0, 55a50ad300380001066b000081
2022-11-09 21:14:22 dist = 00053, t = 1, sig= 01711, 0, 55a50ad30035000106af0000c2
2022-11-09 21:14:22 dist = 00049, t = 1, sig= 00516, 0, 55a50ad300310001020400000f
2022-11-09 21:14:22 dist = 00051, t = 1, sig= 00524, 0, 55a50ad300330001020c000019

"""
