# log data from radar module HLK-LD2415H
# J.Beale 28-Apr-2025

import serial
import time, os
from datetime import datetime

outPath = "slog1.csv"
sPort='/dev/ttyUSB0'  # serial port to log data from
logDir = '/home/john/Documents/doppler' # directory to log data in

# Configure the serial port settings
ser = serial.Serial(
    port = sPort,
    baudrate=9600,
    timeout=1            # Read timeout in seconds
)

now = datetime.now()
tsLog = now.strftime("%Y%m%d_%H%M%S_SerialLog.csv")
logfile = os.path.join(logDir, tsLog)

time.sleep(1)

# 434602 = cmdn, 0 = coming+going, 1 = 11 frames/sec, 0=km/hr units
config1 = "43 46 02 00 01 00 0d 0a" 
bytes = bytes.fromhex(config1)
ser.write(bytes) # send config command

try:
    with open(logfile,"w") as fout:
        fout.write("epoch,kmh\n")
        print("Logging %s to %s. Press Ctrl+C to stop." % (sPort, logfile))
        while True:
            line = ser.readline().decode('utf-8').strip()  # Read until \n, decode, and strip \r\n
            if line:
                if line.startswith('V'): # remove leading 'V'
                    line = line[1:]
                epoch = time.time()
                outs = (f"{epoch:.2f},{line}\n")
                print(outs,end="")
                fout.write(outs)
            
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()
