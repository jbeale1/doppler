## Doppler Radar object detection and speed measurement

This project uses cheap (< $10) doppler radar modules from various online vendors, such as the HB100 at 10.525 GHz or the smaller CDM324 at 24 GHz. These modules are low power, short range, and intended for automatic door openers and similar purposes. They generate a weak (microphone-level) audio signal output. They can detect cars to 100+ feet and people to 30+ feet. A simple aluminum-foil horn antenna reduces sensitivity towards the side and back, and extends this range somewhat.

When connecting a circuit, be careful because the output pin has a DC bias, and is static sensitive, and they will usually not survive being connected incorrectly. Both modules run from +5V but there is no rejection of power-supply noise whatsoever so it may deliver no useful signal with +5V straight from a USB or switching power supply. I suspect that is why many people quickly give up on these, not realizing how critical it is to have RF-clean power.  It works on battery (4 NiMH AA cells is close enough to 5V), or a low-noise, RF-filtered supply. I use a cheap noisy 5V->12V booster ("KUNCAN 5ft USB 5v to DC 12v Step Up"), through several LC filters and ferrite beads, into a LM78L05 linear regulator back to +5V.  Note the modules have a simple diode detector that will also respond to any strong local RF, so keep it away from your wifi hub and cellphone.

You can run the weak doppler output signal through a preamp and into a USB audio adaptor (~$10) to get data into a Raspberry Pi (or any old phone/tablet/computer). On a Pi or other Linux based machine, 'arecord' can record the frequency signals into mp3 or wave files, and 'sox' can generate a spectrogram. A car passing by will produce a curved line on the spectogram. Remember the points on the curve are proportional to the *radial* velocity of the car at any given time, meaning velocity directly towards or away from the sensor. For example if the radar sensor sits by the side of the road pointing to the right, and a car passes by the sensor going to the right, the radial velocity will start at 0 as the car is just even with the sensor and its velocity is at right angles to the sensor, but then quickly rises as the car receeds and is moving more directly away from the sensor.

These cheap radar sensors do not have an I+Q output but just a magnitude output, so you cannot directly tell apart approaching and receeding velocity. However if the sensor is pointed along the road, cars passing by going the left and right will generate an "L" shaped curve in the spectrum pointing left and right respectively, as they pass the sensor either at the start or the end of the time inside the field of view. In this way you can still infer the car's direction.  However this does become ambiguous if a car slows down and stops inside the field of view.  

Walkers or joggers generate a distictive wavy pattern (FM signal) in the spectrogram from the ~2 Hz motion of their arms and legs, and of course they move much slower than cars. As cars pass nearby the sensor, reflections from different parts of the rotating wheels generate some velocity spread. The top of the wheel is moving 2X the car speed, while the bottom of the wheel is momentarily near 0 speed. All such effects depend on reflectivity of the various parts so it is less visible on wheel hubs with a smoother profile.  If a vehicle is very close to the sensor, you may see other cars passing from different angles as the signal bounces off first one car and then another.
