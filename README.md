## Doppler Radar object detection and speed measurement

This project uses cheap doppler radar modules: low power, short range motion detectors intended for automatic door openers and similar uses. The HB100 at 10.525 GHz or the smaller, more directional CDM324 at 24 GHz are available on eBay under $10 and Amazon for a bit more. They generate a weak (microphone-level) audio signal output, with a frequency proportional to the speed of the detected object. They can detect people to 25+ feet and cars to 100+ feet. Watch [this video](https://www.youtube.com/watch?v=r2NwtRPYWK4) to hear the signal as cars pass. A simple aluminum-foil horn antenna reduces sensitivity towards the side and back, and extends this range somewhat. Below is a CDM324 sensor in a small 3D printed conical horn, before applying the copper-foil tape for the reflector. The  sensor PCB is about 1 inch square. The horn front opening is 3 inches in diameter and when assembled, it picks up SUVs or trucks around 200 feet, somewhat less for small cars.

![CDM324_horn](https://github.com/jbeale1/doppler/blob/master/CDM324-horn-system.jpg)

When connecting a circuit, be careful because the output pin has a DC bias, and is static sensitive, and they will usually not survive being connected incorrectly. These modules run from +5V but there is no rejection of power-supply noise whatsoever, so it may deliver no useful signal with +5V straight from a USB or switching power supply. I suspect that is why many people quickly give up on them, not realizing how critical it is to have RF-clean power.  It works OK on battery power (4 NiMH AA cells is close enough to 5V), or a low-noise, RF-filtered supply. I start with a cheap noisy 5V->12V booster ("KUNCAN 5ft USB 5v to DC 12v Step Up") and put that 12V through several LC filters and ferrite beads, into a LM78L05 linear regulator to deliver reasonably clean +5V. The CDM324 draws around 30 mA at 5V.  Note the modules have a simple diode detector that will also respond to any strong local RF, so keep it away from your wifi hub and cellphone. There is a clear description of how the CDM324 works at [The Signal Path](http://thesignalpath.com/blogs/2018/08/12/tutorial-experiment-teardown-of-a-24ghz-doppler-radar-module/).

You can run the weak doppler output signal through a preamp (eg. opamp OPA209A, gain of 200x = 46 dB) and into a USB audio adaptor to get data into a Raspberry Pi (or any old phone/tablet/computer). On a Pi or other Linux based machine, **ffmpeg** can record the frequency signals into mp3 or wave files, and **sox** can generate a spectrogram. A car passing by at a constant speed will produce a curved line on the spectogram, because the signal is proportional to the relative *radial* velocity of the car at any given time. In other words, that part of the total velocity going directly towards or away from the sensor. 

For example, if the radar sensor sits by the side of the road looking to the right, and a car approaches from the right and then passes, at first the car's motion is almost directly towards the sensor (radial velocity ~ total velocity) but then radial velocity drops to zero at the moment it draws even with the sensor and passes by. After that its radial velocity goes negative as it receeds, but meanwhile it has also passed behind the sensor and out of the radar antenna pattern, so the signal fades out at some point before reaching a frequency of 0 (not to mention the preamp is AC-coupled). On a two-lane road, signals from oncoming traffic in the far lane fade into the noise sooner than with-traffic in the near lane, since cars in the far lane remain farther away as they pass.

![CarDopplerSignal](https://github.com/jbeale1/doppler/blob/master/DopplerSignal1.jpg)

These cheap radar sensors do not have an I+Q output but just a magnitude output, so you cannot directly tell apart approaching and receeding velocity. However if the sensor is pointed along the road, cars passing by towards the right and left will generate an "L" shaped curve in the spectrum with the horizontal part pointing right and left respectively, as they pass the sensor either at the start or the end of the time they are inside the sensor's field of view (see illustration above). In this way you can still tell the car's direction.  This assumes the car's speed is roughly constant; it can be harder to tell if a car slows down and stops while inside the field of view. Below is the signal from an approaching car which slows down nearly to a stop, then speeds up as it passes by. The first dip and rise is due to the car's actual speed changing, while the final dropoff is only from the angle relative to the sensor.

![SlowerThenFaster](https://github.com/jbeale1/doppler/blob/master/D_SlowFast.jpg)

If you had two sensors positioned back-to-back on the side of the road, so one was looking to the right and the other to the left, it would be easier to separate a change in speed from a change in heading angle. First one sensor would detect an approaching car until it passed by, and then the other sensor would see it receeding. Below is an illustration of this, where the two sensor outputs are color-coded and combined into one image. The true velocity of the car is most likely to be a smooth curve that bridges above the gap between the red and green curves (right- and left-facing sensor outputs).

![TwoSensors](https://github.com/jbeale1/doppler/blob/master/TwoSensors_1.jpg)

Walkers or joggers generate a distictive wavy pattern (FM signal) in the spectrogram from the ~2 Hz motion of their arms and legs, and of course they move much slower than cars. As cars pass nearby the sensor, reflections from different parts of the rotating wheels generate some velocity spread. The top of the wheel is moving 2X the car speed, while the bottom of the wheel is momentarily near 0 speed. All such effects depend on reflectivity of the various parts so it is less visible on wheel hubs with a smoother profile.  If a vehicle is very close to the sensor while another one passes, you may get a signal from more than one angle at once, as part of the radar energy ping-pongs off one surface and then another.

Given that the green signal is from the left-pointing sensor, you can see below a person walking briskly (4 mph) goes from left to right, followed 50 seconds later by a truck going the same way. The truck reaches 19 mph but is slowing down as it departs to the right.  There is a spike in the pedestrian signal from both sensors just as he passes by. He was holding a cellphone in right hand, and this might have been receiver overload from a momentary glint reflecting off the flat metal case.

![Walk-Truck](https://github.com/jbeale1/doppler/blob/master/D_Walk-Truck.jpg)
