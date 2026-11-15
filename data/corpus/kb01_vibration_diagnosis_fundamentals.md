---
title: Vibration-Based Diagnosis of Rolling Element Bearings - Fundamentals
type: technical guide (project knowledge base, written by the IntelliMaintain team from standard condition-monitoring practice)
---

# 1. Why bearings fail and why vibration is monitored

Rolling element bearings support rotating shafts in motors, pumps, fans, compressors and gearboxes. Even a correctly selected bearing eventually fails by rolling contact fatigue, but most premature failures are caused by inadequate lubrication, contamination, misalignment, incorrect mounting, overloading or electrical current passing through the bearing. A localized defect such as a spall, pit or crack on the outer race, inner race or a rolling element produces a short mechanical impact every time a rolling element passes over it. These impacts excite structural resonances of the bearing housing at high frequency (typically 1 kHz to 10 kHz), which is why accelerometers mounted on the housing are the standard sensor for bearing condition monitoring.

Vibration monitoring detects bearing damage weeks or months before functional failure. The interval between the first detectable symptom (point P) and functional failure (point F) is called the P-F interval. Inspection and data-collection frequency must be shorter than the P-F interval, otherwise the defect can progress to failure between two measurements.

# 2. Time-domain condition indicators

Root mean square (RMS) measures the overall energy of the vibration signal. RMS rises as damage grows, but it is insensitive to early, small localized defects because the impacts are short and carry little energy.

Kurtosis measures how impulsive (peaky) a signal is. A healthy bearing produces approximately Gaussian vibration with a kurtosis close to 3. Localized defects produce periodic impulses and raise kurtosis well above 3; values above about 4 to 5 indicate impulsive content and values above 8 to 10 indicate strong impacting. High kurtosis is therefore an early indicator of a localized race or rolling-element defect. Kurtosis can fall again at an advanced stage when damage spreads around the race and the signal becomes more random, so kurtosis must be trended together with RMS.

Crest factor is the ratio of peak amplitude to RMS. A sine wave has a crest factor of 1.41 and Gaussian noise typically 3 to 4. Crest factor values above about 5 to 6 indicate impacting. Like kurtosis, crest factor rises early and may fall in the final stage of degradation when RMS increases strongly.

Skewness describes asymmetry of the amplitude distribution. It is less widely used but can change when impacts occur preferentially in one direction, for example because of load-zone effects.

# 3. Frequency-domain indicators and FFT band energy

The fast Fourier transform (FFT) converts a time window into a spectrum. Band energy features divide the spectrum into frequency bands and measure the share of energy in each band. Bearing defects typically increase energy in the high-frequency bands where housing resonances are excited, while unbalance and misalignment increase energy at 1x and 2x running speed. A rising share of energy above 2 kHz is a typical early sign of bearing damage or lubrication problems.

# 4. Envelope analysis (demodulation)

Envelope analysis is the most reliable classical method to confirm a bearing defect. The signal is band-pass filtered around a resonance band excited by the impacts, the envelope is computed (for example with the Hilbert transform), and the spectrum of the envelope is calculated. Peaks in the envelope spectrum at a bearing defect frequency and its harmonics identify the damaged component: BPFO for the outer race, BPFI for the inner race, two times BSF for a rolling element, and FTF for the cage. Inner race defects usually show sidebands spaced at shaft speed around BPFI because the defect moves in and out of the load zone once per revolution. Outer race defects on a stationary outer ring usually show clean BPFO harmonics without shaft-speed sidebands.

Rolling element (ball) defects are the hardest to confirm with envelope analysis because the ball spins and the defect does not strike a race on every revolution. Published benchmark studies on the Case Western Reserve University dataset report that many ball-fault recordings do not show clear ball-spin frequency components.
