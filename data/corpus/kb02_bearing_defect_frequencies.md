---
title: Bearing Defect Frequencies and the CWRU Test Bearing
type: technical reference (project knowledge base)
---

# 1. Kinematic defect frequency formulas

For a bearing with n rolling elements, ball diameter d, pitch diameter D, contact angle phi and shaft rotation frequency fr = rpm / 60 (rotating inner ring, stationary outer ring), the characteristic defect frequencies are:

- Fundamental train (cage) frequency: FTF = (fr / 2) x (1 - (d / D) cos phi)
- Ball pass frequency, outer race: BPFO = (n x fr / 2) x (1 - (d / D) cos phi)
- Ball pass frequency, inner race: BPFI = (n x fr / 2) x (1 + (d / D) cos phi)
- Ball spin frequency: BSF = (D / 2d) x fr x (1 - ((d / D) cos phi)^2)

These formulas assume pure rolling without slip. In real machines the observed frequencies differ by 1 to 2 percent because of slip, so frequency matching should use a tolerance band of a few percent. BPFO plus BPFI equals n times the shaft frequency, which is a useful sanity check.

# 2. The CWRU drive-end test bearing (SKF 6205-2RS JEM)

The Case Western Reserve University (CWRU) bearing test rig uses a 2 hp induction motor, a torque transducer/encoder and a dynamometer. The drive-end bearing is a deep-groove ball bearing SKF 6205-2RS JEM with 9 balls, ball diameter 0.3126 inch (7.94 mm), pitch diameter 1.537 inch (39.04 mm) and a contact angle of zero.

Expressed as multiples of shaft speed, the drive-end defect frequencies are approximately: BPFI 5.415, BPFO 3.585, FTF 0.398 and BSF 2.357. At 1797 rpm (0 hp load, shaft frequency 29.95 Hz) this gives BPFI of about 162.2 Hz, BPFO of about 107.4 Hz and BSF of about 70.6 Hz. At 1730 rpm (3 hp load) BPFI is about 156.1 Hz and BPFO about 103.4 Hz.

# 3. CWRU seeded faults and operating conditions

Single-point faults were introduced with electro-discharge machining at diameters of 0.007, 0.014 and 0.021 inch (0.178, 0.356 and 0.533 mm) on the inner race, the rolling element and the outer race. Outer race faults were placed at the 3, 6 and 12 o'clock positions relative to the load zone; the 6 o'clock position is centred in the load zone and is the most commonly used. Each fault was recorded at motor loads of 0, 1, 2 and 3 hp with speeds of approximately 1797, 1772, 1750 and 1730 rpm.

Drive-end fault data are available at 12 kHz and 48 kHz sampling rates. The normal baseline recordings were sampled at 48 kHz, so they must be resampled to 12 kHz before being combined with 12 kHz fault data; otherwise a classifier can learn the sampling rate instead of the bearing condition.

# 4. Using defect frequencies in a diagnosis

A diagnosis is strongest when three independent indicators agree: (1) an elevated impulsiveness indicator such as kurtosis or crest factor, (2) envelope spectrum peaks at the matching defect frequency with harmonics, and (3) a data-driven classifier prediction with high confidence. When the classifier and the defect-frequency evidence disagree, the result should be treated as uncertain and confirmed by a qualified analyst before any maintenance action.
