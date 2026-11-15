---
title: How Bearing Faults Occur - Damage Mechanisms and Why Each Fault Vibrates Differently
type: technical explanation (project knowledge base)
---

# 1. How a rolling bearing carries load

A rolling bearing transfers the shaft load through very small contact areas between the rolling elements and the raceways. Because the contact area is tiny, the contact pressure is very high, typically between 1 and 4 GPa in a loaded ball bearing. This pressure is called Hertzian contact stress. The highest shear stress does not occur on the surface but a short distance below it, usually a few tenths of a millimetre under the raceway. Every time a ball rolls over a point on the raceway, that point is loaded and unloaded once, so a bearing running at 1800 rpm applies millions of stress cycles to its raceways every day.

Only part of the bearing carries load at any moment. Under a radial load the rolling elements at the bottom of a horizontal shaft carry most of the force; this region is called the load zone. A defect inside the load zone is hit hard, while the same defect outside the load zone is hit gently or not at all. The load zone is the key to understanding why the vibration of each fault type looks different.

# 2. Rolling contact fatigue: the natural way a bearing fails

Even a perfectly installed and lubricated bearing eventually fails by rolling contact fatigue. The repeated stress cycles below the surface start a microscopic crack, often at a small non-metallic inclusion in the steel. The crack grows slowly parallel to the surface and then turns up towards the raceway. When it reaches the surface, a small flake of material breaks out and leaves a cavity called a spall or pit. This process is called spalling or flaking, and ISO 15243 classifies it as subsurface-initiated fatigue.

The rated life of a bearing (L10 life) is the number of revolutions that 90 percent of a group of identical bearings will complete before the first sign of fatigue spalling. Real bearings rarely reach this natural fatigue limit, because most failures are premature and are caused by the operating conditions.

# 3. Causes of premature bearing damage

Surface-initiated fatigue happens when the lubricant film is too thin to separate the surfaces, so the asperities of the ball and the raceway touch. Insufficient lubricant, the wrong viscosity, degraded grease, high temperature or too long a relubrication interval all thin the film and start micro-pitting on the surface.

Contamination is one of the most common causes of early failure. Hard particles such as sand, metal chips or grinding dust are over-rolled and press dents into the raceways. Each dent has raised edges that concentrate stress, so a spall often starts just behind a dent. Water or moisture causes corrosion pits that act in the same way.

Incorrect mounting damages a bearing before it ever runs. Hammer blows transmitted through the rolling elements, or forcing the bearing onto the shaft through the wrong ring, leave indentations at the ball spacing called true brinelling. Vibration while the machine is stopped can wear similar marks called false brinelling.

Misalignment and overload concentrate the load on one edge of the raceway or on fewer rolling elements, which raises the contact stress and shortens fatigue life sharply. An interference fit that is too tight removes internal clearance and has the same effect.

Electric current passing through the bearing, for example from a variable frequency drive without shaft grounding, causes tiny arcs across the lubricant film. These arcs melt small craters and eventually produce a washboard pattern on the raceway called fluting.

# 4. How a defect produces vibration

When a rolling element passes over a spall, it drops into the cavity and strikes the far edge. Each strike is a short mechanical impact lasting well under a millisecond. The impact excites the natural resonances of the bearing, the housing and the sensor mount, which typically lie between 1 kHz and 10 kHz, so each impact appears in the vibration signal as a short burst of ringing that decays quickly.

Because the shaft turns at a constant speed and the balls are evenly spaced, the impacts repeat at a regular rate that depends on where the defect is. This repetition rate is the bearing defect frequency: BPFO for the outer race, BPFI for the inner race and BSF for a rolling element. The defect frequencies themselves are low, usually between 50 Hz and 300 Hz, but the energy of the impacts sits at the high resonance frequencies.

Envelope analysis recovers the hidden repetition rate. The signal is band-pass filtered around the resonance, the envelope (the outline of the bursts) is calculated with the Hilbert transform, and a spectrum of the envelope is taken. Peaks at the defect frequency and its harmonics in the envelope spectrum show that the bursts repeat at that rate, which identifies the faulty component.

Statistical indicators react to the impacts as well. Sharp bursts on a quiet background raise the kurtosis well above the value of 3 that a healthy, Gaussian-like signal shows, and they raise the crest factor, which is the ratio of peak to RMS. As damage spreads, the impacts merge, RMS rises and kurtosis can fall again, so a falling kurtosis is not always a sign of improvement.

# 5. Why an outer race fault looks the way it does

The outer ring is normally stationary in the housing, so an outer race defect stays at the same position. If the defect lies in the load zone, every rolling element that passes over it strikes it with about the same force. The result is a train of evenly spaced impacts of nearly constant amplitude at BPFO. The envelope spectrum shows clear peaks at BPFO and its harmonics without strong sidebands, which is the classic and easiest bearing fault to detect. In the CWRU data the outer race faults were placed at the 6 o'clock position, which is centred in the load zone, so they give strong BPFO lines.

# 6. Why an inner race fault looks the way it does

The inner ring rotates with the shaft, so an inner race defect travels around with it. Once per shaft revolution the defect passes through the load zone, where the impacts are strong, and then moves out of it, where they become weak. The impact amplitude is therefore modulated at the shaft rotation frequency. In the envelope spectrum this produces peaks at BPFI with sidebands spaced at 1x shaft speed on each side. The impacts are also transmitted to the sensor through more interfaces, the rolling elements and the outer ring, so they arrive weaker than outer race impacts and are often harder to detect at an early stage. Inner race damage tends to progress quickly because the defect is loaded on every revolution.

# 7. Why a ball fault looks the way it does

A defect on a rolling element spins with the ball and strikes the inner race and the outer race alternately. It therefore produces impacts at twice the ball spin frequency (2 x BSF). The ball also travels around the bearing with the cage, moving in and out of the load zone, so the impacts are modulated at the cage frequency (FTF) and the envelope spectrum shows sidebands at FTF. A ball can also roll so that the defect does not touch the races on every turn, which makes the impacts irregular. For these reasons ball faults are the hardest bearing faults to detect with envelope analysis.

# 8. Stages of bearing degradation

Bearing failure develops in stages. In stage 1 the first subsurface damage creates only ultrasonic emission and is not visible in normal vibration spectra. In stage 2 small defects begin to ring the bearing resonances, so high-frequency energy, kurtosis and envelope readings rise. In stage 3 the defect frequencies and their harmonics appear clearly in the envelope and velocity spectra, and sidebands grow as the damage spreads. In stage 4 the discrete frequencies are replaced by a raised broadband noise floor, RMS and temperature rise quickly, and the bearing is close to failure. Trending these indicators over successive measurements shows how fast a bearing is moving through the stages and how much time is left to plan a replacement.

# 9. Seeded faults in test rigs compared with real faults

Laboratory datasets such as CWRU use seeded faults made by electro-discharge machining, which cuts a clean round pit of a known diameter into the raceway or ball. A seeded pit gives sharp, repeatable impacts, which makes it easier to classify than natural fatigue damage. Natural spalls have irregular edges, grow over time and are usually accompanied by wear, contamination and lubrication problems, so real signals are noisier. A model trained only on seeded faults should therefore be checked on new data before it is trusted on a different machine.
