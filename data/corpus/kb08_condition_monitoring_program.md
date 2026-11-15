---
title: Condition Monitoring Programme, Alarm Levels and Decision Making
type: maintenance guideline (project knowledge base)
---

# 1. Baselines and trending

An effective condition monitoring programme records a baseline for each machine after commissioning or after a bearing replacement, then measures at a fixed interval under comparable operating conditions (speed and load). Decisions are based on trends: a steady increase in RMS, kurtosis, high-frequency band energy or envelope amplitude at a defect frequency is more informative than any single absolute reading. Because vibration amplitude depends on speed and load, readings taken at different loads must be compared with care; data-driven models should be validated on load conditions not seen during training.

# 2. Alarm levels

Typical programmes use two alarm levels. An alert (warning) level is often set at about two to three times the baseline value of an indicator and triggers increased monitoring and a planned inspection. A danger level, often set at about four to six times baseline or at a machine-specific limit, triggers a recommendation to stop or replace at the earliest opportunity. For overall velocity vibration of the machine, the ISO 10816 / ISO 20816 series defines evaluation zones (A: newly commissioned, B: acceptable for long-term operation, C: unsatisfactory for long-term operation, D: severe enough to cause damage), with zone limits that depend on machine group, size and support type; the applicable part of the standard should be consulted for exact values.

# 3. Stages of bearing degradation

Bearing degradation is commonly described in four stages. Stage 1: ultrasonic and high-frequency acoustic emission only; standard vibration is normal. Stage 2: small defects excite bearing component natural frequencies; high-frequency energy and kurtosis increase. Stage 3: defect frequencies and harmonics appear clearly in the spectrum and envelope spectrum, often with sidebands; the bearing should be scheduled for replacement. Stage 4: broadband noise floor rises, discrete defect frequencies may disappear into random vibration, RMS and temperature increase; failure is near.

# 4. Decision support and human review

Automated diagnostic tools, including machine-learning classifiers and AI assistants, should be used as decision support. Their output must be reviewed by a qualified person before maintenance decisions are made, particularly when the tool reports low confidence, when the indicators disagree, or when the recommended action involves stopping a critical machine. Every recommendation should be traceable to the measured data and to the procedure or manual that supports it.
